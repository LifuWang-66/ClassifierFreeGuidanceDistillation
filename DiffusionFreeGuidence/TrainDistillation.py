import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

from DiffusionFreeGuidence.DiffusionDistillation import GaussianDiffusionDistillationSampler, GaussianDiffusionDistillationTrainer
from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler
from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.ModelDistillation import DistillationUNet
from Scheduler import GradualWarmupScheduler
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = DistillationUNet(T=modelConfig["T"], num_labels=modelConfig["num_class"], W=modelConfig["w"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["distillation_training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["distillation_save_dir"], "ckpt_" + modelConfig["distillation_training_load_weight"] + "_.pt"), map_location=device), strict=False)
        print("Model load weight done.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    
    teacher = UNet(T=modelConfig["T"], num_labels=modelConfig["num_class"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    ckpt = torch.load(os.path.join(
        modelConfig["teacher_save_dir"], modelConfig["teacher_test_load_weight"]), map_location=device)
    teacher.load_state_dict(ckpt)
    print("Teacher load weight done.")
    teacher.eval()

    trainer = GaussianDiffusionDistillationTrainer(
        teacher, net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], W=modelConfig["w"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                loss = trainer(x_0, labels).sum() / b ** 2.
                # start = time.time()
                loss.backward()
                # print("backprop: ", time.time() - start)
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["distillation_save_dir"], 'ckpt_' + str(e) + "_.pt"))
        
def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
    batch_size = 2000 # You can adjust this as needed
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_images, train_labels = next(iter(dataloader))
    real_images = real_images.to(device)
    fake_images = []

    # load model and evaluate
    with torch.no_grad():
        for i in range(20):
            step = int(modelConfig["batch_size"] // 10)
            labelList = []
            k = 0
            for i in range(1, modelConfig["batch_size"] + 1):
                labelList.append(torch.ones(size=[1]).long() * k)
                if i % step == 0:
                    if k < 10 - 1:
                        k += 1
            labels = torch.cat(labelList, dim=0).long().to(device) + 1
            print("labels: ", labels)
            model = DistillationUNet(T=modelConfig["T"], num_labels=modelConfig["num_class"], W=modelConfig["w"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
            ckpt = torch.load(os.path.join(
                modelConfig["distillation_save_dir"], "ckpt_" + modelConfig["distillation_test_load_weight"] + "_.pt"), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionDistillationSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            save_image(saveNoisy, os.path.join(
                modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
            print(torch.randint(modelConfig["w"], size=(noisyImage.shape[0], )).shape)
            sampledImgs = sampler(noisyImage, labels, w = torch.randint(modelConfig["w"], size=(noisyImage.shape[0], ), device=device))
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            save_image(sampledImgs, os.path.join(
                modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
            fake_images.append(sampledImgs)
        
        
        fake_images = torch.cat(fake_images, dim=0)
        fid = FrechetInceptionDistance(normalize=True)
        fid.to(device)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        print(f"FID: {float(fid.compute())}")

def compare(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # load model and evaluate
    with torch.no_grad():
        model = DistillationUNet(T=modelConfig["T"], num_labels=modelConfig["num_class"], W=modelConfig["w"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["distillation_save_dir"], "ckpt_" +modelConfig["distillation_test_load_weight"] + "_.pt"), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionDistillationSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        
        teacher_model = UNet(T=modelConfig["T"], num_labels=modelConfig["num_class"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["teacher_save_dir"], modelConfig["teacher_test_load_weight"]), map_location=device)
        teacher_model.load_state_dict(ckpt)
        print("teacher load weight done.")
        teacher_model.eval()
        
        # image_list = []
        # for i in range (modelConfig["num_class"]):
        #     teacher_list = []
        #     student_list = []
        #     label = torch.tensor([i + 1]).to(device)
        #     for j in range(modelConfig["w"]):
        #         noisyImage = torch.randn(
        #             size=[1, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        #         teacher_sampler = GaussianDiffusionSampler(
        #             teacher_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=j).to(device)
        #         teacher_sampledImgs = teacher_sampler(noisyImage, label)
        #         teacher_sampledImgs = teacher_sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        #         teacher_list.append(teacher_sampledImgs)

        #         # Sampled from standard normal distribution
        #         noisyImage = torch.randn(
        #             size=[1, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        #         sampledImgs = sampler(noisyImage, label, w = torch.tensor([j]).to(device))
        #         sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        #         student_list.append(sampledImgs)
        #     class_pair = torch.cat(teacher_list + student_list, dim = 0)
        #     image_list.append(class_pair)
        # image_list = torch.cat(image_list, dim = 0)
        torch.manual_seed(43)

        wList = []
        for i in range(modelConfig["w"]):
            curr_class = torch.ones(modelConfig["num_class"]) * i
            wList.append(curr_class)
        wList = torch.cat(wList, dim=0).to(torch.int64).to(device)

        labelList = []
        for i in range(modelConfig["w"]):
            curr_class = torch.arange(modelConfig["num_class"]) + 1
            labelList.append(curr_class)
        labelList = torch.cat(labelList, dim=0).to(device)
        noisyImage = torch.randn(
                size=[modelConfig["w"] * modelConfig["num_class"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        sampledImgs = sampler(noisyImage, labelList, wList)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  "student.png"), nrow=modelConfig["num_class"])
        

        teacher_sample_list = []
        for i in range(modelConfig["w"]):
            labels = torch.arange(modelConfig["num_class"]) + 1
            labels = labels.to(device)
            # teacher_noisyImage = torch.randn(
            #     size=[labels.shape[0], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
            teacher_noisyImage = noisyImage[i*modelConfig["num_class"]: (i+1) * modelConfig["num_class"], :, :, :]
            teacher_sampler = GaussianDiffusionSampler(
                teacher_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=i).to(device)
            teacher_sampledImgs = teacher_sampler(teacher_noisyImage, labels)
            teacher_sampledImgs = teacher_sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            teacher_sample_list.append(teacher_sampledImgs)
        teacher_sample_list = torch.cat(teacher_sample_list, dim=0)       
        save_image(teacher_sample_list, os.path.join(
            modelConfig["sampled_dir"],  "teacher.png"), nrow=modelConfig["num_class"])

                