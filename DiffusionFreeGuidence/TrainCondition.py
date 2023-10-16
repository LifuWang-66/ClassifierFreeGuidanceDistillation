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

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler
import os

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
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
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
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
    batch_size = 2000 # You can adjust this as needed
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_images, train_labels = next(iter(dataloader))
    real_images = real_images.to(device)
    # test FID
    # fid = FrechetInceptionDistance(normalize=True)
    # fid.to(device)
    # fid.update(real_images.to(device), real=True)
    # fid.update(next(iter(dataloader))[0].to(device), real=False)
    # print(f"FID: {float(fid.compute())}")
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
            model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
            ckpt = torch.load(os.path.join(
                modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            save_image(saveNoisy, os.path.join(
                modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
            # sampledImgs = sampler(noisyImage, labels)
            # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            # save_image(sampledImgs, os.path.join(
            #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
            # fake_images.append(sampledImgs)

            ddim_sampler = DDIMSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
            ddim_sampledImgs = ddim_sampler(noisyImage, labels)
            ddim_sampledImgs = ddim_sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            save_image(ddim_sampledImgs, os.path.join(
                modelConfig["sampled_dir"],  "ddim_sampledImgName.png"), nrow=modelConfig["nrow"])

        
        fake_images = torch.cat(fake_images, dim=0)
        fid = FrechetInceptionDistance(normalize=True)
        fid.to(device)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        print(f"FID: {float(fid.compute())}")