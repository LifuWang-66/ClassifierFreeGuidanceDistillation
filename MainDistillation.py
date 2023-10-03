from DiffusionFreeGuidence.TrainDistillation import train, eval, compare


def main(model_config=None):
    modelConfig = {
        "state": "compare", # or eval
        "epoch": 70,
        "batch_size": 100,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "num_class": 10,
        "w": 8,
        "teacher_save_dir": "./CheckpointsCondition/",
        "distillation_save_dir": "./CheckpointsDistillation/",
        "distillation_training_load_weight": None,
        "distillation_test_load_weight": "42",
        "teacher_test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledDistillationGuidenceImgs.png",
        "nrow": 10
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    else:
        compare(modelConfig)

if __name__ == '__main__':
    main()
