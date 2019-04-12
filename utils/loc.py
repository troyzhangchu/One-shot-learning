resnet_model_dir = "/srv/BigData/Chuan_data/results/resnet/version/model_best.pth.tar"
VGG_model_dir = "/srv/BigData/Chuan_data/results/VGG/version/model_best.pth.tar"
densenet_model_dir = "/srv/BigData/Chuan_data/results/densenet/version/model_best.pth.tar"
inception_model_dir = "/srv/BigData/Chuan_data/results/inception/version/model_best.pth.tar"

train_dir = "/srv/BigData/Chuan_data/mini_imagenet/mini-imagenet-v2/pretrain_data/train"
val_dir = "/srv/BigData/Chuan_data/mini_imagenet/mini-imagenet-v2/pretrain_data/val"
test_dir = "/srv/BigData/Chuan_data/mini_imagenet/mini-imagenet-v2/test"

siamese_dir = "/srv/BigData/Chuan_data/results/siamese"
resnet_dir = "/srv/BigData/Chuan_data/results/resnet"
VGG_dir = "/srv/BigData/Chuan_data/results/VGG"
densenet_dir = "/srv/BigData/Chuan_data/results/densenet"
inception_dir = "/srv/BigData/Chuan_data/results/inception"

params = {
    "cuda": True,
    "feature_extractor_name": "dense",
    "margin": 10,
    "epoch": 5,
    "pretrain": False,
    "mode": "train",
    "version": None,
    "p": None,
    "batch_size": 48,
    "fine_tune_resnet": True
}
