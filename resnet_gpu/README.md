# MindSpore ResNet-50 Tutorial with GPU backend
This is a tutorial for training MindSpore ResNet-50 model to classifying mushrooms.

> **NOTICE:** The codebase of this tutorial is developed based on `v1.0` MindSpore [ModelZoo](https://github.com/mindspore-ai/mindspore/tree/r1.0/model_zoo/official/cv/resnet).

## Guidelines

### Install some system packages

* System package

    ```
    sudo apt install -y unzip
    ```

* MindSpore (**v1.0**)

    For MindSpore installation, please refer to [MindSpore install page](https://www.mindspore.cn/install).

### Download source code

```
git clone https://github.com/leonwanghui/ms-resnet50-mushroom.git
cd ms-resnet50-mushroom/
```

### Download mushroom dataset

```
cd mushroom-dataset/ && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip
unzip mushrooms.zip
cd ../resnet_gpu/
```

### Model training

```
cd ./scripts/ && bash run_standalone_train_gpu.sh resnet50 imagenet2012 ../../mushroom-dataset/train
# Check if the process running
ps –ef |grep python
# Track the log message
tail -f ./train/log
```

### Download the pre-trained ResNet-50 model

```
cd ./ckpt_files && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-50_209.ckpt
```

### Model evaluation

```
python eval.py --net resnet50 --dataset imagenet2012 --checkpoint_path ./ckpt_files/resnet-50_209.ckpt --dataset_path ../mushroom-dataset/train --device_target GPU
```

### Model prediction

```
python predict.py --checkpoint_path ./ckpt_files/resnet-50_209.ckpt --dataset_path ../mushroom-dataset/test --device_target GPU
```

## Disclaimers

MindSpore ModelZoo only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on MindSpore ModelZoo, or wish to update it in any way. Please contact us through a [Gitee](https://gitee.com/mindspore/mindspore/issues)/[GitHub](https://github.com/mindspore-ai/mindspore/issues) issue. Your understanding and contribution to this community is greatly appreciated.

## License

[Apache License 2.0](../LICENSE)
