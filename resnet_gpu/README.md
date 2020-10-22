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
unzip mushrooms.zip && rm mushrooms.zip
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
```
epoch: 90 step: 201, loss is 1.5889285
epoch: 90 step: 202, loss is 1.377257
epoch: 90 step: 203, loss is 1.6227098
epoch: 90 step: 204, loss is 1.5957711
epoch: 90 step: 205, loss is 1.4774182
epoch: 90 step: 206, loss is 1.3818822
epoch: 90 step: 207, loss is 1.2700025
epoch: 90 step: 208, loss is 1.5183961
epoch: 90 step: 209, loss is 1.5881176
Epoch time: 11870.333, per step time: 56.796
```

### Download the pre-trained ResNet-50 model

```
cd ./ckpt_files && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-50_209.ckpt
```

### Model evaluation

```
python eval.py --net resnet50 --dataset imagenet2012 --checkpoint_path ./ckpt_files/resnet-50_209.ckpt --dataset_path ../mushroom-dataset/train --device_target GPU
```
```
result: {'top_5_accuracy': 0.9594796650717703, 'top_1_accuracy': 0.6402511961722488} ckpt= ./ckpt_files/resnet-50_209.ckpt
```

### Model prediction

```
python predict.py --checkpoint_path ./ckpt_files/resnet-50_209.ckpt --dataset_path ../mushroom-dataset/test --device_target GPU
```
```
---The 1 prediction---
Expected Amanita毒蝇伞,伞菌目,鹅膏菌科,鹅膏菌属,主要分布于我国黑龙江、吉林、四川、西藏、云南等地,有毒,
	 got Amanita毒蝇伞,伞菌目,鹅膏菌科,鹅膏菌属,主要分布于我国黑龙江、吉林、四川、西藏、云南等地,有毒

---The 2 prediction---
Expected Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒,
	 got Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒

---The 3 prediction---
Expected Boletus丽柄牛肝菌,伞菌目,牛肝菌科,牛肝菌属,分布于云南、陕西、甘肃、西藏等地,有毒,
	 got Boletus丽柄牛肝菌,伞菌目,牛肝菌科,牛肝菌属,分布于云南、陕西、甘肃、西藏等地,有毒

---The 4 prediction---
Expected Cortinarius掷丝膜菌,伞菌目,丝膜菌科,丝膜菌属,分布于湖南等地(夏秋季在山毛等阔叶林地上生长),
	 got Cortinarius掷丝膜菌,伞菌目,丝膜菌科,丝膜菌属,分布于湖南等地(夏秋季在山毛等阔叶林地上生长)
```

## Disclaimers

MindSpore ModelZoo only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on MindSpore ModelZoo, or wish to update it in any way. Please contact us through a [Gitee](https://gitee.com/mindspore/mindspore/issues)/[GitHub](https://github.com/mindspore-ai/mindspore/issues) issue. Your understanding and contribution to this community is greatly appreciated.

## License

[Apache License 2.0](../LICENSE)
