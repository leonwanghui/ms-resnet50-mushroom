# MindSpore ResNet-50 Tutorial with GPU backend
This is a tutorial for training MindSpore ResNet-50 model to classifying mushrooms.

> **NOTICE:** The codebase of this tutorial is developed based on `v1.0` MindSpore [ModelZoo](https://github.com/mindspore-ai/mindspore/tree/r1.0/model_zoo/official/cv/resnet).

## Guidelines

### Install some system packages

* System package

    ```
    sudo apt install -y unzip
    ```

* Python package

    ```
    pip install opencv-python
    ```

* MindSpore (**v1.0**)

    For MindSpore installation, please refer to [MindSpore install page](https://www.mindspore.cn/install).

### Download source code

```
git clone https://github.com/leonwanghui/ms-resnet50-mushroom.git
cd ms-resnet50-mushroom/
```

# Download mushroom dataset

```
cd mushroom-dataset/ && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip
unzip mushrooms.zip && rm mushrooms.zip
cd ../resnet_gpu/
```

Or you can directly open [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip) to download the dataset from the browser.

### Model training

```
python train.py --dataset_path ../mushroom-dataset/train
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
cd ./ckpt_files && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt
```

Or you can directly open [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt) to download the pre-trained model from the browser.

### Model evaluation

```
python eval.py --checkpoint_path ./ckpt_files/resnet-90_209.ckpt --dataset_path ../mushroom-dataset/train
```
```
result: {'top_1_accuracy': 0.7034988038277512, 'top_5_accuracy': 0.9732356459330144} ckpt= ./ckpt_files/resnet-90_209.ckpt
```

### Model prediction

```
python predict.py --checkpoint_path ./ckpt_files/resnet-90_209.ckpt --image_path ./tum.jpg
```
```
预测的蘑菇标签为:
	Lactarius松乳菇,红菇目,红菇科,乳菇属,广泛分布于亚热带松林地,无毒
```

## License

[Apache License 2.0](../LICENSE)
