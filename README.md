# tinygrad-detr

This project implements DETR (DEtection TRansformer) in tinygrad as described in the paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

DETR is currently a strong contender in the object detection space, [although the newer Yolo outperform significantly](https://huggingface.co/spaces/hf-vision/object_detection_leaderboard)

The DETR model is implemented in `detr.py` and the training loop is implemented in `train.py`.
Semi-effective inference is implemented in `inference.py`.

There are no dependecies, so you can run the software without any external library.

If you run the training as is, you will train on the coco dataset trying to predict '___' on '___' images

This project is under the Apache License 2.0, as per the original DETR code.

**DELETE**:

Update remote submodels:
```
git submodule update --remote
```