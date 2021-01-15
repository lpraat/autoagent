# You Only Look Once (YOLO)
Pytorch-based implementation with structural and algorithmic changes presented in the v4/v5 versions, including Cross Stage Partial modules (CSP), Spatial Pyramid Pooling (SPP), Path Aggregation Network (PAN), Mosaic Augmentation, CIoU-loss, Random training shapes.

<img src="./sample_data/out.jpg"/>

<hr/>

## Results on VOC dataset
| Model                                | AP          |  AP@0.5  | Input size (train and val)   | 
| ------------------------------------ | ----------- | -------- | ---------------------------- |
| Yolov5-large (training from scratch) | 54.92       | 78.62    | 416x416                      |
| Yolov5-large (fine-tuning from COCO) | 67.32       | 87.32    | 416x416                      |

## Training and Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lpraat/autoagent/blob/master/autoagent/models/vision/yolo/run_on_colab.ipynb)


### References
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- https://github.com/AlexeyAB/darknet
- https://github.com/ultralytics/yolov5
