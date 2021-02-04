# AI_whiteboard

## Idea

The idea of this project is quite interesting. 
I want to transform any wall or surface into interactive whiteboard just with a camera and your hand!

## Hardware

- Jetson Xavier NX 
- Raspberry Pi Camera + ArduCam (8MP IMX219 Sensor Module)

## Details

2 DNN [1]:
 - Yolo - for hand-detection
 - modified VGG16 - for fingertip detection

## Launch

To run AI whiteboard:

`$ python3 camera_ai_whiteboard.py -j --trt`

#### Control
| To paint  | To move cursor  |  To erase | To clean | To save | 
| :---------------:|:---------------:|:---------------:|:--------------:|:---------------:|
| ![](images/to_paint.jpg)  |![](images/to_move.jpg) | ![](images/to_erase.jpg) |![](images/to_clean.jpg)|![](images/to_save.jpg)|


## Train Hand-detector
`$ python3 yolo_train.py`

#### Custom Dataset
- Train: 8,000 images
- Test : 1500 images

## Test Hand-detector
`$ python3 yolo_test.py`

## Convert .h5 model to TensorRT engine

`$ python3 h5_to_trt.py`

## Performance



## References
1. Unified Gesture and Fingertip Detection : https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection