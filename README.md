# AI_whiteboard

![](images/ai_whiteboard.gif)

## Idea

The idea of this project is quite interesting. 
I want to transform any wall or surface into an interactive whiteboard just with a camera and your hand!

## Hardware

- Jetson Xavier NX 
- Raspberry Pi Camera + ArduCam (8MP IMX219 Sensor Module)

## Details

2 DNN [1]:
 - Yolo - for hand-detection
 - modified VGG16 - for fingertip detection

## Launch

To run AI whiteboard:

`$ python3 ai_whiteboard.py -j --trt --hd 0.8 --ft 0.5`

#### Control
| To draw | To move | To erase | To clean | To save | 
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|![](images/to_paint.jpg=228x216)|![](images/to_move.jpg)|![](images/to_erase.jpg)|![](images/to_clean.jpg)|![](images/to_save.jpg)|
 

## Train Hand-detector
`$ python3 yolo_train.py`

#### Custom Dataset
- Train: 10,000 images
- Test : 1500 images

## Test Hand-detector
`$ python3 yolo_test.py`

## Convert .h5 model to TensorRT engine

`$ python3 h5_to_trt.py`

## Performance



## References
1. Unified Gesture and Fingertip Detection : https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection