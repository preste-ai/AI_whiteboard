import cv2
import numpy as np
import time
import copy
import argparse

from unified_detector import Fingertips
from hand_detector.detector import YOLO
from whiteboard_utils import config, get_centered_whiteboart, drow_on_whiteboard
from gst_cam import gstreamer_pipeline

COLOR = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]


def run_whiteboard(args):

    # init model
    print('-- init models --')
    k_f = 0
    # try:

    hand = YOLO(weights='weights/best_yolo.h5', trt_engine = 'weights/converted/model_yolo.fp16.engine', threshold=0.75, trt = args.trt & args.jetson)
    fingertips = Fingertips(weights='weights/classes8.h5', trt_engine = 'weights/converted/model_classes8.fp16.engine', trt = args.trt & args.jetson)
    
    if args.jetson:
        cam = cv2.VideoCapture(gstreamer_pipeline(capture_width=config['cam_w'],
                                                    capture_height=config['cam_h'],
                                                    display_width=config['cam_w'],
                                                    display_height=config['cam_h'],
                                                    framerate=config['framerate']), 
                                cv2.CAP_GSTREAMER)  
    else:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, config['cam_w']) # 1280
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config['cam_h']) # 720

    origin_w  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    origin_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print('origin width : ', origin_w)
    print('origin height: ', origin_h)
    origin_center_x = int(origin_w/2)
    origin_center_y = int(origin_h/2)
    # tl - top left | br - bottom right
    

    cropped_x_s = origin_center_x - int(origin_h/2)
    cropped_x_e = origin_center_x + int(origin_h/2)
    whiteboard_tl, whiteboard_br = get_centered_whiteboart(int((cropped_x_e - cropped_x_s)/2), origin_center_y)
    # Create a whiteboard
    whiteboard = np.zeros((config['z_koef']*config['whiteboard_h'],config['z_koef']*config['whiteboard_w'],3), np.uint8) + 255
    info_whiteboard = copy.deepcopy(whiteboard)
    while True:
        ret, image = cam.read()
        #print('image shape : ', image.shape)
        image = image[:,cropped_x_s:cropped_x_e,:]

        if ret is False:
            break

        start = time.time()

        # hand detection
        tl, br = hand.detect(image=image)
        if tl and br is not None and br[0] - tl[0] >= 5 and  br[1] - tl[1] >= 5:
            cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
            height, width, _ = cropped_image.shape

            # gesture classification and fingertips regression
            prob, pos = fingertips.classify(image=cropped_image)
            pos = np.mean(pos, 0)

            # post-processing
            prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
            for i in range(0, len(pos), 2):
                pos[i] = pos[i] * width + tl[0]
                pos[i + 1] = pos[i + 1] * height + tl[1]

            ref_pos = []
            for i in range(0, len(pos), 2):
                tmp_x = max(-5, pos[i] - whiteboard_tl[0])/config['whiteboard_w']
                tmp_y = max(-5, pos[i+1] - whiteboard_tl[1])/config['whiteboard_h']
                ref_pos.append(tmp_x)
                ref_pos.append(tmp_y)
            # drawing
            index = 0
            
            image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
            whiteboard, info_whiteboard, k_f = drow_on_whiteboard(whiteboard, prob, ref_pos)

            for c, p in enumerate(prob):
                if p > 0.5:
                    image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                       color=COLOR[c], thickness=-2)
                index += 2

        k = cv2.waitKey(1)
        if k==27:       # Esc key to stop
            break
        # elif k_f == 1:  # Save whiteboard
        #     # cv2.imwrite('saved/whiteboard.jpg', whiteboard) 
        #     break

        end = time.time()

        str_fps = '{:.1f} fps'.format(1/(end-start ))
        # print(str_fps)
        cv2.putText(image, str_fps,(25,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        image = cv2.rectangle(image, (whiteboard_tl[0], whiteboard_tl[1]), (whiteboard_br[0], whiteboard_br[1]), (255, 255, 255), 2)
        
        # display image
        cv2.imshow('Fingertips Detection', image)
        # display whiteboard
        cv2.imshow('Whiteboard', info_whiteboard)


    cam.release()
    cv2.destroyAllWindows()

    # except Exception as e:
    #     print("Error: {}".format(e))
    #     exit(1)


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Whiteboard arguments')
    
    parser.add_argument('-j','--jetson', dest='jetson', action='store_true', help='Run whiteboard on jetson')
    parser.set_defaults(jetson=False)
    parser.add_argument('--trt', dest='trt', action='store_true', help='Use TensoRT model')
    parser.set_defaults(trt=False)
    return parser.parse_args()

if __name__ == "__main__":
    args=parse_args()
    run_whiteboard(args)
