import cv2
import numpy as np
import copy

config = {
	'whiteboard_w': 220, # 640 # 1100
	'whiteboard_h': 220, # 480 # 620
	'cam_w'       : 320,
	'cam_h'       : 240,
	'framerate'   : 60,
	'z_koef'      : 2,   # zoom koef
}


def get_centered_whiteboart(origin_image_x, origin_image_y):

	tl = (int( origin_image_x - config['whiteboard_w']/ 2), int( origin_image_y -  config['whiteboard_h']/ 2) )
	rb = (int( origin_image_x + config['whiteboard_w']/ 2), int( origin_image_y +  config['whiteboard_h']/ 2) )
	
	return tl, rb


def draw_circle (info_whiteboard, center, color = (0, 0, 0), radius=8, thickness= -1):
	cv2.circle(info_whiteboard, 
				center, 
				radius=radius,
				color=color, 
				thickness= thickness)

	

def drow_on_whiteboard(whiteboard, prob, pos, whiteboard_tl, whiteboard_br):

	n_fingers = int(np.sum(prob))
	k = 0
	# one finger detected : INDEX  | action: paint
	if n_fingers == 1 and prob[1] == 1.0:
		center = (int(pos[2] - whiteboard_tl[0]), int(pos[3]- whiteboard_tl[1]) )
		draw_circle (whiteboard, center)

		info_whiteboard = copy.deepcopy(whiteboard)
		draw_circle (info_whiteboard, center, color = (0, 20, 200), thickness = 2)
	
	# two fingers detected: THUMB + INDEX | action: show pointer
	elif n_fingers == 2 and prob[1] == 1.0 and prob[0] == 1.0:
		center = (int(pos[2] - whiteboard_tl[0]), int(pos[3]- whiteboard_tl[1]) )
		
		info_whiteboard = copy.deepcopy(whiteboard)
		draw_circle (info_whiteboard, center, color = (255,0,0), thickness = 2)

	# five fingers detected | action:  erase 
	elif n_fingers == 5 :
		center = (int(pos[2] - whiteboard_tl[0]), int(pos[3]- whiteboard_tl[1]) )

		whiteboard = cv2.circle(whiteboard, center, radius=10,
								color=(255,255,255), thickness=-1)

		info_whiteboard = copy.deepcopy(whiteboard)
		draw_circle (info_whiteboard, center, color = (0, 255, 0), radius=12, thickness= 2)

	# two fingers detected: THUMB + PINKY | action: clean whiteboard
	elif n_fingers == 2 and prob[0] == 1.0 and prob[4] == 1.0:
		whiteboard = np.zeros((config['z_koef']*config['whiteboard_h'],config['z_koef']*config['whiteboard_w'],3), np.uint8) + 255
		info_whiteboard = copy.deepcopy(whiteboard)
	
	# three fingers detected: THUMB + MIDDLE + RING | action: save whiteboard
	elif n_fingers == 3 and prob[1] == 1.0 and prob[2] == 1.0 and prob[3] == 1.0:
		cv2.imwrite('saved/whiteboard_f.jpg', whiteboard)
		print('=== SAVED ===')
		info_whiteboard = copy.deepcopy(whiteboard)

	# three fingers detected: THUMB + INDEX + PINKY | action: exit
	elif n_fingers == 3 and prob[0] == 1.0 and prob[1] == 1.0 and prob[4] == 1.0:
		info_whiteboard = copy.deepcopy(whiteboard)
		k = 1
		print('=== EXIT ===')
	else:
		info_whiteboard = copy.deepcopy(whiteboard)
	
	return whiteboard, info_whiteboard, k
