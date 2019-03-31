import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

print('Welcome to ARCS image recognition program. Designed by Siddharth Iyer for the ARCS project\n')

print("Please select the type of camera : \n1) A USB camera is attached.\n2) An IP camera is to be used")

cam = input()

if cam == 1:
    print('Connecting to camera')
    capture = cv2.VideoCapture(0)

if cam == 2:
    url = raw_input('Enter IP address: ')
    # Selections for the program
    print('Connecting to camera')
    capture = cv2.VideoCapture(url)

if capture.isOpened() == False:
    print('Connection to camera failed')
    exit()

else:
    print('Connection successful.')

print("Select processing type: \n 1) Fast processing\n 2) Accurate processing\n")

opt = input()

if opt == 1:
    options = {
        'model': 'cfg/tiny-yolo-voc.cfg',
        'load': 'bin/tiny-yolo-voc.weights',
        'threshold': 0.4,
        'gpu': 0.9
    }

if opt == 2:
    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.4,
        'gpu': 0.9
    }

print('Initializing program ')
time.sleep(1)

tfnet = TFNet(options)

print('Program initialized. Press ctrl + c to exit')

colors = [tuple(255 * np.random.rand(3))
          for _ in range(10)]

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:

        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            label = result['label']

            if label == 'person':
                tl = (result['topleft']['x'], result['topleft']['y'])

                br = (result['bottomright']['x'], result['bottomright']['y'])

                frame = cv2.rectangle(frame, tl, br, color, 5)

        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
