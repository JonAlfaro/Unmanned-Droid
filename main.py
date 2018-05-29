from droidcv import *
import time

new_vid = VideoFeed(0)

while True:
    new_vid.update(True)
    new_logic = HSVcv(new_vid.frame, BLUE_MIN, BLUE_MAX, YELLOW_MIN, YELLOW_MAX)
    print('new frame', new_logic.median_hsv)
    for i in range(10):
        cv2.imshow('hsv', new_logic.hsv)
        new_logic.find_next()
        new_logic.black_fill_last_roi()
        print('new median', new_logic.median_hsv)
    new_vid.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
