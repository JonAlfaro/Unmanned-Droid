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
    pos_logic = PositionLogic(new_vid.full_frame, np.array(new_logic.roi_lists['LeftLane']), np.array(new_logic.roi_lists['RightLane']))
    pos_logic.draw_middle(HEIGHT)
    pos_logic.draw_lanes(HEIGHT)
    cv2.imshow('pos_logic', pos_logic.frame_to_draw)
    new_vid.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
