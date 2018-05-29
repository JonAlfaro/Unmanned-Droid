from droidcv import *
from datetime import datetime
import time
from pynput.keyboard import Key, Controller

new_vid = VideoFeed(0)
timer = int(str(datetime.now().microsecond)[:1])
pos_logic = PositionLogic(20, 4, 1)  # arguments: +/- ignore range, false pos range (normal) &, (larger/missing lane)
UE4_Simulate_keyboard = False

if UE4_Simulate_keyboard:
    keyboard = Controller()
    time.sleep(5)

# infinite loop that runs 10 scans a second, can be pushed further if needed
while True:
    current_time = int(str(datetime.now().microsecond)[:1])
    if timer != current_time:

        # simulate key releases for UE4 simulation
        if UE4_Simulate_keyboard:
            keyboard.release('a')
            keyboard.release('d')

        # Update video feed frame and being hsvcv logic
        new_vid.update(True)
        new_logic = HSVcv(new_vid.frame, BLUE_MIN, BLUE_MAX, YELLOW_MIN, YELLOW_MAX)

        # scan for 2 lanes, 10 sample areas each
        for i in range(10):
            cv2.imshow('hsv', new_logic.hsv)
            new_logic.find_next()
            new_logic.black_fill_last_roi()

        # prepare position logic to feed into droid
        pos_logic.update(new_vid.full_frame, np.array(new_logic.roi_lists['LeftLane']),
                                  np.array(new_logic.roi_lists['RightLane']))
        pos_logic.draw_middle(HEIGHT)
        pos_logic.draw_lanes(HEIGHT)
        pos_logic.show()
        direction = pos_logic.calc_direction()
        print(direction)

        # use direction output of position logic to simulate movement in UE4 by pressing keys
        if UE4_Simulate_keyboard:
            if direction > 0:
                keyboard.press('a')
            elif direction < 0:
                keyboard.press('d')

        timer = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()