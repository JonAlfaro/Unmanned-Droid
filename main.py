from droidcv import *

new_vid = VideoFeed(0)

while True:
    new_vid.update(True)
    new_logic = HSVcv(new_vid.frame, BLUE_MIN, BLUE_MAX, YELLOW_MIN, YELLOW_MAX)
    new_vid.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
