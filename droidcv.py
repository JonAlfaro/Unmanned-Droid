import cv2
import numpy as np

BLUE_MIN = np.array([90, 80, 0], np.uint8)
BLUE_MAX = np.array([110, 255, 255], np.uint8)
YELLOW_MIN = np.array([20, 150, 100], np.uint8)
YELLOW_MAX = np.array([40, 255, 255], np.uint8)
HEIGHT = 200 # px
SAMPLE_SIZE = 25


class VideoFeed(object):
    def __init__(self, cam_select=0):
        self.cap = cv2.VideoCapture(cam_select)
        self.ret, self.frame = self.cap.read()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def update(self, crop_check=False):
        self.ret, self.frame = self.cap.read()  ##READ FRAME
        self.auto_crop()
        if crop_check:
            self.crop()

    def auto_crop(self, threshold=0):
        if len(self.frame.shape) == 3:
            flat_image = np.max(self.frame, 2)
        else:
            flat_image = self.frame
        assert len(flat_image.shape) == 2

        rows = np.where(np.max(flat_image, 0) > threshold)[0]
        if rows.size:
            cols = np.where(np.max(flat_image, 1) > threshold)[0]
            crop = self.frame[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
        else:
            crop = self.frame[:1, :1]

        # Now crop the image, and save to frame
        self.frame = crop.copy()

    def crop(self):
        x = self.frame.shape[1]
        y = self.frame.shape[0]
        self.frame = self.frame[y - HEIGHT:y, 0:x].copy()

    def show(self, name='feed', next_frame=True):
        if self.ret:
            cv2.imshow(name, self.frame)
        if next_frame:
            self.update()


class HSVcv(object):

    def __init__(self, frame, blue_min, blue_max, yellow_min, yellow_max):
        self.bgr = frame.copy()
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        self.y, self.x = self.hsv.shape[0], self.hsv.shape[1]
        self.median_hsv = {'LeftLane': [blue_min.copy(), blue_max.copy()], 'RightLane': [yellow_min.copy(), yellow_max.copy()]}
        first_rois = self.find_first()
        self.roi_lists = {'LeftLane': [first_rois[0]], 'RightLane': [first_rois[1]]}

    def find_first(self):
        pos_yx = [[0, 0, 0, -1], [0, 0, 0, -1]]  # default values for defining roi, [[max count, y, x, segment], ...]

        # Check left and right borders of frame against respective thresholds ( left = blue, right = yellow)
        y = 0
        while y < self.y - SAMPLE_SIZE:
            left_lane_selected_left_roi = frame2roi(y, 0, self.hsv, self.median_hsv['LeftLane'])
            right_lane_selected_right_roi = frame2roi(y, self.x - SAMPLE_SIZE, self.hsv, self.median_hsv['RightLane'])
            left_nonzero_count = cv2.countNonZero(left_lane_selected_left_roi)
            right_nonzero_count = cv2.countNonZero(right_lane_selected_right_roi)
            # Find max nonzero count regions of interest and set y and x position
            if left_nonzero_count > pos_yx[0][0]:
                pos_yx[0][0] = left_nonzero_count
                pos_yx[0][1] = y
                pos_yx[0][2] = 0

            if right_nonzero_count > pos_yx[1][0]:
                pos_yx[1][0] = right_nonzero_count
                pos_yx[1][1] = y
                pos_yx[1][2] = self.x - SAMPLE_SIZE

            y += SAMPLE_SIZE

        # # Check bottom border of frame against both thresholds (left = blue, rihgt = yellow)
        x = 0
        while x < self.x - SAMPLE_SIZE:
            bottom_lane_selected_left_roi = frame2roi(self.y - SAMPLE_SIZE, x, self.hsv, self.median_hsv['LeftLane'])
            bottom_lane_selected_right_roi = frame2roi(self.y - SAMPLE_SIZE, x, self.hsv, self.median_hsv['RightLane'])
            bottoml_leftr_nonzero_count = cv2.countNonZero(bottom_lane_selected_left_roi)
            bottoml_rightr_nonzero_count = cv2.countNonZero(bottom_lane_selected_right_roi)

            if bottoml_leftr_nonzero_count > pos_yx[0][0]:
                pos_yx[0][0] = bottoml_leftr_nonzero_count
                pos_yx[0][1] = y - SAMPLE_SIZE
                pos_yx[0][2] = x

            if bottoml_rightr_nonzero_count > pos_yx[1][0]:
                pos_yx[1][0] = bottoml_rightr_nonzero_count
                pos_yx[1][1] = self.y - SAMPLE_SIZE
                pos_yx[1][2] = x

            x += SAMPLE_SIZE

        return pos_yx.copy()


def rshift(seq, n=0):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]


def frame2roi(y, x, frame, median_hsv=None, size=SAMPLE_SIZE):
    roi = frame[y: y + size, x: x + size].copy()
    if median_hsv is not None:
        return cv2.inRange(roi, median_hsv[0], median_hsv[1])
    else:
        return roi
