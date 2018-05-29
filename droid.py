import cv2
import numpy as np

BLUE_MIN = np.array([90, 80, 0], np.uint8)
BLUE_MAX = np.array([110, 255, 255], np.uint8)
YELLOW_MIN = np.array([20, 150, 100], np.uint8)
YELLOW_MAX = np.array([40, 255, 255], np.uint8)
HEIGHT = 200        # px
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


def rshift(seq, n=0):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]


def frame2roi(y, x, frame, median_hsv=None, size=SAMPLE_SIZE):
    roi = frame[y: y + size, x: x + size].copy()
    if median_hsv is not None:
        return cv2.inRange(roi, median_hsv[0], median_hsv[1])
    else:
        return roi
