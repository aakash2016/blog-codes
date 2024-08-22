#

import math
import numpy as np

class ComputeMetric:
    @staticmethod
    def angle(p1, p2, p3):
        # Angle enclosed at p2
        angle_23 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        angle_21 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        angle_23 = angle_23 + 360 if angle_23 < 0 else angle_23
        angle_21 = angle_21 + 360 if angle_21 < 0 else angle_21
        a = abs(angle_23 - angle_21)
        return min(a, 360 - a)

    @staticmethod
    def distance(p1, p2, kps, as_ratio=None):
        if not as_ratio:
            as_ratio = (1, 1)
        l1x = (kps[p1][0] - kps[p2][0]) * as_ratio[0]
        l1y = (kps[p1][1] - kps[p2][1]) * as_ratio[1]
        l22   = l1x ** 2 + l1y ** 2
        l2 = l22 ** 0.5
        return l2

    @staticmethod
    def normalize_kps(keypoints, shape):
        for kp in keypoints:
            kp[0] /= shape[1]
            kp[1] /= shape[0]
        return keypoints


class SmoothFilter:
    def __init__(self):
        self.kps = None

    def update(self, kps, alpha=0.4):
        kps = np.asarray(kps, dtype=np.float32)
        if self.kps is None:
            self.kps = kps
        else:
            self.kps = alpha * self.kps + (1.0 - alpha) * kps

    def __call__(self, *args, **kwargs):
        return self.kps


class ZeroCrossing:
    def __init__(self, lag, reference):
        self.y = []
        self.lag = lag
        self.reference = reference

    def update(self, new_value):
        self.y.append(new_value)  # append
        self.window = self.y[-self.lag:]  # slice

    def checkCross(self):
        rl = self.window[-1]
        ru = self.window[0]
        if rl < self.reference and ru > self.reference:
            return True
        return False
