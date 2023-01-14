# %

from .compute import ComputeMetric
from .constants import POSE_DICT, metricNames


class Metrics:
    def __init__(self, params=None):
        self.state = {el:0 for el in metricNames}
        self.lenMetrics = len(self.state)
        self.as_ratio = params

    def update(self, kps):
        self.state["shl_dist"] = ComputeMetric.distance(
            POSE_DICT["left_shoulder"],
            POSE_DICT["right_shoulder"],
            kps,
            self.as_ratio
        )

        self.state["lshl_lpalm_dist"] = kps[POSE_DICT["left_shoulder"]][1] - kps[POSE_DICT["left_wrist"]][1]
        self.state["rshl_rPalm_dist"] = kps[POSE_DICT["right_shoulder"]][1] - kps[POSE_DICT["right_wrist"]][1]
        self.state["lshl_rpalm_dist"] = kps[POSE_DICT["left_shoulder"]][1] - kps[POSE_DICT["right_wrist"]][1]
        self.state["rShl_lpalm_dist"] = kps[POSE_DICT["right_shoulder"]][1] - kps[POSE_DICT["left_wrist"]][1]

        self.state["lshl_lHip_dist"] = kps[POSE_DICT["left_hip"]][1] - kps[POSE_DICT["left_shoulder"]][1]
        self.state["rshl_rhip_dist"] = kps[POSE_DICT["right_hip"]][1] - kps[POSE_DICT["right_shoulder"]][1]

        self.state["lknee_lhip_dist"] = kps[POSE_DICT["left_knee"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rknee_rhip_dist"] = kps[POSE_DICT["right_knee"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lknee_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_knee"]][1]
        self.state["rknee_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_knee"]][1]

        self.state["lhip_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rhip_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lpalm_lhip_dist"] = kps[POSE_DICT["left_wrist"]][1] - kps[POSE_DICT["left_hip"]][1]
        self.state["rpalm_rhip_dist"] = kps[POSE_DICT["right_wrist"]][1] - kps[POSE_DICT["right_hip"]][1]

        self.state["lpalm_lfeet_dist"] = kps[POSE_DICT["left_ankle"]][1] - kps[POSE_DICT["left_wrist"]][1]
        self.state["rpalm_rfeet_dist"] = kps[POSE_DICT["right_ankle"]][1] - kps[POSE_DICT["right_wrist"]][1]

        self.state["lrpalm_dist"] = kps[POSE_DICT["left_wrist"]][0] - kps[POSE_DICT["right_wrist"]][0]

        self.state["lshl_angle"] = ComputeMetric.angle(
            kps[POSE_DICT["left_elbow"]],
            kps[POSE_DICT["left_shoulder"]],
            kps[POSE_DICT["left_hip"]]
        )

        self.state["rshl_angle"] = ComputeMetric.angle(
            kps[POSE_DICT["right_elbow"]],
            kps[POSE_DICT["right_shoulder"]],
            kps[POSE_DICT["right_hip"]]
        )

    def getMetrics(self):
        return self.state
