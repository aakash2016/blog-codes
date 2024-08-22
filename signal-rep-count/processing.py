"""
    A pre-process script to calculate the zero-crossing reference line.
    input: the trainer video
"""

import os, json

from signal_rep_count.utils.compute import ComputeMetric, SmoothFilter
from signal_rep_count.utils.metrics import Metrics, metricNames
from motion_detection.inference.movenet_infer import *

config_path = "signal_rep_count/CONFIG.json"
SDthreshold = 0.4 # to filter stationary signals

def main(path, normalize, name):
    config_dict = {}

    if path:
        cap = cv2.VideoCapture(path)
        fname = 'op_' + str(path.split("/")[-1])
    else:
        cap = cv2.VideoCapture(0)
        fname = 'op_video.mp4'

    metrics = Metrics()
    track = [[] for _ in range(metrics.lenMetrics)]
    lpftrack = [SmoothFilter() for _ in range(metrics.lenMetrics)]

    ## Writing the video with keypoints
    fps = cap.get(cv2.CAP_PROP_FPS)  # 25
    size = (input_size * 2, input_size * 2)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(fname, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_kp, image = get_inference(frame)
        curr_kp = preprocess_kps(curr_kp, image.shape[0], image.shape[1])

        kps = curr_kp.copy()
        if normalize:
            kps = ComputeMetric.normalize_kps(kps, image.shape)

        metrics.update(kps)
        for i in range(metrics.lenMetrics):
            x = metrics.state[metricNames[i]] / metrics.state["shl_dist"]
            lpftrack[i].update([x], alpha=0.5)
            track[i].append(lpftrack[i]()[0])

        output = draw_pose(image, curr_kp, preprocess=False)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        outimage = np.asarray(output, dtype=np.uint8)
        outimage = cv2.resize(outimage, size)

        video_writer.write(outimage)
        cv2.imshow("frame", outimage)

        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    height, width, c = image.shape

    ## filter stationary wave
    std_array = np.std(track, axis=1)
    nonStation = [i for i, s in enumerate(std_array) if s >= SDthreshold]
    if nonStation == []:
        nonStation = (-std_array).argsort()[:3] # top 3 metrics with highest deviation

    motionMetric = list(np.array(metricNames)[nonStation])

    statistics = {}
    overall_signal = np.sum(np.array(track)[nonStation], axis=0)
    statistics["mean"] = np.mean(overall_signal) * 1.0
    statistics["width"] = width
    statistics["height"] = height

    exerciseData = [motionMetric, statistics]

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
        data[name] = exerciseData
        with open(config_path, 'w') as f:
            f.write(json.dumps(data))
    else:
        config_dict[name] = exerciseData
        with open(config_path, "w") as out_ann_file:
            json.dump(config_dict, out_ann_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video or digit for camera, defaults to 0", required=False, default="0")
    parser.add_argument("--normalize", help="Normalize keypoints (y/n)", required=False, default='n')
    parser.add_argument("--reference", help="name or the key of the config", required=True)
    args = parser.parse_args()

    main(args.video, args.normalize.lower() == 'y', args.reference)
