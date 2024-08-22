"""
    RepCount based on signal processing based features
"""

import json

from signal_rep_count.utils.metrics import Metrics
from signal_rep_count.utils.compute import ComputeMetric, ZeroCrossing, SmoothFilter
from motion_detection.inference.movenet_infer import *

config_path = "signal_rep_count/CONFIG.json"
windowSize = 10

def main(video, reference, normalize):
    with open(config_path, "r") as f:
        data = json.load(f)

    if video:
        cap = cv2.VideoCapture(video)
        fname = 'op_' + str(video.split("/")[-1])
    else:
        cap = cv2.VideoCapture(0)
        fname = 'op_video.mp4'

    metricsThresh = data.get(reference, None)
    nonstat = metricsThresh[0]
    ref = metricsThresh[1]["mean"]
    ref_w = metricsThresh[1]["width"]
    ref_h = metricsThresh[1]["height"]

    if normalize:
        vcap = cv2.VideoCapture(video)
        f_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        as_ratio = (f_width/ref_w, f_height/ref_h)
    else:
        as_ratio = (1,1)

    # initialise all filter objects
    metrics = Metrics(as_ratio)
    track = [[] for _ in range(len(nonstat))]
    lpftrack = [SmoothFilter() for _ in range(len(nonstat))]
    zc = ZeroCrossing(windowSize, ref)

    overall_signal = []
    checkzc = []
    prev = reps = 0

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

        sum_ = 0
        for i in range(len(nonstat)):
            x = (metrics.state[nonstat[i]] * as_ratio[1]) / metrics.state["shl_dist"]
            lpftrack[i].update([x], alpha=0.5)
            track[i].append(lpftrack[i]()[0])
            sum_ += lpftrack[i]()[0]

        overall_signal.append(sum_)
        zc.update(sum_)

        current = zc.checkCross()
        checkzc.append(current)

        if prev == 0 and current == 1:
            reps += 1

        prev = current

        output = draw_pose(image, curr_kp, preprocess=False)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.putText(output, f'reps: {reps}', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (22, 160, 133), 1)

        outimage = np.asarray(output, dtype=np.uint8)
        outimage = cv2.resize(outimage, size)

        video_writer.write(outimage)
        cv2.imshow("frame", outimage)

        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video or digit for camera, defaults to 0", required=False, default="0")
    parser.add_argument("--normalize", help="Normalize keypoints (y/n)", required=False, default='n')
    parser.add_argument("--reference", help="reference config key", required=False)
    args = parser.parse_args()

    main(args.video, args.reference, args.normalize.lower() == 'y')
