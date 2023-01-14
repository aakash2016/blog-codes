"""
    Stillness Scorer using Movenet Pose Estimation model
"""
import cv2
import numpy as np
import tensorflow as tf
from motion_detection.inference.oneEuro import OneEuroFilter

import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.interactive(True)

# Initialize the TFLite interpreter
num_kps = 17
input_size = 256
interpreter = tf.lite.Interpreter(model_path="motion_detection/tflite_folder/thunder_model.tflite")
interpreter.allocate_tensors()

# Color library
COLORS = {
    'm': (88, 214, 141),
    'c': (220, 118, 51),
    'y': (174, 182, 191),
}

# Specify color of the edges
EDGE_TO_COLOR = {
    (0, 1): COLORS['m'],
    (0, 2): COLORS['c'],
    (1, 3): COLORS['m'],
    (2, 4): COLORS['c'],
    (0, 5): COLORS['m'],
    (0, 6): COLORS['c'],
    (5, 7): COLORS['m'],
    (7, 9): COLORS['m'],
    (6, 8): COLORS['c'],
    (8, 10): COLORS['c'],
    (5, 6): COLORS['y'],
    (5, 11): COLORS['m'],
    (6, 12): COLORS['c'],
    (11, 12): COLORS['y'],
    (11, 13): COLORS['m'],
    (13, 15): COLORS['m'],
    (12, 14): COLORS['c'],
    (14, 16): COLORS['c'],
}

def euclidean(v1, v2):
    return sum([(p-q)**2 for p, q in zip(v1, v2)][:2]) ** .5

def add(l, val, size):
    if len(l) == size:
        l = l[1:] + [val]
    else:
        l += [val]
    return l

def preprocess_kps(kps, height, width):
    for i in range(len(kps)):
        temp = kps[i][1]
        kps[i][1] = kps[i][0] * height
        kps[i][0] = temp * width
    return kps

def draw_pose(image, keypoints, radius=2):
    height, width, channel = image.shape
    kps = preprocess_kps(keypoints, height, width)
    for c in kps:
        x, y, s = c
        if s > 0.2:
            cv2.circle(image,
                       (int(x), int(y)),
                       radius, (41, 128, 185), -1)
    for edge_pair, color in EDGE_TO_COLOR.items():
        start, end = edge_pair
        x1, y1, s1 = kps[start]
        x2, y2, s2 = kps[end]
        cv2.line(image,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 color, 1,
                 lineType=cv2.LINE_AA)
    return image

def pad(image, width, height):
    image_width = image.shape[1]
    image_height = image.shape[0]

    # get resize ratio
    resize_ratio = min(width / image_width, height / image_height)

    # compute new height and width
    new_width = int(resize_ratio * image_width)
    new_height = int(resize_ratio * image_height)
    new_img = cv2.resize(image, (new_width, new_height))

    # compute padded height and width
    pad_width = (width - new_width) // 2
    pad_height = (height - new_height) // 2

    padded_image = cv2.copyMakeBorder(new_img,
                                      pad_height,
                                      pad_height,
                                      pad_width,
                                      pad_width,
                                      cv2.BORDER_REPLICATE,
                                      value=0)

    return cv2.resize(padded_image, (input_size, input_size))

# Movenet model
def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke() # Invoke inference.

    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def get_inference(image):
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = pad(image, input_size, input_size)
    image = cv2.resize(image, (input_size, input_size))
    input_image = image

    input_image = np.expand_dims(input_image, axis=0)

    # Run model inference.
    kps = movenet(input_image)[0]

    return kps[0], image

def main(path, lpf):
    if path:
        cap = cv2.VideoCapture(path)
        fname = 'op_' + str(path.split("/")[-1])
    else:
        cap = cv2.VideoCapture(0)
        fname = 'op_video.mp4'

    fps = cap.get(cv2.CAP_PROP_FPS)  # 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    ## Writing the video with keypoints
    size = (input_size * 2, input_size * 2)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(fname, fourcc, fps, size)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'))

    fig = plt.figure()
    plt.xlim(0, (frame_count/fps) + 1); plt.ylim(0, 130)
    plt.xlabel('time'); plt.ylabel('motion score')
    plt.title("Stillness scorer using the low pass filter")

    xs = np.zeros(int(frame_count))
    ys = np.zeros(len(xs))

    line, = plt.plot([], [], "b-")

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)

    prev_kp, img = get_inference(frame1)
    scores = []
    j=0

    global x_track, y_track
    while True:
        ctime = float(cframe - 2) / float(fps)

        old_curr_kp, image = get_inference(frame2)
        curr_kp = [x[:] for x in old_curr_kp]  ## Deepcopy

        if j == 0:
            x_track = [OneEuroFilter(j, curr_kp[k][0], 0.6, 0.015) for k in range(num_kps)]  # track for all keypoints
            y_track = [OneEuroFilter(j, curr_kp[k][1], 0.6, 0.015) for k in range(num_kps)]

        if lpf and j > 1:
            for i in range(num_kps):
                ## x coordinate
                curr_kp[i][0] = x_track[i](j, curr_kp[i][0])

                ## y coordinate
                curr_kp[i][1] = y_track[i](j, curr_kp[i][1])

        output = draw_pose(image, curr_kp)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        outimage = np.asarray(output, dtype=np.uint8)
        outimage = cv2.resize(outimage, size)

        video_writer.write(outimage)
        cv2.imshow("frame", outimage)

        score = [prev_kp[i][2] * euclidean(prev_kp[i], curr_kp[i]) for i in range(num_kps)]
        sc = np.round(np.sum(score), 4)
        scores = add(scores, sc, 5)  ## score smoothing

        ## Plotting
        xs[j] = ctime
        ys[j] = np.mean(scores)

        line.set_xdata(list(xs[:j]))
        line.set_ydata(list(ys[:j]))
        plt.ylim([0, 60])
        plt.show()

        prev_kp = curr_kp
        ret, frame2 = cap.read()
        cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)
        j += 1

        if not ret:
            break

        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    def animate(i):
        line.set_data(xs[:i], ys[:i])
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(xs), blit=True)
    anim.save('graph_' + fname, writer=writer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="pass video path", required=False)
    parser.add_argument("--lpf", help="use lpf (y/n)", required=False, default='n')

    args = parser.parse_args()
    main(args.path, args.lpf.lower() == 'y')
