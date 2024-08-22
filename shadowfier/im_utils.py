# $

import cv2
import numpy as np

from PIL import Image
from io import BytesIO


def pil_to_bytes(im_pil):
    # save the image file
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    return buf.getvalue()


def crop_transparent(src):
    y, x = src[:, :, 3].nonzero()  # get the nonzero alpha coordinates
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)
    return src[miny:maxy, minx:maxx]


def preprocess_img(img, width=None, height=None):
    if width:
        wpercent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((width, hsize), Image.Resampling.LANCZOS)
    if height:
        hpercent = (height / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        return img.resize((wsize, height), Image.Resampling.LANCZOS)


def image_blender(pil_im, new_im):
    src = np.array(pil_im)
    back = cv2.cvtColor(np.array(new_im), cv2.COLOR_BGRA2RGBA)

    h, w, _ = src.shape
    hh, ww, cb = back.shape

    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)

    result = back.copy()
    alpha_s = src[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        im1 = alpha_s * src[:, :, c]
        im2 = alpha_l * back[yoff:yoff + h, xoff:xoff + w, c]
        result[yoff:yoff + h, xoff:xoff + w, c] = (im1 + im2)
    return Image.fromarray(result)


def extract_white_bg(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for white color in HSV
    lower_white = np.array([0, 0, 230])  # Lower bound (H, S, V)
    upper_white = np.array([180, 5, 255])  # Upper bound (H, S, V)

    # Create a binary mask for the white regions
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the white mask to get the mask for non-white regions
    non_white_mask = cv2.bitwise_not(white_mask)

    return non_white_mask


def shift_image(image, shift_pixels):
    # Get the dimensions of the original image
    height, width, channels = image.shape

    # Create an empty canvas with the desired dimensions (add padding to the left)
    shifted_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Copy the original image to the right side of the canvas
    if shift_pixels > 0:
        shifted_image[:, shift_pixels:] = image[:, :width - shift_pixels]
    else:
        shifted_image[:, :width - abs(shift_pixels)] = image[:, abs(shift_pixels):]

    return shifted_image


def has_transparency(img):
    # takes PIL image as input
    if img.info.get("transparency", None) is not None:
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False
