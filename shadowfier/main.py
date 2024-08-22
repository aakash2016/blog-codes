# $

import cv2
import time
import numpy as np
from PIL import Image
from rembg import remove

from im_utils import image_blender, shift_image, has_transparency


class Shadowfier:
    KERNEL_SIZE = 51  # Gaussian Blur kernel size (odd)
    WHITE_LTH = 250  # Lower threshold value for considering a pixel as white
    CONTRAST = 0.15  # shadow contrast
    BRIGHTNESS = 0.3  # shadow brightness

    def __init__(self, type="NORMAL"):
        self.type = type  # type of shadow - Falt or Normal Drop shadow
        self.STRETCH_W = 0.0 if type == "FLAT" else 0.4  # width param of shadow
        self.STRETCH_H = 0.0 if type == "FLAT" else 0.65  # height param of shadow
        self.FLAT_BUFFER = 50 if type == "FLAT" else 0  # offset for flat shadows

    def reduction_factor(self, as_):
        if as_ > 1.5:
            RF = 0.5
        elif as_ < 0.6:
            RF = 2.5
        elif as_ < 0.25:
            RF = 5.0
        else:
            RF = 1.0
        return RF

    def get_whiteness_ratio(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # non zero pixels
        non_zero_pixels = np.argwhere(gray_image > 0)

        # Calculate the bounding box of non-zero pixels
        y1, x1 = non_zero_pixels.min(axis=0)
        y2, x2 = non_zero_pixels.max(axis=0)

        # crop ROI
        cropped_image = gray_image[y1:y2 + 1, x1:x2 + 1]

        # Count the number of white pixels
        white_pixel_count = np.sum(cropped_image > self.WHITE_LTH)

        # Calculate the total number of pixels in the image
        total_pixel_count = cropped_image.size

        # Calculate the percentage of white pixels in the image
        white_percentage = (white_pixel_count / total_pixel_count) * 100

        return white_percentage > 90


    def get_custom_mask(self, src):
        h, w, _ = src.shape  # get input shape for area

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self.WHITE_LTH, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(thresh, thresh)

        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        m_inner = np.zeros_like(gray)
        m_outer = m_inner.copy()

        # to understand hierarchy -> https://stackoverflow.com/a/71891581
        for i, cnt in enumerate(contours):
            # if (100 * cv2.contourArea(cnt) / (h * w)) > 1.0:
            if hier[0, i, 3] == -1:
                cv2.drawContours(m_inner, [cnt], -1, (255, 255, 255), -1)
            # else:
            # cv2.drawContours(m_outer, [cnt], -1, (255, 255, 255), -1)

        mask = cv2.subtract(m_inner, m_outer)
        return mask

    def create_custom_alpha(self, src):
        mask = self.get_custom_mask(src)  # compute mask
        src = cv2.cvtColor(src, cv2.COLOR_RGB2RGBA)  # First create the image with alpha channel
        src[:, :, 3] = mask  # Then assign the mask to the last channel of the image
        return src

    def shadow_gen(self, img, rem_bg):
        if not has_transparency(img):
            if rem_bg:
                img = remove(img, alpha_matting=True)
                src = np.array(img)
            else:
                src = np.array(img)
                src = self.create_custom_alpha(src)
        else:
            src = np.array(img)

        # apply drop shadow algorithm
        shadow = self.transform(src)

        im_pil = Image.fromarray(shadow)
        return im_pil

    def transform(self, src_og):
        # cropping
        y, x = src_og[:, :, 3].nonzero()  # get the nonzero alpha coordinates
        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        im_og = src_og[miny:maxy, minx:maxx]

        if str(self.STRETCH_W).__contains__('-'):
            src = cv2.flip(im_og, 1)
            src_og = cv2.flip(src_og, 1)
        else:
            src = im_og

        h = src.shape[0]
        w = src.shape[1]

        AS = w / h
        w_x = abs(self.STRETCH_W * self.reduction_factor(AS))

        # affine transform
        PAD_R_TOL = 0.05
        toll = PAD_R_TOL * src_og.shape[1]
        if str(self.STRETCH_W).__contains__('-'):
            toll = toll if toll < minx else 0
        else:
            toll = toll if toll < src_og.shape[1] - maxx else 0

        sh_h_n = h * self.STRETCH_H
        sh_w_n = max(0, min(src_og.shape[1] - maxx - toll, w * w_x))

        srcTri = np.array([[0, 0], [w, 0], [0, h]]).astype(np.float32)
        dstTri = np.array([
            [sh_w_n, sh_h_n],
            [w + sh_w_n, sh_h_n],
            [0, h]
        ]).astype(np.float32)

        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (int(w + sh_w_n), h))

        # _, warp_dst = cv2.threshold(warp_dst, 0, 255, cv2.THRESH_BINARY) --> here the alpha still exists
        img = cv2.blur(warp_dst, (5, 5))

        # padding
        PAD_l = minx
        PAD_r = src_og.shape[1] - minx - img.shape[1]
        PAD_t = miny
        PAD_b = src_og.shape[0] - maxy

        img = cv2.copyMakeBorder(img,
                                 PAD_t, PAD_b, PAD_l, PAD_r,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # remove saturation
        alpha = self.CONTRAST  # Contrast control
        beta = self.BRIGHTNESS  # Brightness control

        # call convertScaleAbs function
        mask = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        mask = cv2.GaussianBlur(mask, (self.KERNEL_SIZE, self.KERNEL_SIZE), 0)

        if self.type == "FLAT":
            mask = shift_image(mask, self.FLAT_BUFFER)

        if str(self.STRETCH_W).__contains__('-'):
            mask = np.flip(mask, 1)
            src_og = np.flip(src_og, 1)

        # alpha blending module
        alpha_s = src_og[:, :, 3] / 255.0
        alpha_mask = np.stack([alpha_s] * 4, axis=-1)
        final_img = (src_og * alpha_mask + mask * (1 - alpha_mask)).astype(np.uint8)

        # convert image back to original format
        base = cv2.cvtColor(final_img, cv2.COLOR_RGBA2BGRA)

        return base

    def shadowfy(self, path, show, rem_bg):
        st = time.time()  # start time

        img = Image.open(path)

        # call shadow generator
        im_sh = self.shadow_gen(img, rem_bg)

        # post processing
        dm = max(im_sh.size)  # output dimension
        new_im = Image.new('RGBA', (dm, dm), "#FFF")
        im_sh_n = image_blender(im_sh, new_im)
        im_cv = np.array(im_sh_n)

        cv2.imwrite(filename=f'results/op_{time.time()}_' + str(path.split("/")[-1]), img=im_cv)

        et = time.time()
        print("time taken: ", et - st)

        if show:
            cv2.imshow("frame", im_cv)
            cv2.waitKey(-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="pass image path", required=True)
    parser.add_argument("--type", help="shadow type (NORMAL/FLAT)", required=False, default='NORMAL')
    parser.add_argument("--show", help="show output (y/n)", required=False, default='n')
    parser.add_argument("--remove_bg", help="use rembg (y/n)", required=False, default='n')

    args = parser.parse_args()

    shadowfier = Shadowfier(type=args.type)
    shadowfier.shadowfy(args.path, args.show.lower() == 'y', args.remove_bg.lower() == 'y')
