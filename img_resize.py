"""
    This is specifically designed to resize the FLIR thermal image dataset.
"""

import os

import cv2
import tqdm
from tqdm.auto import tqdm
from collections import Counter

from utils import parser


def main(rgb_dir, ir_dir, dst_rgb, dst_ir):

    files = sorted(os.listdir(rgb_dir))
    print(f"Total: {len(files)}")

    # Lets say dataset has 10 image, but name of last image could be 12
    last_image_number=int(''.join(filter(str.isdigit, files[-1])))

    white = []  # To count white images
    black = []  # To count black images

    # case 3
    margin_left3 = 75
    margin_right3 = 35
    margin_top3 = 22
    margin_bottom3 = 60

    # case 2
    margin_left2 = 35
    margin_right2 = 80
    margin_top2 = 12
    margin_bottom2 = 78

    # case 1
    margin_left1 = 160
    margin_right1 = 45
    margin_top1 = 120
    margin_bottom1 = 170

    k = 0
    others = 0
    sizes=[]

    for i in tqdm(range(1, last_image_number + 1)):

        try:
            im = cv2.imread(rgb_dir + f"/FLIR_{i:0>5d}.jpg")
            h, w, _ = im.shape
            sizes += [(w, h)]

            if im.mean() > 245:
                print(f"White image: FLIR_{i:0>5d}.jpg")
                white += [f"FLIR_{i:0>5d}.jpg"]
                continue
            elif im.mean() < 5:
                print(f"Black image: FLIR_{i:0>5d}.jpg")
                black += [f"FLIR_{i:0>5d}.jpg"]
                continue

            # case 3
            if im.shape[1] == 1280 and im.shape[0] == 1024:

                # RGB
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)

                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                ir = ir[margin_top3:512 - margin_bottom3, margin_left3:640 - margin_right3]
                ir = cv2.resize(ir, (640, 512))
                cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            # case 2
            elif im.shape[1] == 2048 and im.shape[0] == 1536:

                # RGB
                im = im[:, 64:2048 - 64]
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)

                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                ir = ir[margin_top2:512 - margin_bottom2, margin_left2:640 - margin_right2]
                ir = cv2.resize(ir, (640, 512))
                cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            elif im.shape[1] == 1800 and im.shape[0] == 1600:

                # RGB
                im = im[margin_top1:1600 - margin_bottom1, margin_left1:1800 - margin_right1]
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)
                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            else:
                others += 1
        except:
            if os.path.isfile(rgb_dir + f"/FLIR_{i:0>5d}.jpg"):
                print(f"Error: FLIR_{i:0>5d}.jpg exists but cannot be read")
                # raise Exception("???")
            else:
                print(f"No such a file: FLIR_{i:0>5d}.jpg")


    f_rgb = os.listdir(dst_rgb)
    f_ir = os.listdir(dst_ir)

    assert len(f_rgb) == len(f_ir)

    print("*" * 10)

    print("Image sizes in the directory (w x h):")
    print(Counter(sizes))


    print(f"New dataset has {len(f_rgb)} images")

    print(f"Black images: #{len(black)}")
    print(f"White images: #{len(white)}")
    print(f"Non-processed images: #{others}")

    """
    New dataset has 8228 images
    Black images: #0
    White images: #16
    Non-processed images: #119
    """


if __name__ == '__main__':

    args = parser.ResizeParser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.src_rgb) or not os.path.isdir(opt.src_ir):
        raise FileNotFoundError

    if not os.path.isdir(opt.out_dir):
        os.mkdir(opt.out_dir)
        print("Example directory was created")

    dst_rgb = os.path.join(opt.out_dir, "rgb")
    dst_ir = os.path.join(opt.out_dir, "ir")

    if not os.path.isdir(dst_rgb):
        os.mkdir(dst_rgb)
        print("rgb directory was created")

    if not os.path.isdir(dst_ir):
        os.mkdir(dst_ir)
        print("ir directory was created")

    main(rgb_dir=opt.src_rgb, ir_dir=opt.src_ir, dst_rgb=dst_rgb, dst_ir=dst_ir)
