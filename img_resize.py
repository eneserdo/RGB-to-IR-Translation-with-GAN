"""
    This is specifically designed to resize the FLIR thermal image dataset.
"""

import cv2
import glob, os
import tqdm
from collections import Counter
from tqdm.auto import tqdm
from utils import parser


def main(opt):

    rgb_dir = opt.rgb_dir
    ir_dir = opt.ir_dir
    root_dir = r"C:\Users\Enes\Desktop\datasets\flir_processed"

    dst_rgb1 = os.path.join(root_dir, "rgb1")
    dst_rgb2 = os.path.join(root_dir, "rgb2")
    dst_rgb3 = os.path.join(root_dir, "rgb3")

    dst_ir1 = os.path.join(root_dir, "ir1")
    dst_ir2 = os.path.join(root_dir, "ir2")
    dst_ir3 = os.path.join(root_dir, "ir3")

    _, _, files1 = next(os.walk(rgb_dir))

    w = []
    b = []

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

    print(len(files1))

    for i in tqdm(range(1, 8862 + 1)):

        try:
            im = cv2.imread(rgb_dir + f"/FLIR_{i:0>5d}.jpg")

            if im.mean() > 245:
                print(f"White image: FLIR_{i:0>5d}.jpg")
                w += [f"FLIR_{i:0>5d}.jpg"]
                continue
            elif im.mean() < 5:
                print(f"Black image: FLIR_{i:0>5d}.jpg")
                b += [f"FLIR_{i:0>5d}.jpg"]
                continue

            # case 3
            if im.shape[1] == 1280 and im.shape[0] == 1024:


                # RGB
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb3 + f"/FLIR_{k:0>5d}.jpg", im)

                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                ir = ir[margin_top3:512 - margin_bottom3, margin_left3:640 - margin_right3]
                ir = cv2.resize(ir, (640, 512))
                cv2.imwrite(dst_ir3 + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            # case 2
            elif im.shape[1] == 2048 and im.shape[0] == 1536:

                # RGB
                im = im[:, 64:2048 - 64]
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb2 + f"/FLIR_{k:0>5d}.jpg", im)

                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                ir = ir[margin_top2:512 - margin_bottom2, margin_left2:640 - margin_right2]
                ir = cv2.resize(ir, (640, 512))
                cv2.imwrite(dst_ir2 + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            elif im.shape[1] == 1800 and im.shape[0] == 1600:

                # RGB
                im = im[margin_top1:1600 - margin_bottom1, margin_left1:1800 - margin_right1]
                im = cv2.resize(im, (640, 512))
                cv2.imwrite(dst_rgb1 + f"/FLIR_{k:0>5d}.jpg", im)
                # IR
                ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
                cv2.imwrite(dst_ir1 + f"/FLIR_{k:0>5d}.jpg", ir)

                k += 1

            else:
                others += 1
        except:
            if os.path.isfile(rgb_dir + f"/FLIR_{i:0>5d}.jpg"):
                print(f"Error: FLIR_{i:0>5d}.jpg exists but cannot be read")
                raise Exception("???")
            else:
                print(f"No such a file: FLIR_{i:0>5d}.jpg")

    _, _, f1_rgb = next(os.walk(dst_rgb1))
    _, _, f1_ir = next(os.walk(dst_ir1))

    assert f1_rgb == f1_ir

    _, _, f2_rgb = next(os.walk(dst_rgb2))
    _, _, f2_ir = next(os.walk(dst_ir2))

    assert f2_rgb == f2_ir

    _, _, f3_rgb = next(os.walk(dst_rgb3))
    _, _, f3_ir = next(os.walk(dst_ir3))

    assert f3_rgb == f3_ir

    print("*" * 10)

    print(f"Case 3: #{len(f3_rgb)}")
    print(f"Case 2: #{len(f2_rgb)}")
    print(f"Case 1: #{len(f1_rgb)}")

    print(f"Black images: #{len(b)}")
    print(f"White images: #{len(w)}")
    print(f"Non-processed images: #{others}")

    """
    Case 3: #1401
    Case 2: #1973
    Case 1: #4854
    Black images: #0
    White images: #16
    No-processed images: #119
    """


if __name__ == '__main__':

    args = parser.ResizeParser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.rgb_dir) or not os.path.isdir(opt.ir_dir):
        raise FileNotFoundError

    if not os.path.isdir(os.path.join(os.getcwd(),opt.results_dir)):
        os.mkdir(os.path.join(os.getcwd(),opt.results_dir))
        print("Example directory was created")

    main(opt)