"""
    This is specifically designed to resize the FLIR thermal image dataset.
"""

import os

import cv2
import tqdm
from tqdm.auto import tqdm
from collections import Counter

from utils import parser


def main(rgb_dir, segment_dir, dst_rgb, dst_segment):

    files = sorted(os.listdir(rgb_dir))
    print(f"Total: {len(files)}")

    # Lets say dataset has 10 image, but name of last image could be 12
    last_image_number=int(''.join(filter(str.isdigit, files[-1])))

    white = []  # To count white images
    black = []  # To count black images

    k = 0
    others = 0
    sizes=[]

    for i in tqdm(range(1, last_image_number + 1)):

        try:
            im = cv2.imread(rgb_dir + f"/IMG_{i:0>4d}.png")
            h, w, _ = im.shape
            sizes += [(w, h)]

            if im.mean() > 245:
                print(f"White image: IMG_{i:0>4d}.png")
                white += [f"IMG_{i:0>4d}.jpg"]
                continue
            elif im.mean() < 5:
                print(f"Black image: IMG_{i:0>4d}.png")
                black += [f"IMG_{i:0>4d}.png"]
                continue

            # RGB

            im = cv2.resize(im, (640, 480))
            cv2.imwrite(dst_rgb + f"/IMG_{k:0>4d}.png", im)
            # IR
            ir = cv2.imread(segment_dir + f"/IMG_{i:0>4d}.png")
            cv2.imwrite(dst_segment + f"/IMG_{k:0>4d}.png", ir)

            k += 1

        except:
            if os.path.isfile(rgb_dir + f"/IMG_{i:0>4d}.png"):
                print(f"Error: IMG_{i:0>4d}.jpg exists but cannot be read")
                # raise Exception("???")
            else:
                print(f"No such a file: IMG_{i:0>4d}.png")

    f_rgb = os.listdir(dst_rgb)
    f_ir = os.listdir(dst_segment)



    print("*" * 10)

    print("Image sizes in the directory (w x h):")
    print(Counter(sizes))

    print(f"New dataset has {len(f_rgb)} images")

    print(f"Black images: #{len(black)}")
    print(f"White images: #{len(white)}")
    print(f"Non-processed images: #{others}")

    assert len(f_rgb) == len(f_ir)

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

    if not os.path.isdir(opt.src_rgb) or not os.path.isdir(opt.src_segment):
        raise FileNotFoundError

    if not os.path.isdir(opt.out_dir):
        os.mkdir(opt.out_dir)
        print("Example directory was created")

    dst_rgb = os.path.join(opt.out_dir, "rgb")
    dst_segment = os.path.join(opt.out_dir, "segment")

    if not os.path.isdir(dst_rgb):
        os.mkdir(dst_rgb)
        print("rgb directory was created")

    if not os.path.isdir(dst_segment):
        os.mkdir(dst_segment)
        print("ir directory was created")

    main(rgb_dir=opt.src_rgb, segment_dir=opt.src_segment, dst_rgb=dst_rgb, dst_segment=dst_segment)
