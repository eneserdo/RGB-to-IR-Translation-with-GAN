"""
    This is a very inefficient code. To use it, you have to give the part number.
    Code works on 500 image at once i.e. part 0 refers to first 500 image
    To segment the images keras_segmentation module was used.
    As a model, 'pspnet' that trained with city scape dataset was used
"""

from keras_segmentation.pretrained import pspnet_101_cityscapes

from tqdm.auto import tqdm
import os

from utils import parser


def main(opt):

    model = pspnet_101_cityscapes()

    files = sorted(os.listdir(opt.src_rgb))
    print(f"Total: {len(files)}")

    if 500*(opt.part+1)<len(files):
        partial = files[500*opt.part:500*(opt.part+1)]
        print(f"Processing from {500*opt.part} to {500*(opt.part+1)} ...")
    else:
        partial = files[500*opt.part:]
        print(f"Processing from {500*opt.part} to {len(files)} ...")


    for f in tqdm(partial):
        out = model.predict_segmentation(
            inp=os.path.join(opt.src_rgb, f),
            out_fname=os.path.join(opt.out_dir, f))


if __name__ == '__main__':

    args = parser.SegmentParser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.src_rgb):
        raise FileNotFoundError

    if not os.path.isdir(opt.out_dir):
        os.mkdir(opt.out_dir)
        print("Output directory was created")

    main(opt)
