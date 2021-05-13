"""
    To segment the images keras_segmentation module was used.
    As a model, 'pspnet' that trained with city scape dataset was used
"""

from keras_segmentation.pretrained import pspnet_101_cityscapes
from keras import backend as K
from utils import parser
import os
from tqdm.auto import tqdm


def main(opt):

    K.tensorflow_backend._get_available_gpus()

    model = pspnet_101_cityscapes()

    files = sorted(os.listdir(opt.rgb_dir))
    print(f"Total: {len(files)}")

    for f in tqdm(files):
        out = model.predict_segmentation(
            inp=os.path.join(opt.src_rgb, f),
            out_fname=os.path.join(opt.out_dir, f))


if __name__ == '__main__':

    args = parser.ResizeParser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.src_rgb):
        raise FileNotFoundError

    if not os.path.isdir(opt.out_dir):
        os.mkdir(opt.out_dir)
        print("Output directory was created")

    main(opt)
