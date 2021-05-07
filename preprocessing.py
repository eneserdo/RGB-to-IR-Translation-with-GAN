# import cv2
# import glob, os
# import tqdm
# from collections import Counter
#
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
# ]
#
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
#
# root_dir = r"C:\Users\Enes\Desktop\datasets"
#
# src_ms_rgb = r"C:\Users\Enes\Desktop\datasets\ir_det_dataset\Images\rgb"
# src_ms_fir = r"C:\Users\Enes\Desktop\datasets\ir_det_dataset\Images\fir"
#
#
# # dst_flir=os.path.join(root_dir, 'flir')
# # dst_ms=os.path.join(root_dir, 'ms')
# #
# # os.mkdir(dst_ms)
# # os.mkdir(dst_flir)
#
#
# def calc_means(dir):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#
#     b_mean = 0
#     # g_mean = 0
#     # r_mean = 0
#
#     b_std = 0
#     # g_std = 0
#     # r_std = 0
#
#     i = 0
#
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#
#             if is_image_file(fname):
#                 i += 1
#                 path = os.path.join(root, fname)
#                 img = cv2.imread(path)
#                 b, g, r = cv2.split(img)
#
#                 b = b / 255.0
#                 # g=g / 255.0
#                 # r=r / 255.0
#
#                 b_mean += b.mean()
#                 # g_mean += g.mean()
#                 # r_mean += r.mean()
#
#                 b_std += b.std()
#                 # g_std += g.std()
#                 # r_std += r.std()
#
#                 images.append(path)
#
#             if i % 100 == 1:
#                 print(i)
#                 print(b_mean / i)
#                 # print(g_mean / i)
#                 # print(r_mean / i)
#
#                 print(b_std / i)
#                 # print(g_std / i)
#                 # print(r_std / i)
#
#     return b_mean, b_std, i
#     # return b_mean, g_mean, r_mean, b_std, g_std, r_std, i
#
#
# # b_mean, g_mean, r_mean, b_std, g_std, r_std, i = calc_means(src_ms_rgb)
# # b_mean, b_std, i = calc_means(src_ms_rgb)
# # #
# # print("Result")
# # print(b_mean / i)
# # # print(g_mean / i)
# # # print(r_mean / i)
# #
# # print(b_std / i)
# # print(g_std / i)
# # print(r_std / i)
#
# # a, b, c=sorted(os.walk(root_dir))
#
# ex_dir = r'C:\Users\Enes\Desktop\images'
#
# # for a, b, c in sorted(os.walk(ex_dir)):
# #
# #     print("*********")
# #     print(a)
# #     print(b)
# #     print(c)
#
# # import time
# #
# # start=time.time()
# # # print (len([name for name in os.listdir(src_ms_rgb) if os.path.isfile(os.path.join(src_ms_rgb, name))]))
# #
# # path, dirs, files = next(os.walk(src_ms_rgb))
# #
# # print(len(files))
# #
# # print(time.time()-start)
#
# from utils.dataset import CustomDataset
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# import torch.optim as optim
# from utils import parser, utils, dataset
# from models import networks, losses
# import os
# import numpy as np
# import torch as t
# from skimage import io
# import os
# from torchvision import transforms
# from torch.utils.data import Dataset
#
# # root_dir=r"C:\Users\Enes\PycharmProjects\Term_Project\dataset"
# #
# # ds = dataset.CustomDataset(root_dir, is_segment=False)
# # dataloader = DataLoader(ds, batch_size=1, shuffle=True)
# #
# # for e in range(5):
# #     print(f"Epoch #{e}")
# #     for data in dataloader:
# #         rgb = data[0]
# #         ir = data[1]
#
# # composed = transforms.Compose([transforms.ToTensor()])
# # # transforms.Normalize((0.34, 0.33, 0.35), (0.19, 0.18, 0.18))
# # rgb_dir = os.path.join(root_dir, 'rgb')
# #
# # rgb = (io.imread(os.path.join(rgb_dir, f'{0}.jpg'))) / 255.0
# # print(rgb.shape)
# # # rgb = rgb.transpose(2, 0, 1)
# # rgb = composed(rgb)
# #
# # print(rgb.shape)
#
# # {:0>2d}
#
# train_dir = r"C:\Users\Enes\Desktop\datasets\flir data\FLIR_ADAS_1_3\train"
#
# rgb_dir = r"C:\Users\Enes\Desktop\datasets\flir data\FLIR_ADAS_1_3\train\RGB"
# ir_dir = r"C:\Users\Enes\Desktop\datasets\flir data\FLIR_ADAS_1_3\train\thermal_8_bit"
#
# dst_rgb = r"C:\Users\Enes\Desktop\datasets\new_dataset\flir\rgb3"
# dst_ir = r"C:\Users\Enes\Desktop\datasets\new_dataset\flir\ir3"
#
# # temp_dir=r"C:\Users\Enes\Desktop\datasets\new_dataset\flir\temp"
#
# w=[]
# b=[]
#
# margin_left = 75
# margin_right = 35
# margin_top = 22
# margin_bottom = 60
# _, _, files1 = next(os.walk(rgb_dir))
#
# k = 6570
# others = 0
# # for i in range(1, 10):
# print(len(files1))
#
# # for i in tqdm(range(1, len(files1)+1)):
# #
# #     try:
# #         im = cv2.imread(rgb_dir + f"/FLIR_{i:0>5d}.jpg")
# #         if im.shape[1] == 1280 and im.shape[0] == 1024:
# #
# #             if im.mean() > 250:
# #                 print(f"White image: FLIR_{i:0>5d}.jpg")
# #                 w += [f"FLIR_{i:0>5d}.jpg"]
# #                 continue
# #             elif im.mean()<5:
# #                 print(f"Black image: FLIR_{i:0>5d}.jpg")
# #                 b += [f"FLIR_{i:0>5d}.jpg"]
# #                 continue
# #
# #             k+=1
# #             # RGB
# #             im = cv2.resize(im, (640, 512))
# #             cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)
# #
# #             # IR
# #             ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
# #             ir = ir[margin_top:512 - margin_bottom, margin_left:640 - margin_right]
# #             ir = cv2.resize(ir, (640, 512))
# #             cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)
# #         else:
# #             # To count other res
# #             others +=1
# #
# #     except:
# #         if os.path.isfile(rgb_dir + f"/FLIR_{i:0>5d}.jpg"):
# #             print(f"Error: FLIR_{i:0>5d}.jpg exists but cannot be read")
# #             raise Exception("???")
# #         else:
# #             print(f"No such a file: FLIR_{i:0>5d}.jpg")
# #
# #
# # print("*"*10)
# # print(f"#of black images: {len(b)}")
# # print(f"#of white images: {len(w)}")
# # print(f"#of un-processed images: {others}")
#
#
#
# """         """
# # w=[]
# # b=[]
# #
# # margin_left = 35
# # margin_right = 80
# # margin_top = 12
# # margin_bottom = 78
# # _, _, files1 = next(os.walk(rgb_dir))
# #
# # k=4855
# # others=0
# # # for i in range(1, 10):
# #
# # for i in tqdm(range(1, len(files1)+1)):
# #
# #     try:
# #         im = cv2.imread(rgb_dir + f"/FLIR_{i:0>5d}.jpg")
#         if im.shape[1] == 2048 and im.shape[0] == 1536:
#
#             if im.mean() > 250:
#                 print(f"White image: FLIR_{i:0>5d}.jpg")
#                 w += [f"FLIR_{i:0>5d}.jpg"]
#                 continue
#             elif im.mean()<5:
#                 print(f"Black image: FLIR_{i:0>5d}.jpg")
#                 b += [f"FLIR_{i:0>5d}.jpg"]
#                 continue
#
#             k+=1
#             # RGB
#             im = im[:, 64:2048 -64]
#             im = cv2.resize(im, (640, 512))
#             cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)
#
#             # IR
#             ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
#             ir = ir[margin_top:512 - margin_bottom, margin_left:640 - margin_right]
#             ir = cv2.resize(ir, (640, 512))
#             cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)
# #         else:
# #             # To count other res
# #             others +=1
# #
# #     except:
# #         if os.path.isfile(rgb_dir + f"/FLIR_{i:0>5d}.jpg"):
# #             print(f"Error: FLIR_{i:0>5d}.jpg exists but cannot be read")
# #             raise Exception("???")
# #         else:
# #             print(f"No such a file: FLIR_{i:0>5d}.jpg")
# #
# #
# # print("*"*10)
# # print(f"#of black images: {len(b)}")
# # print(f"#of white images: {len(w)}")
# # print(f"#of un-processed images: {others}")
#
#
#
#
# # def part1():
#     #
#     # w = []
#     # b = []
#     #
#     # margin_left = 160
#     # margin_right = 45
#     # margin_top = 120
#     # margin_bottom = 170
#     #
#     # others=0
#     # k=0
#     # # for i in range(1, 10):
#     # for i in tqdm(range(1, len(files1)+1)):
#     #
#     #     try:
#     #         im = cv2.imread(rgb_dir + f"/FLIR_{i:0>5d}.jpg")
#             if im.shape[1] == 1800 and im.shape[0] == 1600:
#
#                 if im.mean() > 245:
#                     print(f"White image: FLIR_{i:0>5d}.jpg")
#                     w += [f"FLIR_{i:0>5d}.jpg"]
#                     continue
#                 elif im.mean()<5:
#                     print(f"Black image: FLIR_{i:0>5d}.jpg")
#                     b += [f"FLIR_{i:0>5d}.jpg"]
#                     continue
#
#                 k+=1
#                 # RGB
#                 im = im[margin_top:1600 - margin_bottom, margin_left:1800 - margin_right]
#                 im = cv2.resize(im, (640, 512))
#                 cv2.imwrite(dst_rgb + f"/FLIR_{k:0>5d}.jpg", im)
#                 # IR
#                 ir = cv2.imread(ir_dir + f"/FLIR_{i:0>5d}.jpeg")
#                 cv2.imwrite(dst_ir + f"/FLIR_{k:0>5d}.jpg", ir)
#     #         elif im.shape[1] == 1800 and im.shape[0] == 1600:
#     #
#     #         else:
#     #             # To count other res
#     #             others +=1
#     #
#     #     except:
#     #         if os.path.isfile(rgb_dir + f"/FLIR_{i:0>5d}.jpg"):
#     #             print(f"Error: FLIR_{i:0>5d}.jpg exists but cannot be read")
#     #             raise Exception("???")
#     #         else:
#     #             print(f"No such a file: FLIR_{i:0>5d}.jpg")
#     #
#     #
#     # print("*"*10)
#     # print(f"#of black images: {len(b)}")
#     # print(f"#of white images: {len(w)}")
#     # print(f"#of un-processed images: {others}")
#
#     # Counter({(1800, 1600): 4870, (2048, 1536): 1973, (1280, 1024): 1401, (720, 480): 119})
#     # 8363
#     # Counter({(640, 512): 8862})
#     # 8862

import this