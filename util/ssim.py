#calc ssim using sklearn
#nedd t change
import numpy as np
import time
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import os
import sys
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentPaarser(
        description='if calc dir, the img1 and img2 must have the same name')
    parser.add_argument('--image', default=False)
    parser.add_argument('--img1_dir', default='/data1/2021/lyf/GcGAN/results/SYN_d2n_rot/test_latest/images')
    parser.add_argument('--img2_dir', default='/data1/2021/lyf/GcGAN/results/SYN_d2n_rot/test_latest/images')
    args = parser.parse_args()
    return args

def cal_ssim_psnr(x,y):
    img1 = Image.open(x)
    w, h = img1.size
    img1 = np.array(img1).astype(np.float32)
    img2 = Image.open(y)
    img2 = img2.resize((w, h))
    # img2 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2 = np.array(img2).astype(np.float32)
    ssim_skimage = structural_similarity(img1, img2, channel_axis=2,data_range=255,gaussian_weights=True)
    psnr=peak_signal_noise_ratio(img1/255.0,img2/255.0)
    return ssim_skimage,psnr

def cal_ssim_bt_dirs(dir1, dir2):
    begin = time.time()
    files=os.listdir(dir1)
    ssim, psnr, len_files = 0, 0, 0
    for file in files:
        file1 = os.path.join(dir1, file)
        file2 = os.path.join(dir2, file)

        if os.path.exists(file2):
            ssim_,psnr_=cal_ssim_psnr(file1,file2)
            ssim+=ssim_
            psnr+=psnr_
            len_files+=1
    print("# of pair images:", len_files)
    ssim/=len_files
    psnr/=len_files
    
    cal_time = (time.time()-begin) /len(files)
    print("ssim_skimage=%f (%f ms)" % (ssim, cal_time*1000))
    print("psnr_skimage=%f (%f ms)" % (psnr, cal_time*1000))
    return ssim, psnr


if __name__ == '__main__':
    args = parse_args()
    dir1=args.img1_dir
    dir2=args.img2_dir
    cal_ssim_bt_dirs(dir1, dir2)
