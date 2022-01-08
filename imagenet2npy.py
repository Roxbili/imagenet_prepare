#-*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import os, sys
import argparse
from multiprocessing import Pool
from tqdm import tqdm

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='~/dataset/ILSVRC2012', type=str, metavar='PATH',
                        help='The ImageNet dataset path')
    parser.add_argument('--dst', default='~/dataset/ILSVRC2012_npy', type=str, metavar='PATH',
                        help='Target ImageNet .npy format path')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='Workers to read and save npy images')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip the .npy file if exists')
    args = parser.parse_args()
    return args

def _check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        # print('Created directory %s' % dir_path)

def img2npy(path):
    '''load image from IMG_EXTENSIONS, save to npy file

        Args:
            path = (img_path_src, img_path_dst)
    '''

    # imap is inconvenient for multiple parmeters, so just use path for src and dst
    img_path_src, img_path_dst = path
    img = Image.open(img_path_src).convert('RGB')
    img = np.array(img)
    np.save(img_path_dst, img)

def convert(args):
    p = Pool(args.workers)

    for root, dirs, files in os.walk(args.src):
        root_dst = root.replace(args.src, args.dst)

        # check if the directory exists
        for dir_name in dirs:
            dir_path = os.path.join(root_dst, dir_name)
            _check_dir_exists(dir_path)

        # save npy
        imgs_path_src = []
        imgs_path_dst = []
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                src = os.path.join(root, f)
                dst = os.path.join(root_dst, base + '.npy')

                if args.resume: # skip the existing file
                    if os.path.exists(dst):
                        continue
                imgs_path_src.append(src)
                imgs_path_dst.append(dst)

        if len(imgs_path_src) != 0:
            list(tqdm(p.imap(img2npy, zip(imgs_path_src, imgs_path_dst)), total=len(imgs_path_src), desc=root.replace(args.src, '')))

    p.close()
    p.join()

if __name__ == '__main__':
    args = _parse_args()
    args.src = os.path.expanduser(args.src)
    args.dst = os.path.expanduser(args.dst)
    _check_dir_exists(args.dst)

    convert(args)