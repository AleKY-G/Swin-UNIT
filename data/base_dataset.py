"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    #### methods from vid2vid
    def update_training_batch(self, ratio): # update the training sequence length to be longer      
        seq_len_max = min(128, self.seq_len_max) - (self.opt.n_frames_G - 1)
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (2**ratio))
            #self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (ratio + 1))
            print('--------- Updating training sequence length to %d ---------' % self.n_frames_total)

    def init_frame_idx(self, A_paths):
        self.n_of_seqs = min(len(A_paths), self.opt.max_dataset_size)         # number of sequences to train
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        self.A, self.B, self.I = None, None, None

    def update_frame_idx(self, A_paths, index):
        if self.opt.isTrain:
            if self.opt.dataset_mode == 'pose':                
                seq_idx = np.random.choice(len(A_paths), p=self.folder_prob) # randomly pick sequence to train
                self.frame_idx = index
            else:    
                seq_idx = index % self.n_of_seqs            
            return None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.A, self.B, self.I = None, None, None
            return self.A, self.B, self.I, self.seq_idx

    def init_data(self, t_scales):
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all, flow_ref_all, conf_ref_all = [None]*t_scales, [None]*t_scales, [None]*t_scales, [None]*t_scales
        
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all        
        return fake_B_last, frames_all


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


# def get_img_params(opt, size):
#     w, h = size
#     new_h, new_w = h, w        
#     if 'resize' in opt.preprocess:   # resize image to be loadSize x loadSize
#         new_h = new_w = opt.loadSize            
#     elif 'scaleWidth' in opt.preprocess: # scale image width to be loadSize
#         new_w = opt.loadSize
#         new_h = opt.loadSize * h // w
#     elif 'scaleHeight' in opt.preprocess: # scale image height to be loadSize
#         new_h = opt.loadSize
#         new_w = opt.loadSize * w // h
#     elif 'randomScaleWidth' in opt.preprocess:  # randomly scale image width to be somewhere between loadSize and fineSize
#         new_w = random.randint(opt.fineSize, opt.loadSize + 1)
#         new_h = new_w * h // w
#     elif 'randomScaleHeight' in opt.preprocess: # randomly scale image height to be somewhere between loadSize and fineSize
#         new_h = random.randint(opt.fineSize, opt.loadSize + 1)
#         new_w = new_h * w // h
#     new_w = int(round(new_w / 4)) * 4
#     new_h = int(round(new_h / 4)) * 4    

#     crop_x = crop_y = 0
#     crop_w = crop_h = 0
#     if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
#         if 'crop' in opt.resize_or_crop:      # crop patches of size fineSize x fineSize
#             crop_w = crop_h = opt.fineSize
#         else:
#             if 'Width' in opt.resize_or_crop: # crop patches of width fineSize
#                 crop_w = opt.fineSize
#                 crop_h = opt.fineSize * h // w
#             else:                              # crop patches of height fineSize
#                 crop_h = opt.fineSize
#                 crop_w = opt.fineSize * w // h

#         crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)        
#         x_span = (new_w - crop_w) // 2
#         crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span/3 + x_span)))        
#         crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
#         #crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
#         #crop_y = random.randint(0, np.maximum(0, new_h - crop_h))        
#     else:
#         new_w, new_h = make_power_2(new_w), make_power_2(new_h)

#     flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
#     return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))
    if 'resize' in opt.preprocess:
        osize = [opt.resize_height, opt.resize_width]
        # if "gta2cityscapes" in opt.dataroot:
        #     osize[0] = opt.load_size // 2
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_shortside' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))

    if 'zoom' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if 'patch' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))

    if 'trim' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))

    # if opt.preprocess == 'none':
    ## transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif 'flip' in params:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    
    if normalize:
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img


def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)


def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True




def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A