import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_grouped_dataset, check_path_valid
from PIL import Image
from pathlib import Path
import numpy as np

class TemporalDataset(BaseDataset):
    '''
    create datasets for frames
    opt.n_frames_G: read how many frames a time
    '''
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = Path(opt.dataroot) / (opt.phase + '_' + opt.dirA) 
        self.dir_B = Path(opt.dataroot) / (opt.phase + '_' + opt.dirB) 
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        check_path_valid(self.A_paths, self.B_paths)

        self.n_of_seqs = len(self.A_paths)                 # number of sequences to train       
        self.seq_len_max = max([len(A) for A in self.A_paths])        
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        '''
        读取第i个序列，抽取其中的n_frames_total帧，在channel维度拼接
        '''
        tG = self.opt.n_frames_G
        A_paths = self.A_paths[index % self.n_of_seqs]
        B_paths = self.B_paths[index % self.n_of_seqs]          

        # get_video_params: setting parameters different between train/test
        # get_video_params2: setting parameters same for train/test start=0 tstep=1
        n_frames_total, start_idx, t_step = get_video_params2(self.opt, self.n_frames_total, len(A_paths), index)     
        
        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')        
        params = get_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB

        # read in images
        A = B = inst = 0
        A_pathlist , B_pathlist = [],[]
        for i in range(n_frames_total):            
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]
            A_pathlist.append(A_path)
            B_pathlist.append(B_path)

            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)            
            Bi = self.get_image(B_path, transform_scaleB)
            
            A = Ai if i == 0 else torch.cat([A, Ai], dim=0)            
            B = Bi if i == 0 else torch.cat([B, Bi], dim=0)            

        return_list = {'A': A, 'B': B, 'inst': inst, 'A_paths': A_pathlist, 'B_paths': B_pathlist}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)     
        A_scaled = transform_scaleA(A_img)
        # if is_label:
        #     A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TemporalDataset'

    def init_data_params(self, data, n_gpus, tG):
        opt = self.opt
        # tG: 每次输入到生成器多少帧 例如3
        # n_frames_load：每次加载到gpu多少帧  例如2 例如1
        # t_len:时间感受野？              2+3-1=4  1+3-1=3
        _, n_frames_total, self.height, self.width = data['B'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1        
        n_frames_total = n_frames_total // opt.output_nc #sequences中有多少帧
        # print("bydguiv",n_frames_total)
        n_frames_load = opt.max_frames_per_gpu * n_gpus #1               # number of total frames loaded into GPU at a time for each batch
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1) # 1
        self.t_len = n_frames_load + tG - 1  # 3                       # number of loaded frames plus previous frames plus following frames
        return n_frames_total-self.t_len+1, n_frames_load, self.t_len

    # def init_data_params(self, data, tG):
    #     opt= self.opt
    #     _, n_frames_total, self.heigh, self.width = data['B'].size()
    #     n_frames_total = n_frames_total // opt.output_nc

    
    def prepare_data(self, data, i, input_nc, output_nc):
        t_len, height, width = self.t_len, self.height, self.width
        input_A = input_B = A_paths = B_paths = None
        if t_len == 1 and self.opt.isTrain:
            input_A = (data['A'][:, i*input_nc:(i+1)*input_nc, ...]).view(t_len, input_nc, height, width)
            input_B = (data['B'][:, i*input_nc:(i+1)*input_nc, ...]).view(t_len, input_nc, height, width)
            A_paths = data['A_paths'][i:i+t_len]
            B_paths = data['B_paths'][i:i+t_len]
        else:
        # 5D tensor: batchSize, # of frames, # of channels, height, width
            input_A = (data['A'][:, i*input_nc:(i+t_len)*input_nc, ...]).view(-1, t_len, input_nc, height, width)
            input_B = (data['B'][:, i*output_nc:(i+t_len)*output_nc, ...]).view(-1, t_len, output_nc, height, width)                
            # inst_A = (data['inst'][:, i:i+t_len, ...]).view(-1, t_len, 1, height, width) if len(data['inst'].size()) > 2 else None
            A_paths = data['A_paths'][i:i+t_len]
            B_paths = data['B_paths'][i:i+t_len]
        return_list = {'A':input_A,'B': input_B, 'A_paths': A_paths, 'B_paths': B_paths}
        return return_list


def get_video_params2(opt, n_frames_total, cur_seq_len, index):
    # tG = opt.n_frames_G
    n_frames_total = min(n_frames_total, cur_seq_len)
    start_idx = 0 
    t_step = 1
    return n_frames_total, start_idx, t_step


def get_video_params(opt, n_frames_total, cur_seq_len, index):
    tG = opt.n_frames_G
    n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)
    if opt.isTrain: 
        #获取视频中一共包含多少帧，从start_idx开始，每隔t_step步抽取一帧       
        n_gpus = opt.n_gpus_gen if opt.batch_size == 1 else 1       # number of generator GPUs for each batch
        n_frames_per_load = opt.max_frames_per_gpu * n_gpus        # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs         
        n_frames_total = n_frames_per_load * n_loadings + tG - 1   # rounded overall number of frames to read from the sequence
        
        max_t_step = min(opt.max_t_step, (cur_seq_len-1) // (n_frames_total-1))
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame        

        start_idx = np.random.randint(offset_max)                 # offset for the first frame to load
        if opt.debug:
            print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                % (n_frames_total, start_idx, t_step))
    else:
        start_idx = 0
        t_step = 1   
    return n_frames_total, start_idx, t_step
