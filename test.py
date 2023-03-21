
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm
import util.util as util
import torch
from thop import profile
from thop import clever_format

from models.cut_model import CUTModel


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    # create a model given opt.model and other options
    model = create_model(opt)
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(
        opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.epoch))

    ##############
    # g = CUTModel(opt)
    # input = torch.randn(1, 3, 768, 1280).cuda()
    # macs, params = profile(g, inputs=(input, ))  # macs = 0.5 * flops
    # macs, params = clever_format([macs, params], "%.3f")
    # print("macs, params:", macs, params)
    ##############

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            # regular setup: load and print networks; create schedulers
            model.setup(opt)

            model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        # if i < 9250:
        #     continue
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    # webpage.save()  # save the HTML
