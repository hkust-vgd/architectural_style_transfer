"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


#Training data structure:

- data_root
    - class_a
        - train
            - XXXX.jpg
            ....
        - test
            - XXXX.jpg            
            ....
    - class_b
        - train
            - XXXX.jpg
            ....
        - test
            - XXXX.jpg            
            ....

"""

import argparse
import sys
import shutil
import os
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import tensorboardX
from utils import get_all_data_loaders, get_config, write_loss, write_2images, Timer
from trainer import DOT_Trainer
import time


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/day2golden_fg.yaml', help='Path to the config file')
    parser.add_argument('--save_name', type=str, default='.', help="Name to save the training results (e.g., day2golden_fg)")   
    parser.add_argument("--resume", type=str, help='Directory with checkpoints(gen,dis,optimizer) to be resumed training (e.g., outputs/day2golden_fg/ckpts)')
    opts = parser.parse_args()

    # Set up GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    class_a = config['class_a']
    class_b = config['class_b']


    # Setup model and data loader
    trainer = DOT_Trainer(config)
    trainer.build_optimizer(config)
    trainer.cuda()

    # Preprocess and load training data
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config, class_a, class_b)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

    # Setup logger and output folders
    log_directory = os.path.join("./logs", opts.save_name)
    output_directory = os.path.join("./outputs", opts.save_name)
    image_directory = os.path.join(output_directory, 'images')
    checkpoint_directory = os.path.join(output_directory, 'ckpts')

    assert not os.path.exists(output_directory), 'There exists the output directory %s'%(output_directory)
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(checkpoint_directory, exist_ok=True)
    train_writer = tensorboardX.SummaryWriter(log_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(opts.resume, hyperparameters=config) if opts.resume is not None else 0
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            # Main training code
            with Timer("Elapsed time in update: %f"): # record iter update time
                dis_loss = trainer.dis_update(images_a, images_b, config, iterations)
                gen_loss = trainer.gen_update(images_a, images_b, config, iterations)
                torch.cuda.synchronize()
                trainer.update_learning_rate()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print('D loss: %.4f\t G loss: %.4f' % (dis_loss, gen_loss))
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))               
                print("Time elapsed: %.4f minutes" %((time.time() - start_time)/60.0))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) == 1 or (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

            if (iterations + 1) % config['image_display_iter'] == 0 or iterations == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                print("--- Time elapsed: %.4f minutes ---" %((time.time() - start_time)/60.0))
                sys.exit('Finish training')

