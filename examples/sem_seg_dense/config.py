import os
import sys
import datetime
import argparse
import shutil
import random
import numpy as np
import torch
import logging
import logging.config
import pathlib
import glob
import time
import uuid
from torch.utils.tensorboard import SummaryWriter


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
        #parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
        # ----------------- Base
        parser.add_argument('--phase', default='test', type=str, help='train or test(default)')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--exp_name', type=str, default='', help='experiment name')
        parser.add_argument('--job_name', type=str, default='', help='full name of exp directory (exp_name + timetamp')
        parser.add_argument('--root_dir', type=str, default='log', help='the dir of all experiment results')

        # ----------------- Dataset related
        parser.add_argument('--data_dir', type=str, default='/data/deepgcn/S3DIS',
                            help="data dir, will download dataset here automatically")
        parser.add_argument('--area', type=int, default=5, help='the cross validated area of S3DIS')
        parser.add_argument('--in_channels', default=9, type=int, help='the channel size of input point cloud ')

        # ----------------- Training related
        parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:16)')
        parser.add_argument('--total_epochs', default=100, type=int, help='number of total epochs to run')
        parser.add_argument('--save_freq', default=1, type=int, help='save model per num of epochs')
        parser.add_argument('--iter', default=0, type=int, help='number of iteration to start')
        parser.add_argument('--lr_adjust_freq', default=20, type=int, help='decay lr after certain number of epochs')
        parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.5, type=float, help='learning rate decay')
        parser.add_argument('--print_freq', default=100, type=int, help='print frequency of training (default: 100)')
        parser.add_argument('--eval_freq', default=1, type=int,
                            help='evaluation frequency of training (default: 1). Set as -1 to disable evaluation')
        parser.add_argument('--n_gpus', type=int, help='Number of GPUs')
        parser.add_argument('--seed', type=int, default=0, help='random seed')

        # ----------------- Testing related
        parser.add_argument('--no_clutter', action='store_true', help='no clutter? set --no_clutter if ture.')
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')

        # ----------------- Model related

        parser.add_argument('--k', default=16, type=int, help='neighbor num (default:16)')
        parser.add_argument('--knn_criterion', default='xyz', type=str, help='xyz, color or MLP(TODO)')
        parser.add_argument('--block', default='plain', type=str, help='graph backbone block type {plain, res, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='batch', type=str, help='{batch, instance, None} normalization')
        parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser.add_argument('--n_blocks', default=28, type=int, help='number of basic blocks')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

        # dilated knn
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
        args = parser.parse_args()

        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        #args.device = torch.cuda.set_device(args.local_rank)
        self.args = args

        # ===> generate log dir
        if self.args.phase == 'train':
            # generate exp_dir when pretrained model does not exist, otherwise continue training using the pretrained
            if not self.args.pretrained_model:
                self._generate_exp_directory()
            else:
                self.args.exp_dir = os.path.dirname(os.path.dirname(self.args.pretrained_model))
                self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")

            # logger
            #self.args.writer = SummaryWriter(log_dir=self.args.exp_dir)
            # loss
            self.args.epoch = -1
            self.args.step = -1

        else:
            self.args.job_name = os.path.basename(args.pretrained_model).split('.')[0]
            self.args.exp_dir = os.path.dirname(args.pretrained_model)
            self.args.res_dir = os.path.join(self.args.exp_dir, 'result', self.args.job_name)
            pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

        self._configure_logger()
        self._print_args()
        self._set_seed(self.args.seed)

    def get_args(self):
        return self.args

    def _generate_exp_directory(self):
        """Creates checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        if not self.args.exp_name:
            self.args.exp_name = '{}-{}-{}-n{}-C{}-norm_{}-lr{}-B{}' \
                .format(os.path.basename(os.getcwd()),  # using the basename as the experiment prefix name.
                        self.args.block, self.args.conv, self.args.n_blocks, self.args.n_filters,
                        self.args.norm, self.args.lr, self.args.batch_size)
        if not self.args.job_name:
            self.args.job_name = '_'.join([self.args.exp_name, timestamp, str(uuid.uuid4())])
        self.args.exp_dir = os.path.join(self.args.root_dir, self.args.job_name)
        self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")
        self.args.code_dir = os.path.join(self.args.exp_dir, "code")
        pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.code_dir).mkdir(parents=True, exist_ok=True)
        # ===> save scripts
        scripts_to_save = glob.glob('*.py')
        if scripts_to_save is not None:
            for script in scripts_to_save:
                dst_file = os.path.join(self.args.code_dir, os.path.basename(script))
                shutil.copyfile(script, dst_file)

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}:{}".format(arg, content))
        logging.info("==========     args END    =============")
        logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevel))

            # configure logger to display and save log data
        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(os.path.join(self.args.exp_dir,
                                                        '{}.log'.format(os.path.basename(self.args.job_name))))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info("saving log, checkpoint and back up code in folder: {}".format(self.args.exp_dir))

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



