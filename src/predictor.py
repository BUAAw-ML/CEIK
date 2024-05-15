"""
main.py:  
    Start functions
        - Read json/jsonnet self.config files
        - Parse args and override parameters in self.config files
        - Find selected data loader and initialize
        - Run Trainer to perform training and testing
"""

import os
import argparse
from tabnanny import verbose
import torch
import wandb
import json
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
logger = logging.getLogger(__name__)

from utils.config_system import process_config
from utils.dirs import *
from utils.cuda_stats import print_cuda_statistics
from utils.seed import set_seed
from utils.metrics_log_callback import MetricsHistoryLogger

from data_loader_manager import *
from trainers import *


class Predictor():
    def __init__(self):


        # super().__init__()
        args = self.parse_args_sys()
        print(args)

        self.config = self.initialization(args)
        if self.config is None:
            exit(0)

        pprint(self.config)

        if self.config.seed:
            set_seed(self.config.seed)
            seed_everything(self.config.seed, workers=True)
            # sets seeds for numpy, torch and python.random.
            logger.info(f'All seeds have been set to {self.config.seed}')
        
        DataLoaderWrapper = globals()[self.config.data_loader.type]
        if DataLoaderWrapper is not None:
            # init data loader
            self.data_loader_manager = DataLoaderWrapper(self.config)
        else:
            raise ValueError(f"Data loader {self.config.data_loader.type} not found")
        
        # Checkpoint Callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.saved_model_path,
            every_n_epochs=self.config.train.save_interval,
            save_top_k=self.config.train.additional.save_top_k,
            monitor=self.config.train.additional.save_top_k_metric if 'save_top_k_metric' in self.config.train.additional.keys() else None,
            mode=self.config.train.additional.save_top_k_mode,
            filename='model_{epoch}',
            save_last=False,
            verbose=True,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=True,
        )
        metrics_history_logger = MetricsHistoryLogger()

        # Get plugins
        plugin_names = self.config.train.additional.plugins
        plugins = [globals()[plugin_name]() for plugin_name in plugin_names]

        additional_args = {
            'accumulate_grad_batches': self.config.train.additional.gradient_accumulation_steps,
            "default_root_dir": self.config.saved_model_path,
            'max_epochs': self.config.train.epochs,
            # 'limit_train_batches': 2 if self.config.data_loader.dummy_dataloader else 1.0,
            # 'limit_val_batches': 2 if self.config.data_loader.dummy_dataloader else 1.0,
            # o'limit_test_batches': 2 if self.config.data_loader.dummy_dataloader else 1.0,
            # 'logger': [tb_logger, wandb_logger, metrics_history_logger],
            'logger': [metrics_history_logger],
            'callbacks': [checkpoint_callback],
            'plugins': plugins,
            'log_every_n_steps': 10,
            # 'accelerator': "cpu", 
            # 'strategy': "ddp",
            # 'devices': 2,
        }

        self.trainer = Trainer.from_argparse_args(args, **additional_args)
        logger.info(f"arguments passed to trainer: {str(args)}")
        logger.info(f"additional arguments passed to trainer: {str(additional_args)}")
        

        self.checkpoint_to_load = self.get_checkpoint_model_path(
            saved_model_path=self.config.saved_model_path,
            load_model_path=self.config.test.load_model_path, 
            load_epoch=self.config.test.load_epoch, 
            load_best_model=self.config.test.load_best_model
        )
        if not self.checkpoint_to_load:
            raise FileNotFoundError("No checkpoint found. Please check your self.config file.")

        
        # init data loader manager
        self.data_loader_manager.build_dataset()
        

    def predict(self, input):
        self.data_loader_manager.set_dataloader(input)
        # init train excecutor
        Train_Executor = globals()[self.config.train.type]
        executor = Train_Executor(self.config, self.data_loader_manager)
        # Start 
        res = self.trainer.predict(
            executor,
            ckpt_path=self.checkpoint_to_load,
        )
        return res


    def initialization(self, args):
        assert args.mode in ['train', 'test', 'run']
        # ===== Process Config =======
        config = process_config(args)

        print(config)
        if config is None:
            return None
        # Create Dirs
        dirs = [
            config.log_path,
        ]
        if config.mode == 'train':
            dirs += [
                config.saved_model_path,
                config.imgs_path,
                config.tensorboard_path
            ]
        if config.mode == 'test':
            dirs += [
                config.imgs_path,
                config.results_path,
            ]

        delete_confirm = 'n'
        if config.reset and config.mode == "train":
            # Reset all the folders
            print("You are deleting following dirs: ", dirs, "input y to continue")
            delete_confirm = input()
            if delete_confirm == 'y':
                for dir in dirs:
                    try:
                        delete_dir(dir)
                    except Exception as e:
                        print(e)
                # Reset load epoch after reset
                config.train.load_epoch = 0
            else:
                print("reset cancelled.")

        create_dirs(dirs)
        print(dirs)

        logger.info(f'Initialization done with the config: {str(config)}')
        return config

    def get_checkpoint_model_path(self, saved_model_path, load_epoch=-1, load_best_model=False, load_model_path=""):

        if load_model_path:
            path_save_model = load_model_path
            if not os.path.exists(path_save_model):
                raise FileNotFoundError("Model file not found: {}".format(path_save_model))
        else:
            if load_best_model:
                file_name = "best.ckpt"
            else:
                if load_epoch == -1:
                    file_name = "last.ckpt"
                else:
                    file_name = "model_{}.ckpt".format(load_epoch)

            path_save_model = os.path.join(saved_model_path, file_name)
            if not os.path.exists(path_save_model):
                logger.warning("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
                logger.info("**First time to train**")
                return '' # return empty string to indicate that no model is loaded
            else:
                logger.info("Loading checkpoint from '{}'".format(path_save_model))
        return path_save_model

    def parse_args_sys(self): #args_list=None
        # parse the path of the json config file

        arg_parser = argparse.ArgumentParser(description="")
        arg_parser.add_argument('config', metavar='config_json_file', default='', help='The Configuration file in json format')
        arg_parser.add_argument('--DATA_FOLDER', type=str, default='', help='The path to data.')
        arg_parser.add_argument('--EXPERIMENT_FOLDER', type=str, default='', help='The path to save experiments.')
        
        arg_parser.add_argument('--mode', type=str, default='test', help='train/test')
        arg_parser.add_argument('--reset', action='store_true', default=False, help='Reset the corresponding folder under the experiment_name')
        
        arg_parser.add_argument('--experiment_name', type=str, default='predict', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
        arg_parser.add_argument("--tags", nargs='*', default=[], help="Add tags to the wandb logger")
        arg_parser.add_argument('--modules', type=str, nargs="+", default=[""], help='Select modules for models. See training scripts for examples.')
        arg_parser.add_argument('--log_prediction_tables', action='store_true', default=False, help='Log prediction tables.')

        # ===== Testing Configuration ===== #
        arg_parser.add_argument('--test_batch_size', type=int, default=-1)
        arg_parser.add_argument('--test_evaluation_name', type=str, default="")
        
        
        arg_parser = Trainer.add_argparse_args(arg_parser)

        arg_parser.add_argument(
            "--opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )
        # print(args_list)

        args = arg_parser.parse_args()
        # if args_list is None:
            
        # else:
        #     args = arg_parser.parse_args(args_list)


        return args

