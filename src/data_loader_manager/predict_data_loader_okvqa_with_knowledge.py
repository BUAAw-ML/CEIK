import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64
from copy import deepcopy
from time import time
from datetime import datetime
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


import logging
logger = logging.getLogger(__name__)
from utils.cache_system import save_cached_data, load_cached_data


from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from transformers import ViTFeatureExtractor

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval

from data_loader_manager.data_loader_okvqa_with_knowledge import DataLoaderOKVQAWithKnowledge
from data_loader_manager.datasets import *

class PredictDataLoaderOKVQAWithKnowledge(DataLoaderOKVQAWithKnowledge):
    '''
    Data loader for our OK-VQA dataset
    Knowledge passages are incorporated
    '''

    def __init__(self, config):
        DataLoaderOKVQAWithKnowledge.__init__(self, config)

    def set_dataloader(self, input):
        """
        This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'data': self.data.vqa_data.train if 'vqa_data_with_dpr_output' not in self.data.keys() \
                    else self.data.vqa_data_with_dpr_output.train,
            'passages': self.data.passages,
            'vinvl_features': self.data.vinvl_features,
            'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'train',
        }
        self.train_dataset = globals()[self.config.data_loader.dataset_type](self.config, train_dataset_dict)
        # for i in self.train_dataset:
        #     pprint(i)
        #     input()
        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
        )

        test_dataset_dict = {
            'data': self.data.vqa_data.test,#
            'passages': self.data.passages,
            'vinvl_features': self.data.vinvl_features,
            'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'test',
        }

        self.test_dataset = globals()[self.config.data_loader.dataset_type](self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size if self.config.mode=='train' else self.config.test.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )


        ###################################predict_dataloader###################################

        predict_vqa_data = EasyDict({})
        predict_vqa_data['data_items'] = []

        for item in self.data.vqa_data.test.data_items:
            # print(isinstance((item['img'] == input['img']), bool))
            # difference = cv2.subtract(item['img'], input['img'])
            # print(not np.any(difference)) #if difference is all zeros it will return False
            if item['img'].shape == input['img'].shape:
                difference = cv2.subtract(item['img'], input['img'])
                if not np.any(difference):#(item['img'] == input['img']).all():
                    item['question'] = input['question']
                    # item['question_id'] = 99999999999999999
                    predict_vqa_data['data_items'].append(item)
                    break
       
        assert len(predict_vqa_data) == 1
        # print(self.data.vqa_data.test.data_items)
        print(predict_vqa_data)

        predict_dataset_dict = {
            'data': predict_vqa_data,#
            'passages': self.data.passages,
            'vinvl_features': self.data.vinvl_features,
            'ocr_features': self.data.ocr_features,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'test',
        }

        self.predict_dataset = globals()[self.config.data_loader.dataset_type](self.config, predict_dataset_dict)

        predict_sampler = SequentialSampler(self.predict_dataset)
        self.predict_dataloader = DataLoader(
            self.predict_dataset,
            sampler=predict_sampler,
            batch_size=self.config.valid.batch_size if self.config.mode=='train' else self.config.test.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )


        # # for i in self.test_dataloader:
        #     print(i)
        #     input()
        logger.info('[Data Statistics]: test data loader: {}'.format(
                                len(self.predict_dataloader)))
