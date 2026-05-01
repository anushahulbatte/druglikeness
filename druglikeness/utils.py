import argparse
import json
import os
import yaml
from pathlib import Path
from typing import Union
import logging
import torch
from sklearn import metrics
import numpy as np

logger = logging.getLogger('utils')

class Runner:

    def __init__(self, args):
        self.args = args
        self.config = Config()
        self.config.update(args.config_path)
    
    def init_torch(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu
        # # device_num = self.config.main_device
        # device_num = 0
        # torch.cuda.set_device(device_num)
        # self.device = torch.device(f'cuda:{device_num}')
        # torch.set_num_threads(3)

        self.device = torch.device('cpu')
        torch.set_num_threads(3)
        # torch.multiprocessing.set_start_method('spawn')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Config:

    def __init__(self, arg: Union[str, Path, dict, argparse.Namespace] = None):
        self.update(arg)
    
    def from_yaml(self, config_path, ignore_none):
        print(config_path)
        if not os.path.exists(config_path):
            raise FileExistsError(OSError)
        self.file_path = config_path
        with open(self.file_path, encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.from_dict(config_dict, ignore_none)

    def from_json(self, config_path: Union[str, Path], ignore_none):
        with open(config_path) as config_file:
            config_dict = json.load(config_file)
        self.from_dict(config_dict, ignore_none)
    
    def from_dict(self, config_dict: dict, ignore_none):
        for key, value in config_dict.items():
            if value is None and ignore_none:
                continue
            setattr(self, key, value)

    def update(self, arg: Union[str, Path, dict, argparse.Namespace], ignore_none=False) -> None:
        if isinstance(arg, (str, Path)):
            if str(arg).endswith('.json'):
                self.from_json(arg, ignore_none)
            elif str(arg).endswith('.yaml'):
                self.from_yaml(arg, ignore_none)
            else:
                raise TypeError(f'Can not update Config from type {type(arg)}')

        elif isinstance(arg, dict):
            self.from_dict(arg, ignore_none)
        elif isinstance(arg, argparse.Namespace):
            self.from_dict(arg.__dict__, ignore_none)
        elif arg is not None:
            raise TypeError(f'Can not update Config from type {type(arg)}')
    
    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = [str(i) for i in v]
            else:
                d[k] = v
        return d

    def to_json(self, fp: Path):
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, 'w') as f:
            d = self.to_dict()
            json.dump(d, f)
    
    def to_yaml(self, fp: Path):
        with open(fp, encoding='utf-8', mode='w') as f:
            d = self.to_dict()
            return yaml.dump(d, stream=f, allow_unicode=True)

    def valid_term(self, term: str, type_='generic') -> bool:
        """ Determine whether a term in the config is valid.
        """
        invalid_list = ['none', 'None', 'null', 'Null']
        if type_ == 'bool':
            assert type(getattr(self, term)) is bool, f"Non-bool value for a bool config term '{term}'"
        elif not hasattr(self, term):
            logger.warning(f"The term '{term}' does not exist in config, and recognized as False. "
                           f"This may lead to unpredictable results!")
            return False
        elif type_ == 'path':
            if getattr(self, term) or getattr(self, term) == '':
                return True
        elif getattr(self, term):
            return getattr(self, term) not in invalid_list
        return False

    def get(self, key: str, default_value):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            return default_value


def set_logger(save_path=None, debug=False):
    logger = logging.getLogger('utils')
    logger.handlers = []
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    fh_fmt = logging.Formatter('[%(asctime)s]  %(message)s')
    sh_fmt = logging.Formatter('[%(asctime)s] [%(levelname)s]  %(message)s')
    if save_path:
        fh = logging.FileHandler(save_path, 'a')
        fh.setFormatter(fh_fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(sh_fmt)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    
    return logger


def cal_metric(label, pred, mask):
    num_tasks = label.shape[1]
    result = []
    for i in range(num_tasks):
        task_mask = mask[:, i]
        if not (~task_mask).all():
            task_label = label[:, i][task_mask]
            task_pred = pred[:, i][task_mask]
            if len(np.unique(task_label)) > 1:
                result.append(metrics.roc_auc_score(task_label, task_pred))
            else:
                logger.warning(f'Skipping metric computation for task {i} due to lack of class variance.')
                pass
    return result

def cal_specdl_metric(label, pred):
    if label.shape != pred.shape:
        raise ValueError('label and pred must have the same shape')

    mask = ~np.isnan(label)    
    return cal_metric(label, pred, mask)

def cal_generaldl_metric(label, pred_raw):
    if label.shape[0] != pred_raw.shape[0]:
        raise ValueError('label and pred must have the same length')
    mask = ~np.isnan(label)
    subset_idx = mask.argmax(axis=1)

    pred = np.full_like(label, np.nan)
    for i in range(len(pred_raw)):
        pred[i, subset_idx[i]] = pred_raw[i]

    return cal_metric(label, pred, mask)
