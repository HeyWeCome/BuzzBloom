import logging
import os
from datetime import datetime
from typing import List

import torch
import random
import numpy as np


def init_seed(seed=1506):
    """
    Fix some random seed for reproducibility.
    :param seed: the number of random seed
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_time():
    """
    Get the current time
    """
    return datetime.now().strftime('%y-%m-%d %H:%M:%S')


def check(check_list):
    """
    observe selected tensors during training.
    :param check_list:
    :return:
    """
    logging.info('')
    for i, t in enumerate(check_list):
        d = np.array(t[1].detach().cpu())
        logging.info(os.linesep.join(
            [t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
        ) + os.linesep)


def check_dir(file_name):
    """
    check whether dir exist
    :param file_name:
    :return:
    """
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str

def format_float_for_filename(value):
    """
    Formats a float into a string suitable for a filename, replacing '.' with '_'.
    Example: 0.1 -> "0_1", 1e-05 -> "1e-05"
    """
    return str(value).replace('.', '_')
