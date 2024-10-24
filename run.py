import argparse
import logging
import os
import sys
import time
import importlib
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.Metrics import Metrics
from layers import GraphBuilder
from utils import Utils, Constants
from utils.Optim import ScheduledOptim
from helpers.BaseLoader import BaseLoader, DataLoader
from helpers.BaseRunner import BaseRunner


def parse_global_args(parser):
    parser.add_argument('-data_name', default='twitter')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-train_rate', type=float, default=0.8)
    parser.add_argument('-valid_rate', type=float, default=0.1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    return parser


def main(model_class):
    logging.info('-' * 45 + 'BEGIN: ' + Utils.get_time() + ' ' + '-' * 45)
    # Fix the seed
    Utils.init_seed()

    logging.info(Utils.format_arg_str(args, exclude_lst=[]))

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    # Set Loader
    data_loader = BaseLoader(args)
    user_size, total_cascades, timestamps, train, valid, test = data_loader.split_data(
                                                                           args.train_rate,
                                                                           args.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=args.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=args.batch_size, load_dict=True, cuda=False)

    # # 加载友谊网络
    # logging.info("[ - Loading the friendship network]")
    # relation_graph = GraphBuilder.build_friendship_network(data_loader)
    #
    # # 加载扩散超图的列表
    # logging.info("[ - Loading the hyper diffusion network list]")
    # hyper_graph_list = GraphBuilder.build_diff_hyper_graph_list(total_cascades, timestamps, user_size)

    # ============ Preparing Model ============#
    model = model_class(args, data_loader)

    # Run Model
    runner = BaseRunner(model)
    runner.run(model, train_data, valid_data, test_data, args)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='DyHGCN', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()

    # Dynamically import the specified model module
    try:
        model_module = importlib.import_module(f'models.{init_args.model_name}')
        # Assuming the class has the same name as the module, dynamically get the class
        model_class = getattr(model_module, init_args.model_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module models/{init_args.model_name} not found.")
    except AttributeError:
        raise ValueError(f"Model class {init_args.model_name} not found in module models/{init_args.model_name}.")

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_class.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.data_name]
    # for arg in ['lr', 'l2'] + model_class.extra_log_args:
    #     log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    args.log_file = 'log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    args.model_path = 'saved/{}/{}.pt'.format(init_args.model_name, log_file_name)

    Utils.check_dir(args.log_file)
    Utils.check_dir(args.model_path)
    logging.basicConfig(filename=args.log_file, level='INFO')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main(model_class)
