import argparse
import logging
import os
import sys
import importlib
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from utils import Utils
from helpers.BaseLoader import BaseLoader, collate_fn
from helpers.BaseRunner import BaseRunner


def parse_global_args(parser):
    """
    Defines and parses global arguments used across all models.
    """
    parser.add_argument('--data_name', type=str, default='memetracker', help='Name of the dataset.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation.')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of the model (embeddings, hidden layers).')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Proportion of data for training.')
    parser.add_argument('--valid_rate', type=float, default=0.1, help='Proportion of data for validation.')
    parser.add_argument('--n_warmup_steps', type=int, default=1000, help='Number of warm-up steps for the optimizer.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use (e.g., "0"), set to "" for CPU.')
    parser.add_argument('--filter_num', type=int, default=1, help='Minimum length of a cascade to be included.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading. 0 for main process.')
    return parser


def main(model_class, args):
    """
    The main function to set up and run the experiment.
    """
    logging.info('-' * 45 + ' BEGIN: ' + Utils.get_time() + ' ' + '-' * 45)

    # Initialize random seeds for reproducibility
    Utils.init_seed(seed=args.seed)
    logging.info(Utils.format_arg_str(args, exclude_lst=[]))

    # ====================  Add TF32 setting ====================
    # Enable TensorFloat32 for better performance on supported GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        logging.info("Enabling TensorFloat32 for matrix multiplications.")
        torch.set_float32_matmul_precision('high')
    # ====================================================================

    # Set device (GPU or CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = args.gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if use_cuda else torch.device('cpu')
    logging.info(f'Device: {args.device}')

    # Load and split data
    data_loader = BaseLoader(args)
    user_size, _, _, train_dataset, valid_dataset, test_dataset = data_loader.split_data(
        args.train_rate,
        args.valid_rate,
        load_dict=True
    )

    # Create data loaders for each set
    train_data = TorchDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )
    valid_data = TorchDataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )
    test_data = TorchDataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )

    # Prepare model
    model = model_class(args, data_loader)
    logging.info(f"Model: \n{model}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ====================  Add torch.compile to Speed training ====================
    if torch.__version__ >= "2.0.0":
        logging.info("PyTorch 2.0+ detected. Compiling model for performance optimization...")
        model = torch.compile(model)
        logging.info("Model compiled successfully.")
    else:
        logging.warning("PyTorch version is less than 2.0.0. Skipping model compilation.")

    # Initialize runner and start training
    runner = BaseRunner(args)
    runner.run(model, train_data, valid_data, test_data, args)

    logging.info('-' * 45 + ' END: ' + Utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    # Initial parsing to get model name
    init_parser = argparse.ArgumentParser(description='Model Runner Initializer', add_help=False)
    init_parser.add_argument('--model_name', type=str, default="DyHGCN",
                             help='The name of the model to run (e.g., DyHGCN).')
    init_args, remaining_argv = init_parser.parse_known_args()

    # Dynamically import the model class
    try:
        model_module = importlib.import_module(f'models.{init_args.model_name}')
        Model = getattr(model_module, init_args.model_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Could not find or import model '{init_args.model_name}'. "
                         f"Please ensure 'models/{init_args.model_name}.py' exists and "
                         f"contains a class named '{init_args.model_name}'. Error: {e}")

    # Set up the main argument parser
    parser = argparse.ArgumentParser(description='Model Runner')
    parser = parse_global_args(parser)
    parser = Model.parse_model_args(parser)  # Add model-specific arguments
    args = parser.parse_args(remaining_argv, namespace=init_args)

    # Construct log and model filenames
    log_filename_parts = [args.model_name, args.data_name]

    # Add extra model-specific parameters to the filename
    if hasattr(Model, 'extra_log_args'):
        extra_params = []
        for arg_name in Model.extra_log_args:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                # Format value for filename
                param_str_value = Utils.format_float_for_filename(value) if isinstance(value, float) else str(value)
                extra_params.append(f"{arg_name}_{param_str_value}")
        if extra_params:
            log_filename_parts.append("__".join(extra_params))

    log_filename_base = "__".join(log_filename_parts)

    # Define log and model paths
    log_dir = os.path.join('log', init_args.model_name)
    model_dir = os.path.join('saved_models', init_args.model_name)
    args.log_file = os.path.join(log_dir, f'{log_filename_base}.txt')
    args.model_path = os.path.join(model_dir, f'{log_filename_base}.pt')

    # Create directories if they don't exist
    Utils.check_dir(args.log_file)
    Utils.check_dir(args.model_path)

    # Configure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level='INFO',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Log file: {args.log_file}")
    logging.info(f"Model path: {args.model_path}")

    main(Model, args)
