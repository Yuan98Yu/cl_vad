import click
import torch
import logging
import random
import numpy as np

from clvad.classifier.trainer.svdd_trainer import DeepSVDDTrainer
from clvad.classifier.model.factory import create_model
from clvad.classifier.data.factory import create_dataset, create_dataloader
from clvad.classifier.utils.options import parse_args


################################################################################
# Settings
################################################################################
def set_logger(args) -> logging.Logger:
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = args.xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % args.data_path)
    logger.info('Export path is %s.' % args.xp_path)

    logger.info('Dataset: %s' % args.dataset_name)
    logger.info('Normal class: %d' % args.normal_class)
    logger.info('Network: %s' % args.net_name)

    # If specified, load experiment config from JSON-file
    if args.load_config:
        args.load_config(import_json=args.load_config)
        logger.info('Loaded configuration from %s.' % args.load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % args.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % args.settings['nu'])


def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    args = parse_args()
    logger = set_logger(args)

    # Set seed
    if args.settings['seed'] != -1:
        random.seed(args.settings['seed'])
        np.random.seed(args.settings['seed'])
        torch.manual_seed(args.settings['seed'])
        logger.info('Set seed to %d.' % args.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Load data
    dataset = create_dataset(args)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD_trainer = DeepSVDDTrainer(args.settings['objective'], args.settings['nu'])
    model = create_model
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if args.pretrain:
        pass

    # Train model on dataset
    model = deep_SVDD_trainer.train(dataset, model)

    args.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
