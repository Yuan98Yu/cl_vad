import logging


################################################################################
# Settings
################################################################################
def set_logger(args) -> logging.Logger:
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = args.log_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % args.objective)
    logger.info('Nu-paramerter: %.2f' % args.nu)

    return logger
