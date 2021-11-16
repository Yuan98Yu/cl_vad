from clvad.classifier.model.linear_model import LinearModel


def create_model(args):
    logger = args.logger
    if args.train_what == 'last':  # for linear probe
        args.final_bn = True
        args.final_norm = True
        args.use_dropout = False
    else:  # for training the entire network
        args.final_bn = False
        args.final_norm = False
        args.use_dropout = True

    if args.model == 'lincls':
        model = LinearModel(network=args.network,
                            output_dim=args.output_dim,
                            dropout=args.dropout,
                            use_dropout=args.use_dropout,
                            use_final_bn=args.final_bn,
                            use_l2_norm=args.final_norm)
    else:
        raise NotImplementedError

    if args.device:
        model.to(args.device)
    else:
        model.cuda()

    # optimizer #
    if args.train_what == 'last':
        logger.info('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})

    elif args.train_what == 'ft':
        logger.info('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})

    else:  # train all
        params = []
        logger.info('=> [optimizer] train all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})

    if args.train_what == 'last':
        logger.info('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name, param.requires_grad)
        logger.info('=================================\n')

    return model, params
