from clvad.feature_learning.model.pretrain import InfoNCE, UberNCE


def create_model(args):
    print("=> creating {} model with '{}' backbone".format(args.model, args.net))
    if args.model == 'infonce':
        model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    elif args.model == 'ubernce':
        model = UberNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    
    return model