from clvad.trainer.base import Trainer
from clvad.trainer.infonce import InfonceTrainer


def create_trainer(args, model) -> Trainer:
    if args.strategy == 'infonce':
        trainer = InfonceTrainer(args, model)
    elif args.strategy == '':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return trainer
