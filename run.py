import argparse
import torch
from pytorch_lightning import loggers as pl_loggers
from configs.default import get_cfg_defaults
import random
import numpy as np
from trainer import BaseTrainer
from pprint import pprint

def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_cfg_defaults()
    config.defrost()
    config.merge_from_file(exp_config)
    if opts:
        config.merge_from_list(opts)
    config.freeze()
    print(f"CONFIG: ")
    pprint(config)
    print("="*20)

    print(f"using seed {config.SEED}")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    trainer = BaseTrainer(config)
    if run_type == "train":
        trainer.train()
    # elif run_type == "eval":
    #     trainer.eval()
    # elif run_type == "inference":
    #     trainer.inference()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()