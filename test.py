# -*- coding: utf-8 -*-
"""
P2G2Net: Guided Depth Super-Resolution testing entrypoint.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch

import utility
from model import net
from Tester import Tester


def parse_args():
    parser = argparse.ArgumentParser(description="DPPRNET")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--file_name", type=str, default="", help="Suffix for checkpoint and logs."
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="test",
        help="Mode for testing. [test, aim_val, aim_test]",
    )
    return parser.parse_args()


def main():
    args_mod = parse_args()
    args = utility.read_yaml(args_mod.config)

    device = torch.device("cpu" if args.cpu else "cuda")
    base_model = net(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base=args.base,
        levels=args.num_pyramid,
    ).to(device)

    # candidate scales and schedulers
    schedulers = ["CosineAnnealing"]

    for strat in schedulers:
        print("=" * 50)
        print(f"|> Testing scheduler={strat}")
        print("=" * 50)

        ckpt_name = f"./checkpoints/{args.model_name}_{strat}/best.pth"
        if not os.path.exists(ckpt_name):
            print(f"|> Checkpoint not found: {ckpt_name}\n|> Skipping test...")
            continue

        print(f"|> Loading checkpoint: {ckpt_name}")
        ckpt = torch.load(ckpt_name, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", {})

        new_state = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        base_model.load_state_dict(new_state)

        epoch_num = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_avg_psnr", -1e10)
        print(f"|> Resumed from epoch {epoch_num}, best PSNR={best_psnr:.4f}\n")

        tester = Tester(args=args, my_model=base_model)
        tester.test(mode=args_mod.test_mode)
        print("|> Test completed.\n")


if __name__ == "__main__":
    main()
