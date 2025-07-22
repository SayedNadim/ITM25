# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchnet as tnt
import tqdm
from prettytable import PrettyTable
from torch.nn.functional import interpolate

import utility
from data import get_dataloader


class Tester:
    def __init__(self, args, my_model):
        self.args = args
        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.model = my_model.to(self.device)
        self.model.eval()
        self.test_name = "None"
        self.save_test_img = args.save_test_images
        self.save_npy = args.save_npy
        self.save_desc = args.save_desc

    def test_model(self, attr):
        if self.args.real:
            test_epoch_dir = os.path.join(self.args.test_dir, "real", attr)
        else:
            test_epoch_dir = os.path.join(self.args.test_dir, attr)

        self.model.eval()
        test_loader = get_dataloader(self.args, attr).data_loader

        sum_times = 0
        rmse_list = []
        psnr_list = []
        ssim_list = []
        names = []

        test_rmse = tnt.meter.AverageValueMeter()
        test_psnr = tnt.meter.AverageValueMeter()
        test_ssim = tnt.meter.AverageValueMeter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        print(f"|> {attr}: {len(test_loader)} samples")
        for batch_idx, sample in enumerate(tqdm.tqdm(test_loader)):
            gt_img = sample.get("gt_img", None)
            ldr_img = sample["ldr_img"]
            min_value = sample["min_value"]
            max_value = sample["max_value"]

            gt_img = gt_img.to(self.device) if gt_img is not None else None
            ldr_img = ldr_img.to(self.device)
            min_value = min_value.to(self.device)
            max_value = max_value.to(self.device)
            name = sample["name"]
            if isinstance(name, (list, tuple)):
                names += name
            else:
                names.append(name)

            start.record()

            out = self.model(x=ldr_img)

            end.record()
            torch.cuda.synchronize()
            sum_times += start.elapsed_time(end)

            if attr not in ["aim_val", "aim_test"]:
                rmse = utility.root_mean_sqrt_error(out, gt_img)
                psnr = utility.batch_psnr(
                    utility.hdr_to_pu(out),
                    utility.hdr_to_pu(gt_img),
                    data_range=1.0,
                )
                ssim = utility.calculate_ssim(
                    utility.hdr_to_pu(out),
                    utility.hdr_to_pu(gt_img),
                )
            else:
                rmse = 0.0
                psnr = 0.0
                ssim = 0.0

            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            test_rmse.add(rmse)
            test_psnr.add(psnr)
            test_ssim.add(ssim)

            if self.save_test_img:
                if not os.path.exists(test_epoch_dir):
                    os.makedirs(test_epoch_dir, exist_ok=True)
                if attr not in ["aim_val", "aim_test"]:
                    ldr_tone = ldr_img  # utility.range_compressor_cuda(ldr_img)
                    gt_tone = utility.range_compressor_cuda(gt_img)
                    out_tone = utility.range_compressor_cuda(out)

                    prefix = f"test_{attr}"
                    utility.save_visualization(
                        ldr_tone,
                        out_tone,
                        gt_tone,
                        min_value,
                        max_value,
                        batch_idx,
                        prefix,
                        test_epoch_dir,
                    )
                else:
                    raise ValueError(
                        "Saving test images in aim val/test is disabled. Set --save_test_images to True to enable."
                    )

            if self.save_npy:
                if not os.path.exists(test_epoch_dir):
                    os.makedirs(test_epoch_dir, exist_ok=True)
                out_np = out[0].cpu().numpy()
                out_np = np.clip(out_np, 0., 1.)
                out_np = out_np.transpose(1, 2, 0)  # CHW to HWC
                out_np = (out_np * 1000).astype(np.float32)
                save_name = os.path.basename(name[0]).split(".")[0]
                np.save(f"{test_epoch_dir}/{save_name}.npy", out_np)
        if self.save_desc:
            with open(os.path.join(test_epoch_dir, "readme.txt"), "w") as f:
                f.write(f"Parameters: N/A\n")
                f.write("FLOPs [G]: N/A\n")
                f.write("Extra Data [1] / No Extra Data [0] : 0\n")
                f.close()

        return (
            test_rmse.value()[0],
            test_psnr.value()[0],
            test_ssim.value()[0],
            rmse_list,
            psnr_list,
            ssim_list,
            names,
            sum_times / len(test_loader),
        )

    def test(self, mode="test"):
        if mode == "test":
            print("|> Testing model...")
            test_data_name = []
            test_data_rmse = []
            test_data_psnr = []
            test_data_ssim = []
            test_data_time = []
            with torch.no_grad():

                if self.args.real:
                    pass
                elif "+" in self.args.test_set:
                    test_set = self.args.test_set.split("+")
                else:
                    test_set = [self.args.test_set]
                print(f"|> Testing on {test_set}")
                for test_name in test_set:
                    self.test_name = test_name
                    (
                        avg_rmse,
                        avg_psnr,
                        avg_ssim,
                        rmse_list,
                        psnr_list,
                        ssim_list,
                        names,
                        avg_time,
                    ) = self.test_model(
                        attr=test_name,
                    )

                    test_data_name.append(test_name)
                    test_data_rmse.append(avg_rmse)
                    test_data_psnr.append(avg_psnr)
                    test_data_ssim.append(avg_ssim)
                    test_data_time.append(avg_time)

            table = PrettyTable(test_data_name)
            table.add_row(test_data_rmse)
            table.add_row(test_data_psnr)
            table.add_row(test_data_ssim)
            table.add_row(test_data_time)
            print(table)
        elif mode in ["aim_val", "aim_test"]:
            print(f"|> Testing model on {mode}...")
            test_data_name = []
            test_data_rmse = []
            test_data_psnr = []
            test_data_ssim = []
            test_data_time = []
            with torch.no_grad():
                (
                    avg_rmse,
                    avg_psnr,
                    avg_ssim,
                    rmse_list,
                    psnr_list,
                    ssim_list,
                    names,
                    avg_time,
                ) = self.test_model(
                    attr=mode,
                )

                test_data_name.append(mode)
                test_data_rmse.append(avg_rmse)
                test_data_psnr.append(avg_psnr)
                test_data_ssim.append(avg_ssim)
                test_data_time.append(avg_time)

            table = PrettyTable(test_data_name)
            table.add_row(test_data_rmse)
            table.add_row(test_data_psnr)
            table.add_row(test_data_ssim)
            table.add_row(test_data_time)
            print(table)

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(self.device).contiguous()

        return [_prepare(a) for a in args]
