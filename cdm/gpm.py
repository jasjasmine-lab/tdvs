import re
import torch
import timm

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from utils.util import cal_anomaly_map, log_local, create_logger
from utils.eval_helper import dump, log_metrics, merge_together, performances, save_metrics
from cdm.param import no_trained_para, control_trained_para, contains_any, sub_
from cdm.mha import MultiheadAttention

import os

from cdm.sd_amn import SD_AMN
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cdm.vit import *


class CDAD(SD_AMN):
    """
    The implementation of GPM and iSVD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = {}
        self.act = {}
        # 

    def get_activation(self, name):
        def hook(model, input, output):
            if (isinstance(model, nn.Linear)
                    or isinstance(model, nn.modules.linear.NonDynamicallyQuantizableLinear)
                    or isinstance(model, MultiheadAttention)):

                input_channel = input[0].shape[-1]

                mat = input[0].reshape(-1, input_channel).t().cpu()

                if name in self.act.keys():
                    self.act[name] = torch.cat([self.act[name], mat], dim=1)
                else:
                    self.act[name] = mat

            elif isinstance(model, nn.Conv2d):
                batch_size, input_channel, input_map_size, _ = input[0].shape
                padding = model.padding[0]
                kernel_size = model.kernel_size[0]
                stride = model.stride[0]

                mat = F.unfold(input[0], kernel_size=kernel_size, stride=stride, padding=padding).transpose(0, 1).reshape(kernel_size*kernel_size*input_channel, -1).detach().cpu()

                if name in self.act.keys():
                    self.act[name] = torch.cat([self.act[name], mat], dim=1)
                else:
                    self.act[name] = mat

        return hook

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            _ = self.log_images_test(batch)

    @torch.no_grad()
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            for name, act in self.act.items():
                U, S, Vh = torch.linalg.svd(act.cuda(), full_matrices=False)

                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                r = max(torch.sum(torch.cumsum(sval_ratio, dim=0) < 0.999), 1)
                self.act[name] = U[:, :r].cpu()

    @torch.no_grad()
    def on_test_end(self):
        for name, act in self.act.items():
            if not name in self.project.keys():
                self.project[name] = act.cuda()
            else:
                U1, S1, Vh1 = torch.linalg.svd(act.cuda(), full_matrices=False)

                sval_total = (S1**2).sum()

                act_hat = act.cuda() - self.project[name] @ self.project[name].t() @ act.cuda()

                U, S, Vh = torch.linalg.svd(act_hat)
          
                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / sval_total

                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    # if accumulated_sval < 0.99:
                    if accumulated_sval < 0.5:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break

                self.project[name] = torch.cat([self.project[name], act_hat[:, :r]], dim=1)

        torch.save(self.project, f"project/{self.log_name}.pt")

        for value in self.hook_handle.values():
            value.remove()

        self.task_id += 1
        self.max_check = 0.0

    @torch.no_grad()
    def on_test_start(self):

        self.hook_handle = {}
        del self.act
        self.act = {}
        for name, module in self.model.diffusion_model.named_modules():
            if name in self.unet_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))

        for name, module in self.control_model.named_modules():
            if name in self.control_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))


