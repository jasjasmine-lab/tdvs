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

# 【SVD替换所需的额外导入】
# 如果要使用sklearn的分解方法，需要取消注释以下导入：
# from sklearn.decomposition import TruncatedSVD, IncrementalPCA
# from sklearn.utils.extmath import randomized_svd
# import numpy as np


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
        """
        【需要修改的函数1】批次结束时的SVD处理
        原功能：使用torch.linalg.svd进行奇异值分解，保留99.9%的能量
        修改点：可以替换为其他SVD方法，如：
        1. 使用sklearn的TruncatedSVD: from sklearn.decomposition import TruncatedSVD
        2. 使用randomized SVD: torch.svd_lowrank()
        3. 使用incremental PCA: from sklearn.decomposition import IncrementalPCA
        """
        if batch_idx % 10 == 0:
            for name, act in self.act.items():
                # 【原始iSVD实现】- 可替换为其他SVD方法
                U, S, Vh = torch.linalg.svd(act.cuda(), full_matrices=False)

                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                r = max(torch.sum(torch.cumsum(sval_ratio, dim=0) < 0.999), 1)
                self.act[name] = U[:, :r].cpu()
                
                # 【替换示例1 - 使用randomized SVD】
                # U, S, Vh = torch.svd_lowrank(act.cuda(), q=min(act.shape)-1)
                # sval_total = (S**2).sum()
                # sval_ratio = (S**2) / sval_total
                # r = max(torch.sum(torch.cumsum(sval_ratio, dim=0) < 0.999), 1)
                # self.act[name] = U[:, :r].cpu()
                
                # 【替换示例2 - 使用TruncatedSVD】
                # from sklearn.decomposition import TruncatedSVD
                # n_components = min(min(act.shape)-1, 100)  # 限制组件数量
                # svd = TruncatedSVD(n_components=n_components)
                # U_truncated = svd.fit_transform(act.cuda().cpu().numpy())
                # self.act[name] = torch.tensor(U_truncated).float()

    @torch.no_grad()
    def on_test_end(self):
        """
        【需要修改的函数2】测试结束时的增量SVD处理
        原功能：实现增量奇异值分解(iSVD)，将新的激活与已有投影进行正交化后再次SVD
        修改点：这是iSVD的核心实现，可以替换为：
        1. 标准SVD + 重新计算所有投影
        2. 使用sklearn的IncrementalPCA进行增量学习
        3. 使用在线SVD算法
        4. 使用随机化SVD方法
        """
        for name, act in self.act.items():
            if not name in self.project.keys():
                # 第一次处理该层，直接保存
                self.project[name] = act.cuda()
            else:
                # 【原始iSVD实现】- 增量SVD的核心逻辑
                # 步骤1：对当前激活进行SVD分解
                U1, S1, Vh1 = torch.linalg.svd(act.cuda(), full_matrices=False)

                sval_total = (S1**2).sum()

                # 步骤2：计算正交化后的激活（去除已有投影的影响）
                act_hat = act.cuda() - self.project[name] @ self.project[name].t() @ act.cuda()

                # 步骤3：对正交化后的激活进行SVD
                U, S, Vh = torch.linalg.svd(act_hat)
          
                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / sval_total

                accumulated_sval = (sval_total - sval_hat) / sval_total

                # 步骤4：根据能量阈值选择保留的维度
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    # if accumulated_sval < 0.99:
                    if accumulated_sval < 0.5:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break

                # 步骤5：将新的主成分添加到现有投影中
                self.project[name] = torch.cat([self.project[name], act_hat[:, :r]], dim=1)
                
                # 【替换示例1 - 使用IncrementalPCA】
                # from sklearn.decomposition import IncrementalPCA
                # if not hasattr(self, 'ipca_models'):
                #     self.ipca_models = {}
                # if name not in self.ipca_models:
                #     self.ipca_models[name] = IncrementalPCA(n_components=min(act.shape[0], 100))
                #     self.project[name] = torch.tensor(self.ipca_models[name].fit_transform(act.cuda().cpu().numpy().T)).cuda().T
                # else:
                #     self.ipca_models[name].partial_fit(act.cuda().cpu().numpy().T)
                #     self.project[name] = torch.tensor(self.ipca_models[name].components_).cuda()
                
                # 【替换示例2 - 简单的标准SVD重新计算】
                # # 将新激活与已有投影合并后重新计算SVD
                # combined_act = torch.cat([self.project[name], act.cuda()], dim=1)
                # U_new, S_new, Vh_new = torch.linalg.svd(combined_act, full_matrices=False)
                # # 根据能量阈值选择维度
                # sval_total_new = (S_new**2).sum()
                # sval_ratio_new = (S_new**2) / sval_total_new
                # r_new = max(torch.sum(torch.cumsum(sval_ratio_new, dim=0) < 0.99), 1)
                # self.project[name] = U_new[:, :r_new]

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


