import copy
import importlib

import torch
import torch.nn as nn
from utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg
    构建模型辅助类，根据配置动态组装模型，支持冻结层和自动参数传递。
    """

    def __init__(self, cfg):
        """
        初始化函数，根据cfg动态组装模型。
        支持冻结部分层，以及自动参数传递（如inplanes、instrides）。
        参数:
            cfg: list，每个元素为一个子模块的配置字典，包含name、type、kwargs等。
        """
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        """
        动态import并实例化模块。
        参数:
            mtype: str，模块的完整路径（如"torch.nn.Conv2d"）
            kwargs: dict，初始化参数
        返回:
            实例化后的模块对象
        """
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        """
        切换到cuda设备。
        返回:
            self（已切换到cuda）
        """
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        """
        切换到cpu设备。
        返回:
            self（已切换到cpu）
        """
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        """
        多模块流水线式前向传播。
        输入:
            input: dict，包含"image"等键，支持自动转到目标设备
        返回:
            dict，包含所有子模块输出的合并结果
        """
        input = copy.copy(input)
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def freeze_layer(self, module):
        """
        冻结指定模块的参数，不参与训练。
        参数:
            module: nn.Module，需要冻结的模块
        """
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        支持部分层冻结的训练模式切换。
        仅对未冻结的层调用train(mode)，冻结层保持eval状态。
        参数:
            mode: bool，True为训练模式，False为评估模式
        返回:
            self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
