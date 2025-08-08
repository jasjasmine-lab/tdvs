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
from cdm.experience_replay import ExperienceReplayProjector

class CDAD_ExperienceReplay(SD_AMN):
    """
    基于Experience Replay机制的持续异常检测类 (CDAD)
    
    这是原有CDAD类的完全替代实现，使用Experience Replay机制替代增量SVD(iSVD)。
    主要改进包括：
    
    核心特性：
    1. 经验回放缓冲区：存储和管理历史激活状态
    2. 智能采样策略：基于重要性的优先级采样
    3. 增量投影学习：动态更新降维投影矩阵
    4. 新颖性检测：识别异常或新模式的激活
    5. 内存高效：自动管理缓冲区大小和GPU内存
    
    技术优势：
    - 避免了iSVD的数值不稳定性问题
    - 支持更灵活的经验管理策略
    - 提供更好的异常检测性能
    - 兼容原有的接口和工作流程
    
    使用场景：
    - 持续学习中的灾难性遗忘防护
    - 在线异常检测和新颖性发现
    - 神经网络激活模式分析
    """

    def __init__(self, *args, **kwargs):
        """
        初始化Experience Replay版本的CDAD检测器
        
        设置Experience Replay投影器、配置参数和监控统计。
        继承SD_AMN的基础功能，并扩展Experience Replay特性。
        
        Args:
            *args: 传递给父类SD_AMN的位置参数
            **kwargs: 传递给父类SD_AMN的关键字参数
        """
        super().__init__(*args, **kwargs)
        
        # 初始化Experience Replay投影器 - 核心组件
        self.experience_projector = ExperienceReplayProjector(
            buffer_size=5000,           # 每层经验缓冲区最大容量
            projection_dim=100,         # 降维后的目标维度
            update_frequency=10,        # 每10次添加后更新投影矩阵
            similarity_threshold=0.8,   # 新颖性检测阈值（0-1）
            learning_rate=0.01         # 投影矩阵学习率
        )
        
        # 临时存储：当前批次的激活状态
        # 格式：{layer_name: activation_tensor}
        self.current_activations = {}
        
        # 运行时统计信息
        self.batch_count = 0          # 已处理的批次数量
        self.total_experiences = 0    # 累计收集的经验总数
        
        # 向后兼容性：保持原有的project接口
        # 使得现有代码可以无缝切换到Experience Replay
        self.project = {}

    def get_activation(self, name):
        """
        创建激活捕获钩子函数
        
        为指定层创建前向传播钩子，用于捕获该层的输入激活。
        支持多种层类型：线性层、卷积层、多头注意力层等。
        
        Args:
            name (str): 层的名称标识符
            
        Returns:
            function: 钩子函数，将在该层前向传播时自动调用
            
        Note:
            - 钩子函数会自动处理不同层类型的激活提取
            - 激活会被转换为CPU张量并存储在current_activations中
            - 支持批次内激活的累积（通过torch.cat）
        """
        def hook(model, input, output):
            """
            前向传播钩子函数的具体实现
            
            根据层类型提取和处理激活：
            - 线性层/注意力层：提取输入特征并转置
            - 卷积层：使用unfold操作提取卷积窗口特征
            """
            # 处理线性层、非动态量化线性层、多头注意力层
            if (isinstance(model, nn.Linear)
                    or isinstance(model, nn.modules.linear.NonDynamicallyQuantizableLinear)
                    or isinstance(model, MultiheadAttention)):

                # 获取输入通道数（特征维度）
                input_channel = input[0].shape[-1]
                # 重塑并转置：[batch*seq, features] -> [features, batch*seq]
                mat = input[0].reshape(-1, input_channel).t().cpu()

                # 累积存储当前批次的激活（支持多次前向传播）
                if name in self.current_activations.keys():
                    # 在第二维度上拼接新的激活
                    self.current_activations[name] = torch.cat([self.current_activations[name], mat], dim=1)
                else:
                    # 首次存储该层的激活
                    self.current_activations[name] = mat

            # 处理2D卷积层
            elif isinstance(model, nn.Conv2d):
                # 提取卷积层参数
                batch_size, input_channel, input_map_size, _ = input[0].shape
                padding = model.padding[0]      # 填充大小
                kernel_size = model.kernel_size[0]  # 卷积核大小
                stride = model.stride[0]        # 步长

                # 使用unfold展开卷积窗口：[batch, channels, h, w] -> [features, samples]
                # unfold后形状：[batch, kernel*kernel*channels, num_windows]
                # 最终形状：[kernel*kernel*channels, batch*num_windows]
                mat = F.unfold(input[0], kernel_size=kernel_size, stride=stride, padding=padding).transpose(0, 1).reshape(kernel_size*kernel_size*input_channel, -1).detach().cpu()

                # 累积存储当前批次的激活
                if name in self.current_activations.keys():
                    self.current_activations[name] = torch.cat([self.current_activations[name], mat], dim=1)
                else:
                    self.current_activations[name] = mat

        return hook

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        单个测试步骤的处理
        
        执行测试批次的前向传播，并定期记录图像结果。
        与原版CDAD保持完全兼容。
        
        Args:
            batch: 测试数据批次
            batch_idx (int): 批次索引
            
        Note:
            - 每10个批次记录一次图像测试结果
            - 使用@torch.no_grad()确保不计算梯度
        """
        if batch_idx % 10 == 0:
            # 定期记录图像测试结果，用于可视化和监控
            _ = self.log_images_test(batch)

    @torch.no_grad()
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """
        【Experience Replay版本】批次结束时的核心处理逻辑
        
        替代原有的SVD投影方法，使用Experience Replay机制：
        1. 将当前批次的激活添加到经验缓冲区
        2. 获取投影后的激活以保持接口兼容性
        3. 定期打印统计信息用于监控
        
        Args:
            outputs: 模型输出结果
            batch: 当前批次数据
            batch_idx (int): 批次索引
            dataloader_idx: 数据加载器索引
            
        核心改进：
        - 原功能：使用SVD进行特征投影
        - 新功能：基于Experience Replay的智能激活管理
        - 优势：更稳定的数值计算，更好的异常检测性能
        """
        # 核心步骤1：将当前批次激活添加到Experience Replay系统
        for layer_name, activation in self.current_activations.items():
            # 添加激活到对应层的经验缓冲区
            self.experience_projector.add_activation(layer_name, activation)
            # 更新总经验计数（activation.shape[1]为样本数）
            self.total_experiences += activation.shape[1]
        
        # 核心步骤2：获取投影后的激活（保持与原版CDAD的接口兼容性）
        for layer_name, activation in self.current_activations.items():
            # 使用Experience Replay投影器进行激活投影
            projected = self.experience_projector.project_activation(layer_name, activation)
            if projected is not None:
                # 存储投影结果，供后续异常检测使用
                self.project[layer_name] = projected
                # 输出处理进度和维度信息
                print(f"Batch {batch_idx}: Added activation for layer {layer_name}, "
                      f"shape: {activation.shape}, projected shape: {projected.shape}")
        
        # 清理工作：清空当前批次的激活缓存
        self.current_activations.clear()
        # 更新批次计数器
        self.batch_count += 1
            
            # 打印统计信息
        if self.batch_count % 50 == 0:
                stats = self.experience_projector.get_statistics()
                print(f"\n=== Experience Replay Statistics (Batch {batch_idx}) ===")
                print(f"Total layers: {stats['total_layers']}")
                print(f"Total experiences: {stats['total_experiences']}")
                print(f"Update count: {stats['update_count']}")
                for layer_name, details in stats['layer_details'].items():
                    print(f"Layer {layer_name}: {details['total_activations']} activations, "
                          f"{details['unique_patterns']} unique patterns, "
                          f"buffer size: {details.get('buffer_size', 0)}")
                print("=" * 50)

    @torch.no_grad()
    def on_test_end(self):
        """
        【Experience Replay版本】测试结束时的最终处理
        
        完成整个测试阶段后的收尾工作：
        1. 强制更新所有层的Experience Replay投影矩阵
        2. 保存Experience Replay状态以供后续使用
        3. 保存传统格式的投影矩阵（兼容性）
        4. 输出最终统计信息和性能报告
        
        核心改进：
        - 原功能：执行增量SVD并保存投影
        - 新功能：基于Experience Replay的智能投影管理
        - 优势：更稳定的投影学习，更好的持续学习能力
        """
        print("\n=== Final Experience Replay Processing ===")
        
        # 步骤1：强制更新所有层的投影矩阵
        # 确保所有缓冲区中的经验都被用于最终的投影计算
        self.experience_projector._update_projections()
        print("All projection matrices updated with final experiences")
        
        # 步骤2：获取并整理最终的投影矩阵
        # 将Experience Replay投影转换为传统格式以保持兼容性
        final_projections = {}
        for layer_name in self.experience_projector.experience_buffers.keys():
            projection = self.experience_projector.get_projection(layer_name)
            if projection is not None:
                final_projections[layer_name] = projection
                self.project[layer_name] = projection
                print(f"Final projection for {layer_name}: {projection.shape}")
        
        # 步骤3：保存Experience Replay完整状态
        # 包括所有缓冲区、投影矩阵和统计信息
        save_path = f"project/{self.log_name}_experience_replay"
        self.experience_projector.save_state(save_path)
        print(f"Experience Replay state saved to: {save_path}")
        
        # 步骤4：保存传统格式的投影文件（向后兼容）
        # 确保与原版CDAD和其他组件的完全兼容
        torch.save(self.project, f"project/{self.log_name}.pt")
        print(f"Traditional projection format saved to: project/{self.log_name}.pt")
        
        # 步骤5：输出详细的最终统计报告
        final_stats = self.experience_projector.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total batches processed: {self.batch_count}")
        print(f"Total experiences collected: {self.total_experiences}")
        print(f"Total layers: {final_stats['total_layers']}")
        print(f"Total stored experiences: {final_stats['total_experiences']}")
        print(f"Projection updates: {final_stats['update_count']}")
        
        # 步骤6：输出每层的详细性能分析
        for layer_name, details in final_stats['layer_details'].items():
            novelty_ratio = details['unique_patterns'] / max(details['total_activations'], 1)
            print(f"Layer {layer_name}:")
            print(f"  - Total activations: {details['total_activations']}")
            print(f"  - Unique patterns: {details['unique_patterns']}")
            print(f"  - Novelty ratio: {novelty_ratio:.3f}")
            print(f"  - Buffer utilization: {details.get('buffer_size', 0)}/{self.experience_projector.buffer_size}")
            print(f"  - Memory efficiency: {details.get('buffer_size', 0)/max(1, details['total_activations'])*100:.1f}%")
        
        # 步骤7：清理资源和准备下一任务
        # 移除所有注册的前向传播钩子，释放内存
        for value in self.hook_handle.values():
            value.remove()
        print(f"Removed {len(self.hook_handle)} forward hooks")

        # 步骤8：更新任务状态，准备持续学习的下一阶段
        self.task_id += 1  # 递增任务ID，用于多任务持续学习
        self.max_check = 0.0  # 重置检查点，用于异常检测阈值
        
        # 步骤9：输出完成总结
        print("\n=== Experience Replay Processing Complete ===")
        print(f"Successfully completed task {self.task_id - 1}")
        print(f"System ready for task {self.task_id} or anomaly detection")
        print("All projections saved and system state preserved for continual learning")

    @torch.no_grad()
    def on_test_start(self):
        """
        【Experience Replay版本】测试开始时的初始化处理
        
        在开始新的测试/学习阶段前进行必要的准备工作：
        1. 清理之前的状态和缓存
        2. 尝试加载之前保存的Experience Replay状态
        3. 重新注册模型层的前向传播钩子
        4. 初始化统计计数器
        
        核心改进：
        - 原功能：清理状态，注册钩子
        - 新功能：智能状态恢复和Experience Replay初始化
        - 优势：支持持续学习的状态恢复，更好的任务间知识保持
        """
        print(f"\n=== Starting Experience Replay Test (Task {self.task_id}) ===")
        
        # 清理之前的状态
        self.hook_handle = {}
        self.current_activations.clear()
        self.batch_count = 0
        self.total_experiences = 0
        
        # 尝试加载之前的Experience Replay状态
        if self.task_id > 0:
            load_path = f"project/{self.log_name}_experience_replay"
            if self.experience_projector.load_state(load_path):
                print(f"Loaded previous Experience Replay state from: {load_path}")
                # 打印加载的统计信息
                stats = self.experience_projector.get_statistics()
                print(f"Loaded {stats['total_experiences']} experiences across {stats['total_layers']} layers")
            else:
                print("No previous Experience Replay state found, starting fresh")
        
        # 注册钩子
        for name, module in self.model.diffusion_model.named_modules():
            if name in self.unet_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))

        for name, module in self.control_model.named_modules():
            if name in self.control_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))
        
        print(f"Registered hooks for {len(self.hook_handle)} layers")
        print("=" * 50)
    
    def get_experience_statistics(self):
        """
        获取Experience Replay系统的详细统计信息
        
        提供完整的Experience Replay运行状态报告，包括：
        - 各层的经验缓冲区使用情况
        - 投影矩阵更新次数和频率
        - 新颖性检测统计
        - 内存使用效率分析
        
        Returns:
            dict: 包含以下键的统计字典：
                - total_layers: 总层数
                - total_experiences: 总经验数
                - update_count: 投影更新次数
                - layer_details: 每层的详细信息
                    - total_activations: 该层总激活数
                    - unique_patterns: 唯一模式数
                    - buffer_size: 当前缓冲区大小
                    - novelty_ratio: 新颖性比率
        
        使用场景：
        - 监控Experience Replay系统健康状态
        - 分析不同层的学习效果
        - 调试和优化缓冲区配置
        - 生成性能报告
        """
        return self.experience_projector.get_statistics()
    
    def configure_experience_replay(self, 
                                  buffer_size: int = None,
                                  projection_dim: int = None,
                                  update_frequency: int = None,
                                  similarity_threshold: float = None):
        """
        动态配置Experience Replay系统参数
        
        允许在运行时调整Experience Replay的核心参数，以适应不同的
        学习任务和性能需求。支持增量配置，只更新指定的参数。
        
        Args:
            buffer_size (int, optional): 每层经验缓冲区的最大容量
                - 较大值：保存更多历史经验，更好的知识保持
                - 较小值：更快的更新速度，更低的内存使用
                - 推荐范围：1000-10000
                
            projection_dim (int, optional): 降维投影的目标维度
                - 较大值：保留更多特征信息，更精确的表示
                - 较小值：更快的计算速度，更低的存储需求
                - 推荐范围：50-200
                
            update_frequency (int, optional): 投影矩阵更新频率
                - 较小值：更频繁的更新，更快的适应性
                - 较大值：更稳定的投影，更低的计算开销
                - 推荐范围：5-50
                
            similarity_threshold (float, optional): 新颖性检测阈值
                - 较高值（接近1.0）：更严格的新颖性标准
                - 较低值（接近0.0）：更宽松的新颖性标准
                - 推荐范围：0.7-0.9
        
        Note:
            - 参数更改会立即生效，影响后续的经验处理
            - 建议在任务开始前或任务间隙进行配置
            - 可以通过get_experience_statistics()监控配置效果
        """
        if buffer_size is not None:
            self.experience_projector.buffer_size = buffer_size
        if projection_dim is not None:
            self.experience_projector.projection_dim = projection_dim
        if update_frequency is not None:
            self.experience_projector.update_frequency = update_frequency
        if similarity_threshold is not None:
            self.experience_projector.similarity_threshold = similarity_threshold
        
        print(f"Experience Replay configured: buffer_size={self.experience_projector.buffer_size}, "
              f"projection_dim={self.experience_projector.projection_dim}, "
              f"update_frequency={self.experience_projector.update_frequency}, "
              f"similarity_threshold={self.experience_projector.similarity_threshold}")
    
    def replay_experiences(self, layer_name: str, num_experiences: int = 50):
        """
        重放指定层的历史经验，用于分析、调试和可视化
        
        从指定层的经验缓冲区中采样并展示历史激活模式，
        帮助理解模型的学习过程和异常检测行为。
        
        Args:
            layer_name (str): 目标层的名称标识符
                - 必须是已注册钩子的层名称
                - 可通过get_experience_statistics()查看可用层名
                
            num_experiences (int, optional): 要重放的经验数量
                - 默认值：50
                - 实际返回数量可能少于请求数量（取决于缓冲区内容）
                - 推荐范围：10-100
        
        Returns:
            list: 经验记录列表，每个记录包含：
                - state: 激活状态张量
                - reward: 经验重要性评分
                - metadata: 元数据字典
                    - importance: 重要性权重
                    - is_novel: 是否为新颖模式
                    - timestamp: 记录时间戳
                - timestamp: 经验创建时间
        
        使用场景：
        - 分析特定层的激活模式演化
        - 调试异常检测的决策过程
        - 可视化Experience Replay的工作机制
        - 验证新颖性检测的准确性
        - 研究持续学习中的知识保持
        
        Example:
            >>> experiences = model.replay_experiences('layer1', 20)
            >>> for exp in experiences:
            ...     print(f"State: {exp['state'].shape}, Novel: {exp['metadata']['is_novel']}")
        """
        if layer_name in self.experience_projector.experience_buffers:
            buffer = self.experience_projector.experience_buffers[layer_name]
            experiences = buffer.sample_experiences(num_experiences)
            
            print(f"\n=== Replaying {len(experiences)} experiences for layer {layer_name} ===")
            for i, exp in enumerate(experiences):
                print(f"Experience {i+1}:")
                print(f"  - State shape: {exp['state'].shape}")
                print(f"  - Reward: {exp['reward']:.4f}")
                print(f"  - Importance: {exp['metadata'].get('importance', 'N/A')}")
                print(f"  - Is novel: {exp['metadata'].get('is_novel', 'N/A')}")
                print(f"  - Timestamp: {exp['timestamp']}")
            
            return experiences
        else:
            print(f"No experiences found for layer: {layer_name}")
            return []