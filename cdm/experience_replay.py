import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import pickle
import os

class ExperienceBuffer:
    """
    经验回放缓冲区类 - 用于存储和采样历史激活经验
    
    该类实现了一个固定大小的循环缓冲区，用于存储神经网络层的激活状态作为"经验"。
    支持基于优先级的采样策略，优先采样重要性更高的经验。
    
    主要功能：
    1. 存储激活状态及其重要性分数
    2. 支持优先级采样和均匀随机采样
    3. 提供经验的持久化存储和加载
    4. 管理缓冲区的容量和清理
    
    Args:
        max_size: 缓冲区最大容量，超出时会覆盖最旧的经验
        priority_sampling: 是否启用基于优先级的采样策略
    """
    def __init__(self, max_size: int = 10000, priority_sampling: bool = True):
        """初始化经验缓冲区"""
        self.max_size = max_size  # 缓冲区最大容量
        self.priority_sampling = priority_sampling  # 是否启用优先级采样
        self.buffer = deque(maxlen=max_size)  # 主缓冲区，使用双端队列实现循环缓冲
        self.priorities = deque(maxlen=max_size) if priority_sampling else None  # 优先级队列
        
    def add_experience(self, state: torch.Tensor, reward: float = 1.0, metadata: dict = None):
        """
        添加新的经验到缓冲区
        
        将神经网络层的激活状态作为经验存储到缓冲区中。每个经验包含：
        - 激活状态张量
        - 重要性/奖励分数
        - 时间戳和其他元数据
        
        Args:
            state: 激活状态张量，通常是神经网络层的输出特征
            reward: 经验的重要性/奖励值，用于优先级采样
            metadata: 额外的元数据信息，如层名称、新颖性标记等
        """
        # 创建经验字典，包含所有必要信息
        experience = {
            'state': state.cpu(),  # 将张量移到CPU以节省GPU内存
            'reward': reward,  # 重要性/奖励分数
            'metadata': metadata or {},  # 元数据信息（层名称、新颖性等）
            'timestamp': len(self.buffer)  # 添加时的时间戳，用于追踪经验的新旧程度
        }
        
        # 添加到主缓冲区（双端队列会自动处理容量限制）
        self.buffer.append(experience)
        
        # 如果启用优先级采样，同时更新优先级队列
        if self.priority_sampling:
            # 基于奖励值设置优先级，添加小常数避免零优先级
            priority = abs(reward) + 1e-6  # 避免零优先级导致的采样问题
            self.priorities.append(priority)
    
    def sample_experiences(self, batch_size: int) -> List[dict]:
        """
        从缓冲区采样经验
        
        根据配置的采样策略从缓冲区中选择经验用于学习或分析。
        支持两种采样模式：
        1. 优先级采样：基于重要性分数进行加权采样，重要经验被选中的概率更高
        2. 均匀采样：所有经验被选中的概率相等
        
        Args:
            batch_size: 需要采样的经验数量，会自动限制在缓冲区大小范围内
        
        Returns:
            采样得到的经验列表，每个经验包含state、reward、metadata、timestamp等信息
        """
        # 检查缓冲区是否为空
        if len(self.buffer) == 0:
            return []
            
        # 限制采样数量不超过缓冲区大小
        batch_size = min(batch_size, len(self.buffer))
        
        if self.priority_sampling and self.priorities:
            # 基于优先级的加权采样
            priorities = np.array(list(self.priorities))
            # 计算采样概率（优先级越高，被选中概率越大）
            probabilities = priorities / priorities.sum()
            # 无重复采样，确保不会选择重复的经验
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            # 均匀随机采样，所有经验被选中的概率相等
            return random.sample(list(self.buffer), batch_size)
    
    def get_recent_experiences(self, n: int) -> List[dict]:
        """
        获取最近添加的经验
        
        返回缓冲区中最新添加的若干个经验，用于分析最近的激活模式
        或检查新经验与历史经验的相似性。
        
        Args:
            n: 需要获取的最近经验数量
        Returns:
            按时间倒序排列的最近经验列表（最新的在前）
        """
        # 限制获取数量不超过缓冲区实际大小
        n = min(n, len(self.buffer))
        # 返回最后n个经验（最新的经验在队列末尾）
        return list(self.buffer)[-n:] if n > 0 else []
    
    def clear(self):
        """
        清空经验缓冲区
        
        移除所有存储的经验和优先级信息，将缓冲区重置为初始状态。
        通常在训练的不同阶段或需要重新开始收集经验时使用。
        """
        self.buffer.clear()  # 清空主缓冲区
        if self.priorities:
            self.priorities.clear()  # 清空优先级队列
    
    def save(self, filepath: str):
        """
        将缓冲区状态保存到文件
        
        将当前缓冲区中的所有经验、优先级信息和配置参数序列化保存到指定文件。
        支持训练过程的断点续传和经验的持久化存储。
        
        Args:
            filepath: 保存文件的路径，会自动创建必要的目录结构
        """
        # 确保目标目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 使用pickle序列化保存所有状态信息
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),  # 转换为列表以便序列化
                'priorities': list(self.priorities) if self.priorities else None,
                'max_size': self.max_size,  # 保存配置参数
                'priority_sampling': self.priority_sampling
            }, f)
    
    def load(self, filepath: str):
        """
        从文件加载缓冲区状态
        
        从指定文件中恢复之前保存的缓冲区状态，包括所有经验数据、
        优先级信息和配置参数。用于实现训练的断点续传。
        
        Args:
            filepath: 保存文件的路径
        Returns:
            bool: 加载成功返回True，文件不存在或加载失败返回False
        """
        # 检查文件是否存在
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    # 反序列化数据
                    data = pickle.load(f)
                    
                    # 恢复主缓冲区，保持原有的容量限制
                    self.buffer = deque(data['buffer'], maxlen=self.max_size)
                    
                    # 恢复优先级队列（如果存在）
                    if data['priorities']:
                        self.priorities = deque(data['priorities'], maxlen=self.max_size)
                    
                    return True  # 加载成功
            except Exception as e:
                print(f"Failed to load experience buffer: {e}")
                return False
        return False  # 文件不存在

class ExperienceReplayProjector:
    """
    Experience Replay投影器类 - 管理多层神经网络的经验回放和降维投影
    
    该类是Experience Replay机制的核心组件，负责：
    1. 管理多个神经网络层的经验缓冲区
    2. 基于历史经验计算和更新降维投影矩阵
    3. 检测新激活的新颖性和重要性
    4. 提供高效的激活投影和相似性计算
    
    核心算法：
    - 使用SVD分解历史激活来构建投影矩阵
    - 基于余弦相似性检测激活的新颖性
    - 采用增量更新策略平衡性能和准确性
    
    Args:
        buffer_size: 每个层的经验缓冲区大小，控制存储的历史激活数量
        projection_dim: 投影后的维度大小，用于降维压缩
        update_frequency: 投影矩阵更新频率（每N次添加后更新）
        similarity_threshold: 新颖性检测的相似性阈值，超过此值认为激活相似
        learning_rate: 学习率参数，用于投影矩阵的增量更新
    """
    def __init__(self, 
                 buffer_size: int = 5000,
                 projection_dim: int = 100,
                 update_frequency: int = 10,
                 similarity_threshold: float = 0.8,
                 learning_rate: float = 0.01):
        
        # 配置参数
        self.buffer_size = buffer_size  # 每层缓冲区的最大容量
        self.projection_dim = projection_dim  # 投影后的目标维度
        self.update_frequency = update_frequency  # 投影矩阵更新频率
        self.similarity_threshold = similarity_threshold  # 新颖性检测阈值
        self.learning_rate = learning_rate  # 投影矩阵学习率
        
        # 核心数据结构
        self.experience_buffers: Dict[str, ExperienceBuffer] = {}  # 每层的经验缓冲区
        self.projections: Dict[str, torch.Tensor] = {}  # 每层的投影矩阵
        
        # 统计和监控信息
        self.update_count = 0  # 总更新次数计数器
        self.layer_stats = {}  # 每层的详细统计信息
        
    def _compute_activation_importance(self, activation: torch.Tensor) -> float:
        """
        计算激活状态的重要性分数
        
        基于多个统计指标评估激活的重要性，包括：
        - 方差：衡量激活的变化程度
        - 稀疏性：衡量激活的稀疏程度（非零元素比例）
        - 范数：衡量激活的整体强度
        
        Args:
            activation: 输入的激活张量
        Returns:
            float: 重要性分数，值越大表示越重要
        """
        # 基于激活的方差和稀疏性计算重要性
        variance = torch.var(activation).item()
        sparsity = (activation == 0).float().mean().item()
        norm = torch.norm(activation).item()
        
        # 组合多个指标
        importance = variance * (1 - sparsity) * norm
        return importance
    
    def _compute_similarity(self, act1: torch.Tensor, act2: torch.Tensor) -> float:
        """
        计算两个激活状态之间的相似性
        
        使用余弦相似度度量两个激活向量的相似程度。
        自动处理不同形状的激活张量，通过截断到较小维度来对齐。
        
        Args:
            act1: 第一个激活张量
            act2: 第二个激活张量
        Returns:
            float: 相似性分数，范围[-1, 1]，1表示完全相似，-1表示完全相反
        """
        # 使用余弦相似度
        act1_flat = act1.flatten()
        act2_flat = act2.flatten()
        
        if act1_flat.shape != act2_flat.shape:
            # 如果形状不同，使用较小的维度
            min_dim = min(len(act1_flat), len(act2_flat))
            act1_flat = act1_flat[:min_dim]
            act2_flat = act2_flat[:min_dim]
        
        similarity = torch.cosine_similarity(act1_flat.unsqueeze(0), act2_flat.unsqueeze(0))
        return similarity.item()
    
    def _update_projection_matrix(self, layer_name: str, experiences: List[dict]):
        """
        基于历史经验更新指定层的投影矩阵
        
        使用奇异值分解(SVD)从历史激活经验中学习最优的降维投影。
        算法步骤：
        1. 收集经验中的激活状态和重要性权重
        2. 对不同形状的激活进行填充对齐
        3. 计算加权协方差矩阵
        4. 通过SVD分解得到主成分投影矩阵
        5. 选择前projection_dim个主成分作为投影基
        
        Args:
            layer_name: 目标层的名称标识
            experiences: 该层的历史经验列表，包含激活状态和奖励信息
        """
        if not experiences:
            return
            
        # 收集所有激活状态
        states = [exp['state'] for exp in experiences]
        rewards = [exp['reward'] for exp in experiences]
        
        # 将激活状态堆叠
        if len(states) > 0:
            # 确保所有状态具有相同的形状
            max_features = max(state.shape[0] for state in states)
            
            # 填充较小的状态到相同维度
            padded_states = []
            for state in states:
                if state.shape[0] < max_features:
                    padding = torch.zeros(max_features - state.shape[0], state.shape[1])
                    padded_state = torch.cat([state, padding], dim=0)
                else:
                    padded_state = state[:max_features]
                padded_states.append(padded_state)
            
            # 堆叠所有状态
            stacked_states = torch.stack(padded_states, dim=0)  # [batch, features, samples]
            
            # 加权平均（基于奖励）
            weights = torch.tensor(rewards).float().unsqueeze(-1).unsqueeze(-1)
            weights = weights / weights.sum()
            
            # 计算加权的协方差矩阵
            weighted_states = (stacked_states * weights).sum(dim=0)  # [features, samples]
            
            # 使用PCA进行降维
            try:
                U, S, V = torch.svd(weighted_states)
                
                # 选择前projection_dim个主成分
                n_components = min(self.projection_dim, U.shape[1])
                projection = U[:, :n_components]
                
                self.projections[layer_name] = projection
                
            except Exception as e:
                print(f"SVD failed for layer {layer_name}: {e}")
                # 使用随机投影作为备选
                input_dim = weighted_states.shape[0]
                projection = torch.randn(input_dim, min(self.projection_dim, input_dim))
                projection = torch.qr(projection)[0]  # 正交化
                self.projections[layer_name] = projection
    
    def add_activation(self, layer_name: str, activation: torch.Tensor):
        """
        添加新的激活状态到指定层的经验缓冲区
        
        这是Experience Replay系统的主要入口点。对每个新激活执行：
        1. 计算激活的重要性分数
        2. 与历史经验比较检测新颖性
        3. 基于重要性和新颖性计算奖励值
        4. 将经验存储到缓冲区中
        5. 定期触发投影矩阵更新
        
        Args:
            layer_name: 神经网络层的名称标识
            activation: 该层的激活状态张量
        """
        # 初始化该层的缓冲区
        if layer_name not in self.experience_buffers:
            self.experience_buffers[layer_name] = ExperienceBuffer(
                max_size=self.buffer_size, 
                priority_sampling=True
            )
            self.layer_stats[layer_name] = {'total_activations': 0, 'unique_patterns': 0}
        
        # 计算激活的重要性
        importance = self._compute_activation_importance(activation)
        
        # 检查是否与现有经验相似
        buffer = self.experience_buffers[layer_name]
        is_novel = True
        
        if len(buffer.buffer) > 0:
            # 与最近的几个经验比较相似性
            recent_experiences = buffer.get_recent_experiences(min(10, len(buffer.buffer)))
            for exp in recent_experiences:
                similarity = self._compute_similarity(activation, exp['state'])
                if similarity > self.similarity_threshold:
                    is_novel = False
                    break
        
        # 如果是新颖的激活，给予更高的奖励
        reward = importance * (2.0 if is_novel else 1.0)
        
        # 添加到缓冲区
        metadata = {
            'importance': importance,
            'is_novel': is_novel,
            'layer_name': layer_name
        }
        
        buffer.add_experience(activation, reward, metadata)
        
        # 更新统计信息
        self.layer_stats[layer_name]['total_activations'] += 1
        if is_novel:
            self.layer_stats[layer_name]['unique_patterns'] += 1
        
        # 定期更新投影矩阵
        self.update_count += 1
        if self.update_count % self.update_frequency == 0:
            self._update_projections()
    
    def _update_projections(self):
        """
        批量更新所有层的投影矩阵
        
        遍历所有已注册的层，从各自的经验缓冲区中采样经验，
        并更新对应的投影矩阵。这是一个计算密集的操作，
        因此按照update_frequency的频率定期执行。
        """
        for layer_name, buffer in self.experience_buffers.items():
            # 采样经验进行投影更新
            sample_size = min(100, len(buffer.buffer))
            experiences = buffer.sample_experiences(sample_size)
            
            if experiences:
                self._update_projection_matrix(layer_name, experiences)
    
    def get_projection(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        获取指定层的当前投影矩阵
        
        Args:
            layer_name: 目标层的名称
        Returns:
            投影矩阵张量，如果该层尚未初始化则返回None
        """
        return self.projections.get(layer_name)
    
    def project_activation(self, layer_name: str, activation: torch.Tensor) -> torch.Tensor:
        """
        使用学习到的投影矩阵对激活进行降维投影
        
        将高维激活状态投影到低维空间，保留最重要的特征信息。
        自动处理维度不匹配的情况，通过填充或截断来对齐维度。
        
        Args:
            layer_name: 目标层的名称
            activation: 需要投影的激活张量
        Returns:
            投影后的低维激活张量，如果投影矩阵不存在则返回原激活
        """
        projection = self.get_projection(layer_name)
        if projection is not None:
            # 确保维度匹配
            if activation.shape[0] != projection.shape[0]:
                # 调整激活维度
                if activation.shape[0] < projection.shape[0]:
                    padding = torch.zeros(projection.shape[0] - activation.shape[0], activation.shape[1])
                    activation = torch.cat([activation, padding], dim=0)
                else:
                    activation = activation[:projection.shape[0]]
            
            return projection.t() @ activation
        else:
            return activation
    
    def save_state(self, filepath: str):
        """
        保存Experience Replay投影器的完整状态
        
        将所有重要的状态信息持久化保存，包括：
        - 所有层的投影矩阵
        - 各层的经验缓冲区数据
        - 统计信息和配置参数
        
        支持训练过程的断点续传和模型部署。
        
        Args:
            filepath: 保存文件的基础路径，会生成多个相关文件
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存投影矩阵
        torch.save(self.projections, f"{filepath}_projections.pt")
        
        # 保存经验缓冲区
        for layer_name, buffer in self.experience_buffers.items():
            buffer.save(f"{filepath}_buffer_{layer_name}.pkl")
        
        # 保存统计信息
        with open(f"{filepath}_stats.pkl", 'wb') as f:
            pickle.dump({
                'layer_stats': self.layer_stats,
                'update_count': self.update_count,
                'config': {
                    'buffer_size': self.buffer_size,
                    'projection_dim': self.projection_dim,
                    'update_frequency': self.update_frequency,
                    'similarity_threshold': self.similarity_threshold,
                    'learning_rate': self.learning_rate
                }
            }, f)
    
    def load_state(self, filepath: str):
        """
        从文件加载Experience Replay投影器的状态
        
        恢复之前保存的完整状态，包括投影矩阵、经验缓冲区
        和统计信息。用于实现训练的断点续传。
        
        Args:
            filepath: 保存文件的基础路径
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            # 加载投影矩阵
            if os.path.exists(f"{filepath}_projections.pt"):
                self.projections = torch.load(f"{filepath}_projections.pt")
            
            # 加载统计信息
            if os.path.exists(f"{filepath}_stats.pkl"):
                with open(f"{filepath}_stats.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.layer_stats = data['layer_stats']
                    self.update_count = data['update_count']
            
            # 加载经验缓冲区
            for layer_name in self.layer_stats.keys():
                buffer_path = f"{filepath}_buffer_{layer_name}.pkl"
                if os.path.exists(buffer_path):
                    if layer_name not in self.experience_buffers:
                        self.experience_buffers[layer_name] = ExperienceBuffer(
                            max_size=self.buffer_size, 
                            priority_sampling=True
                        )
                    self.experience_buffers[layer_name].load(buffer_path)
            
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """
        获取Experience Replay系统的详细统计信息
        
        返回包含以下信息的统计字典：
        - 总层数和总经验数量
        - 更新次数和各层详细统计
        - 缓冲区使用情况和新颖性检测结果
        
        用于监控系统运行状态和性能分析。
        
        Returns:
            dict: 包含各种统计指标的字典
        """
        stats = {
            'total_layers': len(self.experience_buffers),
            'total_experiences': sum(len(buffer.buffer) for buffer in self.experience_buffers.values()),
            'update_count': self.update_count,
            'layer_details': self.layer_stats.copy()
        }
        
        # 添加每层的缓冲区大小
        for layer_name, buffer in self.experience_buffers.items():
            stats['layer_details'][layer_name]['buffer_size'] = len(buffer.buffer)
            
        return stats