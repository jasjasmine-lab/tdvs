# AutoDL 部署指南

本指南详细说明如何在 AutoDL 实例中部署 Experience Replay 异常检测项目。

## 🚀 快速开始

### 1. 创建 AutoDL 实例

**推荐配置**：
- **GPU**: RTX 3090 或 RTX 4090（24GB显存）
- **CPU**: 8核心以上
- **内存**: 32GB以上
- **存储**: 100GB以上
- **镜像**: PyTorch 1.11.0 + Python 3.8 + CUDA 11.3

### 2. 环境准备

#### 2.1 连接到实例
```bash
# 通过 AutoDL 提供的 SSH 连接
ssh -p [端口] root@[实例IP]
```

#### 2.2 更新系统包
```bash
apt update && apt upgrade -y
apt install -y git wget curl vim htop
```

### 3. 项目部署

#### 3.1 克隆项目
```bash
# 进入工作目录
cd /root/autodl-tmp

# 克隆项目（替换为你的实际仓库地址）
git clone https://github.com/your-username/One-for-More-1.git
cd One-for-More-1
```

#### 3.2 安装依赖
```bash
# 安装项目依赖
pip install -r requirements.txt

# 如果没有 requirements.txt，手动安装核心依赖
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn
pip install opencv-python pillow
pip install matplotlib seaborn
pip install pytorch-lightning
pip install omegaconf
pip install transformers
pip install diffusers
```

#### 3.3 数据准备
```bash
# 创建数据目录
mkdir -p data/MVTec-AD
mkdir -p data/VisA

# 下载 MVTec-AD 数据集
wget https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C data/MVTec-AD/

# 下载 VisA 数据集（如果需要）
# wget [VisA数据集下载链接]
```

### 4. 配置优化

#### 4.1 GPU 内存优化
```python
# 在训练脚本中添加以下配置
import torch

# 启用混合精度训练
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 设置 GPU 内存增长策略
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)
```

#### 4.2 Experience Replay 参数调优
```python
# 针对 AutoDL 环境的推荐配置
from cdm.gpm_experience_replay import CDAD_ExperienceReplay

# 创建模型实例
model = CDAD_ExperienceReplay(
    # 其他参数...
)

# 配置 Experience Replay 参数
model.configure_experience_replay(
    buffer_size=5000,         # 适中的缓冲区大小
    projection_dim=100,       # 平衡精度和速度
    update_frequency=10,      # 适中的更新频率
    similarity_threshold=0.8  # 标准阈值
)
```

### 5. 运行项目

#### 5.1 训练模型
```bash
# MVTec-AD 数据集训练
python scripts/train_mvtec.py --config models/cdad_mvtec.yaml

# VisA 数据集训练
python scripts/train_visa.py --config models/cdad_visa.yaml
```

#### 5.2 测试模型
```bash
# MVTec-AD 数据集测试
python scripts/test_mvtec.py --config models/cdad_mvtec.yaml

# VisA 数据集测试
python scripts/test_visa.py --config models/cdad_visa.yaml
```

#### 5.3 使用示例
```bash
# 运行 Experience Replay 使用示例
python example_experience_replay_usage.py
```

### 6. 监控和调试

#### 6.1 系统监控
```bash
# 监控 GPU 使用情况
watch -n 1 nvidia-smi

# 监控系统资源
htop

# 监控磁盘使用
df -h
```

#### 6.2 日志管理
```bash
# 创建日志目录
mkdir -p logs

# 运行时保存日志
python scripts/train_mvtec.py 2>&1 | tee logs/training.log
```

### 7. 性能优化建议

#### 7.1 数据加载优化
```python
# 在 DataLoader 中使用多进程
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 根据 CPU 核心数调整
    pin_memory=True,
    persistent_workers=True
)
```

#### 7.2 内存管理
```python
# 定期清理 GPU 内存
import gc
import torch

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

# 在训练循环中定期调用
if batch_idx % 100 == 0:
    cleanup_memory()
```

### 8. 常见问题解决

#### 8.1 CUDA 内存不足
```bash
# 解决方案1：减少批次大小
# 在配置文件中调整 batch_size

# 解决方案2：启用梯度累积
# 在训练脚本中设置 accumulate_grad_batches

# 解决方案3：使用混合精度
pip install apex
```

#### 8.2 数据加载慢
```bash
# 解决方案1：使用 SSD 存储
# 将数据移动到 /root/autodl-tmp（SSD）

# 解决方案2：预处理数据
# 提前将数据转换为适合的格式

# 解决方案3：增加 DataLoader workers
# 在代码中调整 num_workers 参数
```

#### 8.3 网络连接问题
```bash
# 配置代理（如果需要）
export http_proxy=http://proxy.server:port
export https_proxy=http://proxy.server:port

# 使用国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 9. 自动化脚本

#### 9.1 一键部署脚本
```bash
#!/bin/bash
# deploy.sh - 一键部署脚本

echo "开始部署 Experience Replay 项目..."

# 更新系统
apt update && apt upgrade -y

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn opencv-python pillow
pip install matplotlib seaborn pytorch-lightning omegaconf

# 创建必要目录
mkdir -p data logs checkpoints

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "部署完成！"
echo "运行 'python example_experience_replay_usage.py' 开始使用"
```

#### 9.2 训练脚本
```bash
#!/bin/bash
# train.sh - 训练脚本

echo "开始训练模型..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建输出目录
mkdir -p project/logs

# 开始训练
python scripts/train_mvtec.py \
    --config models/cdad_mvtec.yaml \
    --gpus 1 \
    --max_epochs 100 \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！"
```

### 10. 最佳实践

#### 10.1 资源管理
- 定期清理临时文件和日志
- 使用 tmux 或 screen 保持长时间运行的任务
- 设置自动保存检查点

#### 10.2 数据备份
- 定期备份重要的模型检查点
- 使用 AutoDL 的数据盘功能持久化存储
- 考虑使用云存储服务备份结果

#### 10.3 成本优化
- 使用抢占式实例降低成本
- 合理安排训练时间，避免空闲
- 及时释放不需要的实例

### 11. 联系支持

如果在部署过程中遇到问题：
1. 查看项目文档：`README.md`、`EXPERIENCE_REPLAY_REPLACEMENT_GUIDE.md`
2. 运行示例代码：`example_experience_replay_usage.py`
3. 检查日志文件获取详细错误信息
4. 参考 AutoDL 官方文档和社区支持

---

**注意**：请根据你的具体需求调整配置参数，确保在实际部署前先在小规模数据上测试。