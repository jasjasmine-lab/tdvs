# Experience Replay - AutoDL 部署指南

🚀 **一键部署 Experience Replay 异常检测项目到 AutoDL 实例**

## 📋 快速开始

### 1. 克隆项目
```bash
# 在 AutoDL 实例中执行
cd /root/autodl-tmp
git clone <your-repo-url>
cd One-for-More-1
```

### 2. 一键启动
```bash
# 给脚本执行权限
chmod +x quick_start_autodl.sh
chmod +x scripts/*.sh

# 运行快速启动工具
./quick_start_autodl.sh
```

### 3. 选择操作
在启动工具中选择：
- **选项 1**: 一键部署环境（首次使用必选）
- **选项 2**: 运行使用示例（验证安装）
- **选项 3**: 开始训练模型

## 🛠️ 手动部署（可选）

如果需要手动控制部署过程：

```bash
# 1. 部署环境
bash scripts/deploy_autodl.sh

# 2. 训练模型
bash scripts/train_autodl.sh mvtec models/cdad_mvtec.yaml

# 3. 测试模型
bash scripts/test_autodl.sh mvtec checkpoints/latest
```

## 📊 推荐配置

### AutoDL 实例配置
- **GPU**: RTX 3090/4090 (24GB显存)
- **CPU**: 8核心以上
- **内存**: 32GB以上
- **存储**: 100GB以上
- **镜像**: PyTorch 1.11.0 + Python 3.8 + CUDA 11.3

### Experience Replay 参数
```yaml
# 标准配置
buffer_size: 5000
projection_dim: 100
update_frequency: 10
similarity_threshold: 0.8

# 小显存优化 (<12GB)
buffer_size: 3000
projection_dim: 64
update_frequency: 20

# 大显存配置 (>24GB)
buffer_size: 8000
projection_dim: 128
update_frequency: 5
```

## 📁 项目结构

```
One-for-More-1/
├── 🚀 quick_start_autodl.sh          # 快速启动工具
├── 📖 AUTODL_DEPLOYMENT_GUIDE.md     # 详细部署指南
├── 📖 README_AUTODL.md               # 本文件
├── scripts/
│   ├── 🔧 deploy_autodl.sh           # 一键部署脚本
│   ├── 🎯 train_autodl.sh            # 训练脚本
│   └── 🧪 test_autodl.sh             # 测试脚本
├── cdm/                              # 核心模块
│   ├── gpm_experience_replay.py      # Experience Replay 实现
│   └── ...
├── data/                             # 数据集目录
├── logs/                             # 日志文件
├── checkpoints/                      # 模型检查点
└── results/                          # 结果文件
```

## 🎯 使用流程

### 第一次使用
1. **部署环境**: 运行快速启动工具选择选项 1
2. **验证安装**: 选择选项 2 运行示例
3. **准备数据**: 下载 MVTec-AD 或 VisA 数据集
4. **开始训练**: 选择选项 3 训练模型

### 日常使用
1. **启动工具**: `./quick_start_autodl.sh`
2. **查看状态**: 选择选项 5 查看系统状态
3. **继续训练**: 选择选项 3 恢复训练
4. **测试模型**: 选择选项 4 测试性能

## 📈 监控和调试

### 系统监控
```bash
# GPU 使用情况
watch -n 1 nvidia-smi

# 系统资源
htop

# 磁盘空间
df -h
```

### 日志查看
```bash
# 训练日志
tail -f logs/*/training.log

# 错误日志
cat logs/*/error.log

# 测试结果
cat results/*/test_report.md
```

### 常见问题

**Q: CUDA 内存不足**
```bash
# 解决方案：减少批次大小
export BATCH_SIZE=16
# 或使用小显存配置
export EXPERIENCE_BUFFER_SIZE=3000
```

**Q: 数据加载慢**
```bash
# 解决方案：将数据移到 SSD
mv data /root/autodl-tmp/
ln -s /root/autodl-tmp/data data
```

**Q: 训练中断**
```bash
# 解决方案：使用 tmux
tmux new -s training
./quick_start_autodl.sh
# Ctrl+B, D 分离会话
# tmux attach -t training 重新连接
```

## 🔧 高级配置

### 环境变量
```bash
# Experience Replay 参数
export EXPERIENCE_BUFFER_SIZE=5000
export PROJECTION_DIM=100
export UPDATE_FREQUENCY=10
export SIMILARITY_THRESHOLD=0.8

# 性能优化
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
```

### 自定义配置
```bash
# 创建自定义配置文件
cp models/cdad_mvtec.yaml models/my_config.yaml
# 编辑配置
vim models/my_config.yaml
# 使用自定义配置训练
bash scripts/train_autodl.sh mvtec models/my_config.yaml
```

## 📊 性能优化建议

### 内存优化
- 使用混合精度训练
- 调整 DataLoader 的 `num_workers`
- 定期清理 GPU 内存

### 速度优化
- 使用 SSD 存储数据
- 启用 CUDNN 基准测试
- 优化批次大小

### 稳定性优化
- 使用 tmux 运行长时间任务
- 定期保存检查点
- 监控系统资源

## 📞 获取帮助

### 文档资源
- 📖 [详细部署指南](AUTODL_DEPLOYMENT_GUIDE.md)
- 📖 [Experience Replay 替换指南](EXPERIENCE_REPLAY_REPLACEMENT_GUIDE.md)
- 📖 [代码修改总结](CODE_MODIFICATION_SUMMARY.md)

### 示例代码
- 🔍 [使用示例](example_experience_replay_usage.py)
- 🧪 [测试脚本](scripts/test_autodl.sh)

### 调试工具
```bash
# 运行诊断
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查项目文件
find . -name "*.py" | head -10

# 验证 Experience Replay
python -c "from cdm.gpm_experience_replay import CDAD_ExperienceReplay; print('✅ 导入成功')"
```

## 🎉 成功部署检查清单

- [ ] ✅ AutoDL 实例创建并连接
- [ ] ✅ 项目代码克隆完成
- [ ] ✅ 环境部署成功（运行选项 1）
- [ ] ✅ 示例运行通过（运行选项 2）
- [ ] ✅ GPU 和 CUDA 正常工作
- [ ] ✅ 数据集下载和配置
- [ ] ✅ 训练流程测试通过
- [ ] ✅ 监控和日志系统正常

---

**🚀 现在你已经准备好在 AutoDL 上使用 Experience Replay 进行异常检测了！**

如有问题，请查看详细的 [部署指南](AUTODL_DEPLOYMENT_GUIDE.md) 或运行 `./quick_start_autodl.sh` 选择帮助选项。