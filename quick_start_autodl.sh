#!/bin/bash
# AutoDL 快速启动脚本
# 一键部署和运行 Experience Replay 项目

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 打印函数
print_header() {
    echo -e "${PURPLE}" 
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Experience Replay                        ║"
    echo "║                   AutoDL 快速启动工具                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_menu() {
    echo -e "${CYAN}"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                        主菜单                               │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│  1. 🚀 一键部署环境                                         │"
    echo "│  2. 📊 运行使用示例                                         │"
    echo "│  3. 🎯 训练模型                                             │"
    echo "│  4. 🧪 测试模型                                             │"
    echo "│  5. 📈 查看系统状态                                         │"
    echo "│  6. 📁 管理文件                                             │"
    echo "│  7. 🔧 高级选项                                             │"
    echo "│  8. 📖 查看帮助                                             │"
    echo "│  9. 🚪 退出                                                 │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

# 检查环境
check_environment() {
    print_info "检查 AutoDL 环境..."
    
    # 检查是否在 AutoDL 环境
    if [ ! -d "/root/autodl-tmp" ]; then
        print_warning "未检测到标准 AutoDL 环境"
        mkdir -p /tmp/autodl-workspace
        cd /tmp/autodl-workspace
    fi
    
    # 检查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU: $GPU_INFO"
    else
        print_error "未检测到 CUDA 环境"
        return 1
    fi
    
    # 检查 Python
    PYTHON_VERSION=$(python --version 2>&1)
    print_success "Python: $PYTHON_VERSION"
    
    return 0
}

# 一键部署
deploy_environment() {
    print_info "开始一键部署..."
    
    if [ -f "scripts/deploy_autodl.sh" ]; then
        bash scripts/deploy_autodl.sh
    else
        print_warning "部署脚本不存在，执行基础部署..."
        
        # 基础部署
        print_info "更新系统包..."
        apt update -qq && apt upgrade -y -qq
        apt install -y -qq git wget curl vim htop
        
        print_info "安装 Python 依赖..."
        pip install torch torchvision torchaudio numpy scipy scikit-learn opencv-python pillow matplotlib seaborn pytorch-lightning omegaconf -q
        
        print_info "创建目录结构..."
        mkdir -p data/{MVTec-AD,VisA} logs checkpoints results
        
        print_info "设置环境变量..."
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$PYTHONPATH:$(pwd)
    fi
    
    print_success "部署完成！"
}

# 运行使用示例
run_example() {
    print_info "运行 Experience Replay 使用示例..."
    
    if [ -f "example_experience_replay_usage.py" ]; then
        python example_experience_replay_usage.py
    else
        print_warning "示例文件不存在，创建简单示例..."
        
        cat > simple_example.py << 'EOF'
import torch
import numpy as np
from datetime import datetime

print("🧠 Experience Replay 简单示例")
print("=" * 40)

# 检查环境
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 创建示例数据
print("\n📊 创建示例数据...")
batch_size = 4
input_size = (3, 224, 224)
test_input = torch.randn(batch_size, *input_size)

if torch.cuda.is_available():
    test_input = test_input.cuda()
    print(f"✅ 数据已移至 GPU")

print(f"📏 输入形状: {test_input.shape}")
print(f"📈 数据范围: [{test_input.min():.3f}, {test_input.max():.3f}]")

# 模拟 Experience Replay 功能
print("\n🔄 模拟 Experience Replay...")

class SimpleExperienceBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
    
    def add_experience(self, data, priority=1.0):
        if len(self.buffer) >= self.max_size:
            # 移除最旧的经验
            self.buffer.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append(data)
        self.priorities.append(priority)
    
    def sample_experiences(self, num_samples=5):
        if len(self.buffer) == 0:
            return []
        
        # 基于优先级采样
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(self.buffer), 
            size=min(num_samples, len(self.buffer)), 
            p=probabilities,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
    
    def get_statistics(self):
        return {
            'buffer_size': len(self.buffer),
            'max_size': self.max_size,
            'avg_priority': np.mean(self.priorities) if self.priorities else 0,
            'total_experiences': len(self.buffer)
        }

# 创建经验缓冲区
buffer = SimpleExperienceBuffer(max_size=100)

# 添加一些经验
for i in range(20):
    fake_experience = torch.randn(3, 64, 64)
    priority = np.random.uniform(0.1, 1.0)
    buffer.add_experience(fake_experience, priority)

print(f"✅ 已添加 20 个经验到缓冲区")

# 采样经验
sampled = buffer.sample_experiences(5)
print(f"📤 采样了 {len(sampled)} 个经验")

# 显示统计信息
stats = buffer.get_statistics()
print(f"\n📊 缓冲区统计:")
for key, value in stats.items():
    print(f"   {key}: {value}")

print(f"\n🎉 示例运行完成！")
print(f"⏰ 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
EOF
        
        python simple_example.py
    fi
}

# 训练模型
train_model() {
    print_info "选择训练选项:"
    echo "1. MVTec-AD 数据集"
    echo "2. VisA 数据集"
    echo "3. 自定义配置"
    
    read -p "请选择 (1-3): " train_choice
    
    case $train_choice in
        1)
            DATASET="mvtec"
            CONFIG="models/cdad_mvtec.yaml"
            ;;
        2)
            DATASET="visa"
            CONFIG="models/cdad_visa.yaml"
            ;;
        3)
            read -p "请输入数据集名称: " DATASET
            read -p "请输入配置文件路径: " CONFIG
            ;;
        *)
            print_error "无效选择"
            return 1
            ;;
    esac
    
    if [ -f "scripts/train_autodl.sh" ]; then
        bash scripts/train_autodl.sh $DATASET $CONFIG
    else
        print_warning "训练脚本不存在，使用基础训练..."
        
        # 设置环境
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        
        # 创建训练目录
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p logs/train_${DATASET}_${TIMESTAMP}
        
        print_info "开始训练 $DATASET 数据集..."
        
        if [ -f "example_experience_replay_usage.py" ]; then
            python example_experience_replay_usage.py 2>&1 | tee logs/train_${DATASET}_${TIMESTAMP}/training.log
        else
            print_error "未找到训练脚本"
        fi
    fi
}

# 测试模型
test_model() {
    print_info "选择测试选项:"
    echo "1. 测试最新模型"
    echo "2. 指定检查点路径"
    
    read -p "请选择 (1-2): " test_choice
    
    case $test_choice in
        1)
            CHECKPOINT_PATH=""
            ;;
        2)
            read -p "请输入检查点路径: " CHECKPOINT_PATH
            ;;
        *)
            print_error "无效选择"
            return 1
            ;;
    esac
    
    read -p "请输入数据集名称 (mvtec/visa): " DATASET
    
    if [ -f "scripts/test_autodl.sh" ]; then
        bash scripts/test_autodl.sh $DATASET $CHECKPOINT_PATH
    else
        print_warning "测试脚本不存在，执行基础测试..."
        
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        
        print_info "运行基础功能测试..."
        python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(2, 3, 224, 224).cuda()
    print(f'GPU 测试通过: {x.shape}')
"
    fi
}

# 查看系统状态
view_system_status() {
    print_info "系统状态信息:"
    echo ""
    
    echo -e "${YELLOW}🖥️  GPU 状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    else
        echo "未检测到 CUDA"
    fi
    
    echo ""
    echo -e "${YELLOW}💾 内存状态:${NC}"
    free -h
    
    echo ""
    echo -e "${YELLOW}💿 磁盘状态:${NC}"
    df -h
    
    echo ""
    echo -e "${YELLOW}🐍 Python 环境:${NC}"
    python --version
    pip list | grep -E "torch|numpy|opencv" | head -5
    
    echo ""
    echo -e "${YELLOW}📁 项目文件:${NC}"
    ls -la | head -10
}

# 文件管理
manage_files() {
    print_info "文件管理选项:"
    echo "1. 查看项目结构"
    echo "2. 清理日志文件"
    echo "3. 备份检查点"
    echo "4. 查看最新结果"
    
    read -p "请选择 (1-4): " file_choice
    
    case $file_choice in
        1)
            print_info "项目结构:"
            tree -L 3 2>/dev/null || find . -type d -maxdepth 3 | head -20
            ;;
        2)
            print_info "清理日志文件..."
            find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
            print_success "清理完成"
            ;;
        3)
            print_info "备份检查点..."
            BACKUP_DIR="backup_$(date +%Y%m%d)"
            mkdir -p $BACKUP_DIR
            cp -r checkpoints/* $BACKUP_DIR/ 2>/dev/null || true
            print_success "备份到: $BACKUP_DIR"
            ;;
        4)
            print_info "最新结果:"
            find results -name "*.json" -o -name "*.md" | sort -r | head -5
            ;;
        *)
            print_error "无效选择"
            ;;
    esac
}

# 高级选项
advanced_options() {
    print_info "高级选项:"
    echo "1. 配置 Experience Replay 参数"
    echo "2. 性能优化设置"
    echo "3. 调试模式"
    echo "4. 导出模型"
    
    read -p "请选择 (1-4): " adv_choice
    
    case $adv_choice in
        1)
            print_info "当前 Experience Replay 配置:"
            echo "buffer_size: ${EXPERIENCE_BUFFER_SIZE:-5000}"
            echo "projection_dim: ${PROJECTION_DIM:-100}"
            echo "update_frequency: ${UPDATE_FREQUENCY:-10}"
            echo "similarity_threshold: ${SIMILARITY_THRESHOLD:-0.8}"
            
            read -p "是否修改配置? (y/n): " modify
            if [[ "$modify" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                read -p "buffer_size [5000]: " new_buffer_size
                read -p "projection_dim [100]: " new_projection_dim
                read -p "update_frequency [10]: " new_update_freq
                read -p "similarity_threshold [0.8]: " new_similarity
                
                export EXPERIENCE_BUFFER_SIZE=${new_buffer_size:-5000}
                export PROJECTION_DIM=${new_projection_dim:-100}
                export UPDATE_FREQUENCY=${new_update_freq:-10}
                export SIMILARITY_THRESHOLD=${new_similarity:-0.8}
                
                print_success "配置已更新"
            fi
            ;;
        2)
            print_info "应用性能优化..."
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            export CUDA_LAUNCH_BLOCKING=0
            print_success "性能优化已应用"
            ;;
        3)
            print_info "启用调试模式..."
            export CUDA_LAUNCH_BLOCKING=1
            export PYTHONPATH=$PYTHONPATH:$(pwd)
            python -c "import torch; print(f'调试模式: CUDA={torch.cuda.is_available()}')"
            ;;
        4)
            print_info "导出模型功能开发中..."
            ;;
        *)
            print_error "无效选择"
            ;;
    esac
}

# 显示帮助
show_help() {
    cat << 'EOF'
📖 Experience Replay AutoDL 使用指南

🚀 快速开始:
1. 运行 "1. 一键部署环境" 安装所有依赖
2. 运行 "2. 运行使用示例" 验证安装
3. 运行 "3. 训练模型" 开始训练

📁 项目结构:
├── cdm/                    # 核心模块
├── scripts/                # 脚本文件
├── data/                   # 数据集
├── logs/                   # 日志文件
├── checkpoints/            # 模型检查点
└── results/                # 结果文件

🔧 常用命令:
- 查看 GPU: nvidia-smi
- 监控资源: htop
- 查看日志: tail -f logs/*/training.log

💡 提示:
- 使用 tmux 运行长时间任务
- 定期备份重要检查点
- 监控磁盘空间使用

📞 获取帮助:
- 查看文档: cat AUTODL_DEPLOYMENT_GUIDE.md
- 运行示例: python example_experience_replay_usage.py
EOF
}

# 主循环
main_loop() {
    while true; do
        clear
        print_header
        
        # 显示当前状态
        if command -v nvidia-smi &> /dev/null; then
            GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
            echo -e "${CYAN}📊 当前状态: GPU 使用率 ${GPU_USAGE}%, 内存 ${GPU_MEMORY}${NC}"
        fi
        
        echo -e "${CYAN}📍 工作目录: $(pwd)${NC}"
        echo ""
        
        print_menu
        
        read -p "请选择操作 (1-9): " choice
        
        case $choice in
            1)
                deploy_environment
                ;;
            2)
                run_example
                ;;
            3)
                train_model
                ;;
            4)
                test_model
                ;;
            5)
                view_system_status
                ;;
            6)
                manage_files
                ;;
            7)
                advanced_options
                ;;
            8)
                show_help
                ;;
            9)
                print_success "感谢使用 Experience Replay AutoDL 工具！"
                exit 0
                ;;
            *)
                print_error "无效选择，请输入 1-9"
                ;;
        esac
        
        echo ""
        read -p "按 Enter 键继续..."
    done
}

# 初始化
init() {
    # 检查环境
    if ! check_environment; then
        print_error "环境检查失败，请在 AutoDL 实例中运行此脚本"
        exit 1
    fi
    
    # 设置基本环境变量
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    
    # 创建基本目录
    mkdir -p logs checkpoints results data
}

# 错误处理
trap 'print_error "脚本执行中断"; exit 1' INT TERM

# 主程序
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

init
main_loop