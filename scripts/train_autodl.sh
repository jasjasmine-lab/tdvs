#!/bin/bash
# AutoDL 训练脚本
# 使用方法: bash scripts/train_autodl.sh [dataset] [config]
# 示例: bash scripts/train_autodl.sh mvtec models/cdad_mvtec.yaml

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 默认参数
DATASET=${1:-"mvtec"}
CONFIG=${2:-"models/cdad_mvtec.yaml"}
GPUS=${3:-1}
MAX_EPOCHS=${4:-100}
BATCH_SIZE=${5:-32}

echo "🚀 开始在 AutoDL 上训练 Experience Replay 模型"
echo "================================================"
print_info "数据集: $DATASET"
print_info "配置文件: $CONFIG"
print_info "GPU 数量: $GPUS"
print_info "最大轮数: $MAX_EPOCHS"
print_info "批次大小: $BATCH_SIZE"
echo "================================================"

# 检查环境
check_environment() {
    print_info "检查训练环境..."
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未找到 CUDA 环境"
        exit 1
    fi
    
    # 检查 GPU 内存
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    print_info "GPU 内存: ${GPU_MEMORY}MB"
    
    if [ "$GPU_MEMORY" -lt 8000 ]; then
        print_warning "GPU 内存较小，建议减少批次大小"
        BATCH_SIZE=16
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "${DISK_SPACE%.*}" -lt 10 ]; then
        print_warning "磁盘空间不足 10GB，请清理空间"
    fi
    
    print_success "环境检查完成"
}

# 设置环境变量
setup_training_env() {
    print_info "设置训练环境变量..."
    
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export TORCH_CUDNN_V8_API_ENABLED=1
    export CUDA_LAUNCH_BLOCKING=0  # 异步执行以提高性能
    
    # PyTorch 优化设置
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    print_success "环境变量设置完成"
}

# 创建训练目录
setup_training_dirs() {
    print_info "创建训练目录..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_NAME="${DATASET}_${TIMESTAMP}"
    
    mkdir -p logs/$EXPERIMENT_NAME
    mkdir -p checkpoints/$EXPERIMENT_NAME
    mkdir -p results/$EXPERIMENT_NAME
    
    export EXPERIMENT_DIR="logs/$EXPERIMENT_NAME"
    export CHECKPOINT_DIR="checkpoints/$EXPERIMENT_NAME"
    export RESULT_DIR="results/$EXPERIMENT_NAME"
    
    print_success "训练目录创建完成: $EXPERIMENT_NAME"
}

# 检查数据集
check_dataset() {
    print_info "检查数据集: $DATASET"
    
    case $DATASET in
        "mvtec")
            DATA_DIR="data/MVTec-AD"
            if [ ! -d "$DATA_DIR" ]; then
                print_error "MVTec-AD 数据集未找到，请先下载数据集"
                print_info "下载命令: wget https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/mvtec_anomaly_detection.tar.xz"
                exit 1
            fi
            ;;
        "visa")
            DATA_DIR="data/VisA"
            if [ ! -d "$DATA_DIR" ]; then
                print_error "VisA 数据集未找到，请先下载数据集"
                exit 1
            fi
            ;;
        *)
            print_error "不支持的数据集: $DATASET"
            print_info "支持的数据集: mvtec, visa"
            exit 1
            ;;
    esac
    
    print_success "数据集检查完成: $DATA_DIR"
}

# 检查配置文件
check_config() {
    print_info "检查配置文件: $CONFIG"
    
    if [ ! -f "$CONFIG" ]; then
        print_error "配置文件未找到: $CONFIG"
        exit 1
    fi
    
    print_success "配置文件检查完成"
}

# 优化训练参数
optimize_training_params() {
    print_info "优化训练参数..."
    
    # 根据 GPU 内存调整参数
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    if [ "$GPU_MEMORY" -lt 12000 ]; then
        # 小显存优化
        export EXPERIENCE_BUFFER_SIZE=3000
        export PROJECTION_DIM=64
        export UPDATE_FREQUENCY=20
        BATCH_SIZE=16
        print_warning "检测到小显存，使用优化参数"
    elif [ "$GPU_MEMORY" -lt 24000 ]; then
        # 中等显存
        export EXPERIENCE_BUFFER_SIZE=5000
        export PROJECTION_DIM=100
        export UPDATE_FREQUENCY=10
        BATCH_SIZE=32
        print_info "使用标准参数"
    else
        # 大显存
        export EXPERIENCE_BUFFER_SIZE=8000
        export PROJECTION_DIM=128
        export UPDATE_FREQUENCY=5
        BATCH_SIZE=64
        print_info "使用高性能参数"
    fi
    
    print_success "参数优化完成"
}

# 创建训练配置
create_training_config() {
    print_info "创建训练配置..."
    
    cat > $EXPERIMENT_DIR/training_config.yaml << EOF
# AutoDL 训练配置
experiment_name: $EXPERIMENT_NAME
dataset: $DATASET
config_file: $CONFIG
timestamp: $(date)

# 硬件配置
gpus: $GPUS
batch_size: $BATCH_SIZE
max_epochs: $MAX_EPOCHS

# Experience Replay 配置
experience_replay:
  buffer_size: ${EXPERIENCE_BUFFER_SIZE:-5000}
  projection_dim: ${PROJECTION_DIM:-100}
  update_frequency: ${UPDATE_FREQUENCY:-10}
  similarity_threshold: 0.8

# 路径配置
paths:
  data_dir: $DATA_DIR
  checkpoint_dir: $CHECKPOINT_DIR
  result_dir: $RESULT_DIR
  log_dir: $EXPERIMENT_DIR
EOF
    
    print_success "训练配置创建完成"
}

# 启动训练
start_training() {
    print_info "开始训练模型..."
    
    # 创建训练命令
    case $DATASET in
        "mvtec")
            TRAIN_SCRIPT="scripts/train_mvtec.py"
            ;;
        "visa")
            TRAIN_SCRIPT="scripts/train_visa.py"
            ;;
    esac
    
    # 检查训练脚本是否存在
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        print_warning "训练脚本不存在，使用通用训练方法"
        TRAIN_SCRIPT="example_experience_replay_usage.py"
    fi
    
    # 构建训练命令
    TRAIN_CMD="python $TRAIN_SCRIPT"
    
    if [ -f "$CONFIG" ]; then
        TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
    fi
    
    TRAIN_CMD="$TRAIN_CMD --gpus $GPUS --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE"
    
    # 记录训练信息
    echo "训练命令: $TRAIN_CMD" > $EXPERIMENT_DIR/train_command.txt
    echo "开始时间: $(date)" >> $EXPERIMENT_DIR/train_info.txt
    
    print_info "执行训练命令: $TRAIN_CMD"
    
    # 启动训练并记录日志
    $TRAIN_CMD 2>&1 | tee $EXPERIMENT_DIR/training.log
    
    # 记录结束时间
    echo "结束时间: $(date)" >> $EXPERIMENT_DIR/train_info.txt
    
    print_success "训练完成！"
}

# 训练后处理
post_training() {
    print_info "执行训练后处理..."
    
    # 保存系统信息
    nvidia-smi > $EXPERIMENT_DIR/gpu_info.txt
    df -h > $EXPERIMENT_DIR/disk_info.txt
    
    # 检查是否有检查点文件
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
        print_success "检查点文件已保存到: $CHECKPOINT_DIR"
        ls -la $CHECKPOINT_DIR
    fi
    
    # 检查是否有结果文件
    if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR)" ]; then
        print_success "结果文件已保存到: $RESULT_DIR"
        ls -la $RESULT_DIR
    fi
    
    # 生成训练报告
    cat > $EXPERIMENT_DIR/training_report.md << EOF
# 训练报告

## 实验信息
- 实验名称: $EXPERIMENT_NAME
- 数据集: $DATASET
- 配置文件: $CONFIG
- 开始时间: $(head -1 $EXPERIMENT_DIR/train_info.txt | cut -d: -f2-)
- 结束时间: $(tail -1 $EXPERIMENT_DIR/train_info.txt | cut -d: -f2-)

## 硬件配置
- GPU 数量: $GPUS
- 批次大小: $BATCH_SIZE
- 最大轮数: $MAX_EPOCHS

## 文件路径
- 日志文件: $EXPERIMENT_DIR/training.log
- 检查点: $CHECKPOINT_DIR
- 结果: $RESULT_DIR

## 下一步
1. 查看训练日志: cat $EXPERIMENT_DIR/training.log
2. 运行测试: bash scripts/test_autodl.sh $DATASET $CHECKPOINT_DIR
3. 分析结果: python scripts/analyze_results.py $RESULT_DIR
EOF
    
    print_success "训练报告已生成: $EXPERIMENT_DIR/training_report.md"
}

# 错误处理
handle_error() {
    print_error "训练过程中发生错误"
    echo "错误时间: $(date)" >> $EXPERIMENT_DIR/train_info.txt
    
    # 保存错误信息
    if [ -f "$EXPERIMENT_DIR/training.log" ]; then
        tail -50 $EXPERIMENT_DIR/training.log > $EXPERIMENT_DIR/error.log
        print_info "错误日志已保存: $EXPERIMENT_DIR/error.log"
    fi
    
    exit 1
}

# 主函数
main() {
    echo "开始训练流程..."
    
    check_environment
    setup_training_env
    setup_training_dirs
    check_dataset
    check_config
    optimize_training_params
    create_training_config
    start_training
    post_training
    
    echo "================================================"
    print_success "🎉 训练流程完成！"
    echo ""
    echo "📋 训练结果:"
    echo "   - 实验目录: $EXPERIMENT_DIR"
    echo "   - 检查点: $CHECKPOINT_DIR"
    echo "   - 结果: $RESULT_DIR"
    echo ""
    echo "📊 查看结果:"
    echo "   - 训练日志: cat $EXPERIMENT_DIR/training.log"
    echo "   - 训练报告: cat $EXPERIMENT_DIR/training_report.md"
    echo "   - GPU 使用: cat $EXPERIMENT_DIR/gpu_info.txt"
}

# 设置错误处理
trap handle_error ERR

# 运行主函数
main "$@"