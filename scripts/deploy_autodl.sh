#!/bin/bash
# AutoDL 一键部署脚本
# 使用方法: bash scripts/deploy_autodl.sh

set -e  # 遇到错误立即退出

echo "🚀 开始在 AutoDL 实例中部署 Experience Replay 项目..."
echo "================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

# 检查是否在 AutoDL 环境中
check_autodl_env() {
    print_info "检查 AutoDL 环境..."
    if [ ! -d "/root/autodl-tmp" ]; then
        print_warning "未检测到 AutoDL 环境，创建工作目录..."
        mkdir -p /root/autodl-tmp
    fi
    print_success "环境检查完成"
}

# 更新系统包
update_system() {
    print_info "更新系统包..."
    apt update -qq && apt upgrade -y -qq
    apt install -y -qq git wget curl vim htop tree
    print_success "系统更新完成"
}

# 检查 CUDA 环境
check_cuda() {
    print_info "检查 CUDA 环境..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        print_success "CUDA 环境正常"
    else
        print_error "未检测到 CUDA 环境"
        exit 1
    fi
}

# 安装 Python 依赖
install_dependencies() {
    print_info "安装 Python 依赖包..."
    
    # 升级 pip
    pip install --upgrade pip -q
    
    # 安装核心依赖
    print_info "安装 PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    
    print_info "安装科学计算库..."
    pip install numpy scipy scikit-learn -q
    
    print_info "安装图像处理库..."
    pip install opencv-python pillow -q
    
    print_info "安装可视化库..."
    pip install matplotlib seaborn -q
    
    print_info "安装深度学习框架..."
    pip install pytorch-lightning omegaconf -q
    
    print_info "安装其他工具..."
    pip install transformers diffusers tqdm -q
    
    # 如果存在 requirements.txt，也安装它
    if [ -f "requirements.txt" ]; then
        print_info "安装 requirements.txt 中的依赖..."
        pip install -r requirements.txt -q
    fi
    
    print_success "依赖安装完成"
}

# 创建项目目录结构
setup_directories() {
    print_info "创建项目目录结构..."
    
    # 创建必要目录
    mkdir -p data/{MVTec-AD,VisA}
    mkdir -p logs
    mkdir -p checkpoints
    mkdir -p results
    mkdir -p project/logs
    
    print_success "目录结构创建完成"
}

# 设置环境变量
setup_environment() {
    print_info "设置环境变量..."
    
    # 设置 CUDA 设备
    export CUDA_VISIBLE_DEVICES=0
    
    # 设置 Python 路径
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    
    # 设置 PyTorch 优化
    export TORCH_CUDNN_V8_API_ENABLED=1
    
    # 将环境变量写入 .bashrc
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> ~/.bashrc
    echo "export TORCH_CUDNN_V8_API_ENABLED=1" >> ~/.bashrc
    
    print_success "环境变量设置完成"
}

# 下载示例数据（可选）
download_sample_data() {
    print_info "是否下载 MVTec-AD 示例数据？(y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "下载 MVTec-AD 数据集..."
        cd data/MVTec-AD
        
        # 下载一个类别的数据作为示例
        wget -q https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/bottle.tar.xz
        tar -xf bottle.tar.xz
        rm bottle.tar.xz
        
        cd ../..
        print_success "示例数据下载完成"
    else
        print_info "跳过数据下载"
    fi
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    # 检查 Python 包
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
    
    # 检查项目文件
    if [ -f "cdm/gpm_experience_replay.py" ]; then
        print_success "项目文件检查通过"
    else
        print_error "项目文件不完整"
        exit 1
    fi
    
    print_success "安装验证完成"
}

# 创建快速启动脚本
create_quick_start() {
    print_info "创建快速启动脚本..."
    
    cat > quick_start.sh << 'EOF'
#!/bin/bash
# 快速启动脚本

echo "🎯 Experience Replay 项目快速启动"
echo "================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "1. 运行使用示例"
echo "2. 训练 MVTec 模型"
echo "3. 测试模型"
echo "4. 查看系统状态"
echo "5. 退出"

read -p "请选择操作 (1-5): " choice

case $choice in
    1)
        echo "运行 Experience Replay 使用示例..."
        python example_experience_replay_usage.py
        ;;
    2)
        echo "开始训练 MVTec 模型..."
        python scripts/train_mvtec.py --config models/cdad_mvtec.yaml
        ;;
    3)
        echo "测试模型..."
        python scripts/test_mvtec.py --config models/cdad_mvtec.yaml
        ;;
    4)
        echo "系统状态:"
        nvidia-smi
        df -h
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        ;;
esac
EOF
    
    chmod +x quick_start.sh
    print_success "快速启动脚本创建完成"
}

# 主函数
main() {
    echo "开始部署流程..."
    
    check_autodl_env
    update_system
    check_cuda
    install_dependencies
    setup_directories
    setup_environment
    download_sample_data
    verify_installation
    create_quick_start
    
    echo "================================================"
    print_success "🎉 AutoDL 部署完成！"
    echo ""
    echo "📋 接下来你可以："
    echo "   1. 运行快速启动脚本: ./quick_start.sh"
    echo "   2. 查看使用示例: python example_experience_replay_usage.py"
    echo "   3. 开始训练: python scripts/train_mvtec.py --config models/cdad_mvtec.yaml"
    echo "   4. 查看部署指南: cat AUTODL_DEPLOYMENT_GUIDE.md"
    echo ""
    echo "💡 提示：使用 'tmux' 或 'screen' 来运行长时间任务"
    echo "📊 监控：使用 'nvidia-smi' 查看 GPU 状态"
}

# 错误处理
trap 'print_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 运行主函数
main "$@"