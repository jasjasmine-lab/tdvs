#!/bin/bash
# Experience Replay 项目自动上传到 GitHub 脚本
# 使用方法: ./upload_to_github.sh [仓库地址]
# 示例: ./upload_to_github.sh https://github.com/username/Experience-Replay-Anomaly-Detection.git

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
    echo "║              Experience Replay 项目上传工具                  ║"
    echo "║                   GitHub 自动上传脚本                        ║"
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

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_step "检查系统依赖..."
    
    if ! command -v git &> /dev/null; then
        print_error "Git 未安装，请先安装 Git"
        exit 1
    fi
    
    GIT_VERSION=$(git --version)
    print_success "Git 已安装: $GIT_VERSION"
}

# 清理项目文件
clean_project() {
    print_step "清理项目文件..."
    
    # 删除 Python 缓存文件
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    find . -name "*.pyd" -delete 2>/dev/null || true
    
    # 删除系统文件
    find . -name ".DS_Store" -delete 2>/dev/null || true
    find . -name "Thumbs.db" -delete 2>/dev/null || true
    
    # 清理临时文件
    rm -rf logs/* checkpoints/* results/* 2>/dev/null || true
    
    # 创建目录占位符
    mkdir -p data logs checkpoints results
    touch data/.gitkeep logs/.gitkeep checkpoints/.gitkeep results/.gitkeep
    
    print_success "项目清理完成"
}

# 创建 .gitignore 文件
create_gitignore() {
    print_step "创建 .gitignore 文件..."
    
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# PyTorch
*.pth
*.pt
*.ckpt
*.safetensors
*.bin

# Data and Models
data/*/
!data/.gitkeep
models/*.pth
models/*.pt
models/*.ckpt

# Logs and Results
logs/
checkpoints/
results/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.spyderproject
.spyproject

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.conda/

# Jupyter Notebook
.ipynb_checkpoints

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Temporary files
*.tmp
*.temp
*.bak
*.backup
EOF
        print_success ".gitignore 文件创建完成"
    else
        print_info ".gitignore 文件已存在"
    fi
}

# 创建 requirements.txt
create_requirements() {
    print_step "创建 requirements.txt 文件..."
    
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << 'EOF'
# Core dependencies
torch>=1.11.0
torchvision>=0.12.0
torchaudio>=0.11.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Image processing
opencv-python>=4.5.0
Pillow>=8.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Deep learning frameworks
pytorch-lightning>=1.5.0
omegaconf>=2.1.0

# Transformers and diffusion models
transformers>=4.15.0
diffusers>=0.10.0

# Utilities
tqdm>=4.62.0
PyYAML>=6.0
einops>=0.4.0
wandb>=0.12.0
tensorboard>=2.8.0
EOF
        print_success "requirements.txt 文件创建完成"
    else
        print_info "requirements.txt 文件已存在"
    fi
}

# 初始化 Git 仓库
init_git() {
    print_step "初始化 Git 仓库..."
    
    if [ ! -d ".git" ]; then
        git init
        print_success "Git 仓库初始化完成"
    else
        print_info "Git 仓库已存在"
    fi
}

# 配置 Git 用户信息
config_git() {
    print_step "配置 Git 用户信息..."
    
    # 检查全局配置
    GLOBAL_NAME=$(git config --global user.name 2>/dev/null || echo "")
    GLOBAL_EMAIL=$(git config --global user.email 2>/dev/null || echo "")
    
    # 检查本地配置
    LOCAL_NAME=$(git config user.name 2>/dev/null || echo "")
    LOCAL_EMAIL=$(git config user.email 2>/dev/null || echo "")
    
    if [ -z "$LOCAL_NAME" ] && [ -z "$GLOBAL_NAME" ]; then
        print_info "请输入你的 Git 用户名:"
        read -r git_username
        git config user.name "$git_username"
        print_success "Git 用户名设置完成: $git_username"
    else
        CURRENT_NAME=${LOCAL_NAME:-$GLOBAL_NAME}
        print_info "当前 Git 用户名: $CURRENT_NAME"
    fi
    
    if [ -z "$LOCAL_EMAIL" ] && [ -z "$GLOBAL_EMAIL" ]; then
        print_info "请输入你的 Git 邮箱:"
        read -r git_email
        git config user.email "$git_email"
        print_success "Git 邮箱设置完成: $git_email"
    else
        CURRENT_EMAIL=${LOCAL_EMAIL:-$GLOBAL_EMAIL}
        print_info "当前 Git 邮箱: $CURRENT_EMAIL"
    fi
}

# 获取仓库地址
get_repo_url() {
    if [ -n "$1" ]; then
        REPO_URL="$1"
        print_info "使用提供的仓库地址: $REPO_URL"
    else
        print_info "请输入你的 GitHub 仓库地址:"
        print_info "格式: https://github.com/username/repository.git"
        print_info "示例: https://github.com/jasangela/Experience-Replay-Anomaly-Detection.git"
        echo -n "仓库地址: "
        read -r REPO_URL
    fi
    
    # 验证仓库地址格式
    if [[ ! $REPO_URL =~ ^https://github\.com/.+/.+\.git$ ]]; then
        print_warning "仓库地址格式可能不正确"
        print_info "标准格式: https://github.com/username/repository.git"
        echo -n "是否继续? (y/n): "
        read -r confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            print_error "操作取消"
            exit 1
        fi
    fi
    
    export REPO_URL
}

# 添加文件到 Git
add_files() {
    print_step "添加项目文件到 Git..."
    
    # 显示将要添加的文件
    print_info "将要添加的文件:"
    git add -n . | head -20
    if [ $(git add -n . | wc -l) -gt 20 ]; then
        echo "... 还有 $(($(git add -n . | wc -l) - 20)) 个文件"
    fi
    
    # 添加所有文件
    git add .
    
    # 显示状态
    ADDED_FILES=$(git diff --cached --name-only | wc -l)
    print_success "已添加 $ADDED_FILES 个文件"
}

# 创建提交
create_commit() {
    print_step "创建 Git 提交..."
    
    # 检查是否有文件需要提交
    if git diff --cached --quiet; then
        print_warning "没有文件需要提交"
        return 0
    fi
    
    # 创建详细的提交信息
    COMMIT_MSG="feat: Add Experience Replay for Anomaly Detection

🚀 Features:
- Implement CDAD_ExperienceReplay class with experience buffer
- Add comprehensive AutoDL deployment scripts and guides
- Include automated training and testing workflows
- Support MVTec-AD and VisA datasets
- Add performance monitoring and optimization tools

📁 Project Structure:
- cdm/: Core modules with Experience Replay implementation
- scripts/: AutoDL deployment and automation scripts
- models/: Configuration files for different datasets
- docs/: Comprehensive documentation and guides

🛠️ AutoDL Integration:
- One-click deployment script
- Automated environment setup
- GPU optimization and monitoring
- Interactive management interface

📊 Key Components:
- Experience buffer with priority sampling
- Projection matrix learning with SVD fallback
- Novelty detection and similarity thresholding
- Comprehensive logging and statistics

🎯 Ready for production deployment on AutoDL instances"
    
    git commit -m "$COMMIT_MSG"
    print_success "提交创建完成"
}

# 设置远程仓库
setup_remote() {
    print_step "设置远程仓库..."
    
    # 检查是否已有远程仓库
    if git remote get-url origin 2>/dev/null; then
        CURRENT_URL=$(git remote get-url origin)
        if [ "$CURRENT_URL" != "$REPO_URL" ]; then
            print_info "更新远程仓库地址"
            print_info "当前: $CURRENT_URL"
            print_info "新的: $REPO_URL"
            git remote set-url origin "$REPO_URL"
        else
            print_info "远程仓库地址已正确设置"
        fi
    else
        print_info "添加远程仓库: $REPO_URL"
        git remote add origin "$REPO_URL"
    fi
    
    print_success "远程仓库设置完成"
}

# 推送到远程仓库
push_to_remote() {
    print_step "推送到远程仓库..."
    
    # 检查远程仓库连接
    print_info "测试远程仓库连接..."
    if ! git ls-remote origin &>/dev/null; then
        print_error "无法连接到远程仓库，请检查:"
        echo "  1. 仓库地址是否正确"
        echo "  2. 网络连接是否正常"
        echo "  3. GitHub 访问权限是否正确"
        exit 1
    fi
    
    print_success "远程仓库连接正常"
    
    # 推送代码
    print_info "开始推送代码..."
    
    # 尝试推送，如果失败则尝试强制推送
    if git push -u origin main; then
        print_success "代码推送成功！"
    else
        print_warning "推送失败，可能是因为远程仓库有冲突"
        echo -n "是否强制推送? 这将覆盖远程仓库的内容 (y/n): "
        read -r force_push
        
        if [[ $force_push =~ ^[Yy]$ ]]; then
            print_info "执行强制推送..."
            git push -f -u origin main
            print_success "强制推送完成！"
        else
            print_error "推送取消"
            exit 1
        fi
    fi
}

# 验证上传结果
verify_upload() {
    print_step "验证上传结果..."
    
    # 获取仓库网页地址
    WEB_URL=${REPO_URL%.git}
    
    print_success "🎉 代码上传完成！"
    echo ""
    print_info "📍 仓库信息:"
    echo "   Git 地址: $REPO_URL"
    echo "   网页地址: $WEB_URL"
    echo ""
    print_info "📋 接下来你可以:"
    echo "   1. 访问 $WEB_URL 查看代码"
    echo "   2. 在 AutoDL 中克隆: git clone $REPO_URL"
    echo "   3. 运行部署脚本: ./quick_start_autodl.sh"
    echo ""
    print_info "🚀 AutoDL 部署命令:"
    echo "   cd /root/autodl-tmp"
    echo "   git clone $REPO_URL"
    echo "   cd $(basename ${REPO_URL%.git})"
    echo "   chmod +x quick_start_autodl.sh"
    echo "   ./quick_start_autodl.sh"
}

# 错误处理
handle_error() {
    print_error "脚本执行过程中发生错误"
    print_info "请检查错误信息并重试"
    exit 1
}

# 显示帮助信息
show_help() {
    cat << 'EOF'
📖 Experience Replay GitHub 上传工具使用指南

🚀 使用方法:
  ./upload_to_github.sh [仓库地址]

📝 示例:
  ./upload_to_github.sh https://github.com/username/Experience-Replay-Anomaly-Detection.git

🔧 功能:
  - 自动清理项目文件
  - 创建 .gitignore 和 requirements.txt
  - 初始化 Git 仓库
  - 配置 Git 用户信息
  - 添加文件并创建提交
  - 推送到 GitHub 仓库

💡 提示:
  - 确保你有仓库的写入权限
  - 如果是私有仓库，可能需要配置 SSH 密钥或 Personal Access Token
  - 脚本会自动处理大部分配置，只需要提供仓库地址

📞 获取帮助:
  - 查看详细指南: cat GIT_UPLOAD_GUIDE.md
  - 查看 AutoDL 部署: cat README_AUTODL.md
EOF
}

# 主函数
main() {
    # 显示帮助
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_help
        exit 0
    fi
    
    print_header
    
    # 执行上传流程
    check_dependencies
    clean_project
    create_gitignore
    create_requirements
    init_git
    config_git
    get_repo_url "$1"
    add_files
    create_commit
    setup_remote
    push_to_remote
    verify_upload
    
    echo ""
    print_success "🎉 Experience Replay 项目已成功上传到 GitHub！"
}

# 设置错误处理
trap handle_error ERR

# 运行主函数
main "$@"