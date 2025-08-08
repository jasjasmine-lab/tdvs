# Git 仓库上传指南

本指南将帮助你将 Experience Replay 项目代码上传到你的 GitHub 仓库。

## 🚀 快速上传步骤

### 方法一：新建仓库（推荐）

#### 1. 在 GitHub 上创建新仓库
1. 登录 [GitHub](https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `Experience-Replay-Anomaly-Detection`
   - **Description**: `Experience Replay for Anomaly Detection with CDAD`
   - **Visibility**: Public 或 Private（根据需要选择）
   - **不要勾选** "Add a README file"、"Add .gitignore"、"Choose a license"
4. 点击 "Create repository"

#### 2. 在本地初始化 Git 仓库
```bash
# 进入项目目录
cd /Users/jasangela/One-for-More-1

# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: Experience Replay for Anomaly Detection"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/Experience-Replay-Anomaly-Detection.git

# 推送到远程仓库
git push -u origin main
```

### 方法二：使用现有仓库

如果你已经有一个仓库，想要更新代码：

```bash
# 进入项目目录
cd /Users/jasangela/One-for-More-1

# 如果还没有初始化 Git
git init

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 拉取远程仓库（如果有内容）
git pull origin main

# 添加所有文件
git add .

# 提交更改
git commit -m "Add Experience Replay implementation and AutoDL deployment scripts"

# 推送到远程仓库
git push origin main
```

## 📋 推送前检查清单

### 1. 清理不必要的文件
```bash
# 删除缓存文件
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# 删除临时文件
rm -rf logs/* checkpoints/* results/* 2>/dev/null || true
```

### 2. 创建 .gitignore 文件
```bash
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

# PyTorch
*.pth
*.pt
*.ckpt

# Data
data/*/
!data/.gitkeep

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

# Jupyter Notebook
.ipynb_checkpoints

# Model files (large files)
*.safetensors
*.bin
models/*.pth
models/*.pt
models/*.ckpt
EOF
```

### 3. 创建 requirements.txt
```bash
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
EOF
```

## 🔧 详细上传流程

### 步骤 1: 准备仓库
```bash
# 1. 清理项目
echo "🧹 清理项目文件..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# 2. 创建必要的目录占位符
mkdir -p data logs checkpoints results
touch data/.gitkeep logs/.gitkeep checkpoints/.gitkeep results/.gitkeep

echo "✅ 项目清理完成"
```

### 步骤 2: Git 初始化和配置
```bash
# 1. 初始化 Git（如果还没有）
if [ ! -d ".git" ]; then
    echo "🔧 初始化 Git 仓库..."
    git init
    echo "✅ Git 仓库初始化完成"
else
    echo "ℹ️  Git 仓库已存在"
fi

# 2. 配置 Git 用户信息（如果还没有配置）
if [ -z "$(git config user.name)" ]; then
    echo "请输入你的 Git 用户名:"
    read git_username
    git config user.name "$git_username"
fi

if [ -z "$(git config user.email)" ]; then
    echo "请输入你的 Git 邮箱:"
    read git_email
    git config user.email "$git_email"
fi

echo "✅ Git 配置完成"
```

### 步骤 3: 添加文件和提交
```bash
# 1. 添加所有文件
echo "📁 添加项目文件..."
git add .

# 2. 检查状态
echo "📊 检查 Git 状态:"
git status

# 3. 创建提交
echo "💾 创建提交..."
git commit -m "feat: Add Experience Replay for Anomaly Detection

- Implement CDAD_ExperienceReplay class with experience buffer
- Add AutoDL deployment scripts and guides
- Include training and testing automation
- Support MVTec-AD and VisA datasets
- Add comprehensive documentation and examples"

echo "✅ 提交创建完成"
```

### 步骤 4: 推送到远程仓库
```bash
# 1. 添加远程仓库
echo "请输入你的 GitHub 仓库地址 (例如: https://github.com/username/repo.git):"
read repo_url

# 检查是否已有远程仓库
if git remote get-url origin 2>/dev/null; then
    echo "🔄 更新远程仓库地址..."
    git remote set-url origin "$repo_url"
else
    echo "🔗 添加远程仓库..."
    git remote add origin "$repo_url"
fi

# 2. 推送到远程仓库
echo "🚀 推送到远程仓库..."
git push -u origin main

echo "🎉 代码上传完成！"
echo "📍 仓库地址: $repo_url"
```

## 🛠️ 自动化上传脚本

为了简化上传流程，我们提供了两个自动化脚本：

### 方式一：完整版脚本（推荐）

**特点：**
- 完整的环境检查和配置
- 详细的进度显示和错误处理
- 自动创建 .gitignore 和 requirements.txt
- 智能的 Git 配置管理

### 方式二：快速上传脚本

**特点：**
- 简化的操作流程
- 快速执行，适合熟悉 Git 的用户
- 基本的文件清理和提交
- 一键完成所有操作

创建一个自动化脚本来简化上传过程：

```bash
#!/bin/bash
# upload_to_github.sh - 自动上传脚本

set -e

echo "🚀 Experience Replay 项目自动上传脚本"
echo "========================================"

# 1. 清理项目
echo "🧹 清理项目文件..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# 2. 创建目录占位符
mkdir -p data logs checkpoints results
touch data/.gitkeep logs/.gitkeep checkpoints/.gitkeep results/.gitkeep

# 3. 初始化 Git
if [ ! -d ".git" ]; then
    git init
fi

# 4. 检查 Git 配置
if [ -z "$(git config user.name)" ]; then
    read -p "请输入 Git 用户名: " git_username
    git config user.name "$git_username"
fi

if [ -z "$(git config user.email)" ]; then
    read -p "请输入 Git 邮箱: " git_email
    git config user.email "$git_email"
fi

# 5. 添加文件
echo "📁 添加项目文件..."
git add .

# 6. 创建提交
echo "💾 创建提交..."
COMMIT_MSG="feat: Add Experience Replay for Anomaly Detection

- Implement CDAD_ExperienceReplay class
- Add AutoDL deployment scripts
- Include comprehensive documentation
- Support multiple datasets and configurations"

git commit -m "$COMMIT_MSG"

# 7. 推送到远程仓库
if [ -z "$1" ]; then
    read -p "请输入 GitHub 仓库地址: " repo_url
else
    repo_url="$1"
fi

if git remote get-url origin 2>/dev/null; then
    git remote set-url origin "$repo_url"
else
    git remote add origin "$repo_url"
fi

echo "🚀 推送到远程仓库..."
git push -u origin main

echo "🎉 上传完成！"
echo "📍 仓库地址: $repo_url"
echo "📖 查看项目: ${repo_url%.git}"
```

## 📝 使用说明

### 快速上传（推荐）
1. 保存上面的自动化脚本为 `upload_to_github.sh`
2. 给脚本执行权限：`chmod +x upload_to_github.sh`
3. 运行脚本：`./upload_to_github.sh`
4. 按提示输入仓库地址

### 手动上传
按照上面的详细步骤逐步执行

## 🔍 验证上传

上传完成后，你可以：
1. 访问你的 GitHub 仓库页面
2. 检查所有文件是否正确上传
3. 查看 README.md 是否正常显示
4. 测试 AutoDL 部署脚本是否可用

## 🚨 常见问题

### 问题 1: 推送被拒绝
```bash
# 解决方案：强制推送（谨慎使用）
git push -f origin main
```

### 问题 2: 文件太大
```bash
# 解决方案：使用 Git LFS
git lfs install
git lfs track "*.pth"
git lfs track "*.ckpt"
git add .gitattributes
```

### 问题 3: 认证失败
```bash
# 解决方案：使用 Personal Access Token
# 1. 在 GitHub 设置中生成 PAT
# 2. 使用 PAT 作为密码
```

---

**🎯 现在你可以轻松地将 Experience Replay 项目上传到你的 GitHub 仓库了！**