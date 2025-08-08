#!/bin/bash
# Experience Replay 项目快速上传脚本
# 简化版本，适合快速操作

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Experience Replay 快速上传工具${NC}"
echo "================================================"

# 检查 Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git 未安装，请先安装 Git${NC}"
    exit 1
fi

# 获取仓库地址
if [ -z "$1" ]; then
    echo -e "${YELLOW}📝 请输入你的 GitHub 仓库地址:${NC}"
    echo "   格式: https://github.com/username/repository.git"
    echo -n "   仓库地址: "
    read -r REPO_URL
else
    REPO_URL="$1"
fi

echo -e "${BLUE}📦 准备上传到: $REPO_URL${NC}"

# 快速清理
echo -e "${YELLOW}🧹 清理临时文件...${NC}"
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# 创建基本的 .gitignore
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}📄 创建 .gitignore...${NC}"
    cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.DS_Store
Thumbs.db
*.log
logs/
checkpoints/
results/
data/*/
!data/.gitkeep
.env
.venv
venv/
EOF
fi

# Git 操作
echo -e "${YELLOW}🔧 Git 操作...${NC}"

# 初始化（如果需要）
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✅ Git 仓库初始化完成${NC}"
fi

# 添加文件
git add .
echo -e "${GREEN}✅ 文件添加完成${NC}"

# 提交
if ! git diff --cached --quiet; then
    git commit -m "feat: Upload Experience Replay Anomaly Detection Project

🚀 Complete Experience Replay implementation for anomaly detection
📁 Includes AutoDL deployment scripts and comprehensive documentation
🛠️ Ready for production deployment"
    echo -e "${GREEN}✅ 提交创建完成${NC}"
else
    echo -e "${YELLOW}⚠️  没有新的更改需要提交${NC}"
fi

# 设置远程仓库
if git remote get-url origin 2>/dev/null; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi
echo -e "${GREEN}✅ 远程仓库设置完成${NC}"

# 推送
echo -e "${YELLOW}🚀 推送到 GitHub...${NC}"
if git push -u origin main 2>/dev/null; then
    echo -e "${GREEN}✅ 推送成功！${NC}"
else
    echo -e "${YELLOW}⚠️  尝试强制推送...${NC}"
    git push -f -u origin main
    echo -e "${GREEN}✅ 强制推送完成！${NC}"
fi

# 完成
echo ""
echo -e "${GREEN}🎉 上传完成！${NC}"
echo "================================================"
echo -e "${BLUE}📍 仓库地址: ${REPO_URL%.git}${NC}"
echo ""
echo -e "${YELLOW}🚀 AutoDL 部署命令:${NC}"
echo "   git clone $REPO_URL"
echo "   cd $(basename ${REPO_URL%.git})"
echo "   ./quick_start_autodl.sh"
echo ""
echo -e "${GREEN}✨ 现在可以在 AutoDL 中使用你的代码了！${NC}"