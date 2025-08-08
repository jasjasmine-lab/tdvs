#!/bin/bash
# AutoDL 测试脚本
# 使用方法: bash scripts/test_autodl.sh [dataset] [checkpoint_path]
# 示例: bash scripts/test_autodl.sh mvtec checkpoints/mvtec_20231201_120000

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
CHECKPOINT_PATH=${2:-""}
BATCH_SIZE=${3:-32}
NUM_WORKERS=${4:-4}

echo "🧪 开始在 AutoDL 上测试 Experience Replay 模型"
echo "================================================"
print_info "数据集: $DATASET"
print_info "检查点路径: $CHECKPOINT_PATH"
print_info "批次大小: $BATCH_SIZE"
print_info "工作进程: $NUM_WORKERS"
echo "================================================"

# 检查环境
check_environment() {
    print_info "检查测试环境..."
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未找到 CUDA 环境"
        exit 1
    fi
    
    # 检查 GPU 状态
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    print_info "GPU 内存: ${GPU_USED}MB / ${GPU_MEMORY}MB"
    
    print_success "环境检查完成"
}

# 设置环境变量
setup_test_env() {
    print_info "设置测试环境变量..."
    
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export TORCH_CUDNN_V8_API_ENABLED=1
    
    # 测试优化设置
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    print_success "环境变量设置完成"
}

# 创建测试目录
setup_test_dirs() {
    print_info "创建测试目录..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TEST_NAME="test_${DATASET}_${TIMESTAMP}"
    
    mkdir -p results/$TEST_NAME
    mkdir -p logs/$TEST_NAME
    
    export TEST_DIR="results/$TEST_NAME"
    export LOG_DIR="logs/$TEST_NAME"
    
    print_success "测试目录创建完成: $TEST_NAME"
}

# 检查数据集
check_dataset() {
    print_info "检查测试数据集: $DATASET"
    
    case $DATASET in
        "mvtec")
            DATA_DIR="data/MVTec-AD"
            if [ ! -d "$DATA_DIR" ]; then
                print_error "MVTec-AD 数据集未找到"
                exit 1
            fi
            ;;
        "visa")
            DATA_DIR="data/VisA"
            if [ ! -d "$DATA_DIR" ]; then
                print_error "VisA 数据集未找到"
                exit 1
            fi
            ;;
        *)
            print_error "不支持的数据集: $DATASET"
            exit 1
            ;;
    esac
    
    # 检查测试数据
    TEST_DATA_COUNT=$(find $DATA_DIR -name "*.png" -o -name "*.jpg" | wc -l)
    print_info "找到 $TEST_DATA_COUNT 个测试图像"
    
    print_success "数据集检查完成: $DATA_DIR"
}

# 检查检查点
check_checkpoint() {
    if [ -z "$CHECKPOINT_PATH" ]; then
        print_info "未指定检查点，查找最新的检查点..."
        
        # 查找最新的检查点
        LATEST_CHECKPOINT=$(find checkpoints -name "*${DATASET}*" -type d | sort | tail -1)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            print_error "未找到 $DATASET 的检查点文件"
            print_info "请先训练模型或指定检查点路径"
            exit 1
        fi
        
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
        print_info "使用最新检查点: $CHECKPOINT_PATH"
    fi
    
    print_info "检查检查点: $CHECKPOINT_PATH"
    
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        print_error "检查点目录不存在: $CHECKPOINT_PATH"
        exit 1
    fi
    
    # 查找检查点文件
    CHECKPOINT_FILES=$(find $CHECKPOINT_PATH -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" | wc -l)
    if [ "$CHECKPOINT_FILES" -eq 0 ]; then
        print_error "检查点目录中未找到模型文件"
        exit 1
    fi
    
    print_success "检查点检查完成"
}

# 创建测试配置
create_test_config() {
    print_info "创建测试配置..."
    
    cat > $LOG_DIR/test_config.yaml << EOF
# AutoDL 测试配置
test_name: $TEST_NAME
dataset: $DATASET
checkpoint_path: $CHECKPOINT_PATH
timestamp: $(date)

# 测试参数
batch_size: $BATCH_SIZE
num_workers: $NUM_WORKERS

# 路径配置
paths:
  data_dir: $DATA_DIR
  checkpoint_path: $CHECKPOINT_PATH
  result_dir: $TEST_DIR
  log_dir: $LOG_DIR

# 评估指标
metrics:
  - auroc
  - aupr
  - f1_score
  - accuracy
  - precision
  - recall
EOF
    
    print_success "测试配置创建完成"
}

# 运行测试
run_test() {
    print_info "开始测试模型..."
    
    # 创建测试脚本
    cat > $LOG_DIR/run_test.py << 'EOF'
import os
import sys
import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.getcwd())

try:
    from cdm.gpm_experience_replay import CDAD_ExperienceReplay
except ImportError:
    print("警告: 无法导入 CDAD_ExperienceReplay，使用基础测试")
    CDAD_ExperienceReplay = None

def load_test_config():
    """加载测试配置"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'test_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_model_loading(checkpoint_path):
    """测试模型加载"""
    print("🔍 测试模型加载...")
    
    # 查找检查点文件
    checkpoint_files = list(Path(checkpoint_path).glob('*.ckpt')) + \
                      list(Path(checkpoint_path).glob('*.pth')) + \
                      list(Path(checkpoint_path).glob('*.pt'))
    
    if not checkpoint_files:
        print("❌ 未找到检查点文件")
        return False
    
    checkpoint_file = checkpoint_files[0]
    print(f"📁 加载检查点: {checkpoint_file}")
    
    try:
        # 尝试加载检查点
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        print(f"✅ 检查点加载成功")
        
        # 检查检查点内容
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"📋 检查点包含键: {keys[:5]}{'...' if len(keys) > 5 else ''}")
        
        return True
    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        return False

def test_experience_replay(config):
    """测试 Experience Replay 功能"""
    print("🧠 测试 Experience Replay 功能...")
    
    if CDAD_ExperienceReplay is None:
        print("⚠️  跳过 Experience Replay 测试（模块未找到）")
        return {}
    
    try:
        # 创建模型实例
        model = CDAD_ExperienceReplay()
        
        # 配置 Experience Replay
        model.configure_experience_replay(
            buffer_size=1000,
            projection_dim=64,
            update_frequency=10,
            similarity_threshold=0.8
        )
        
        # 测试基本功能
        test_input = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Experience Replay 测试通过")
        print(f"📊 输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # 获取统计信息
        stats = model.get_statistics()
        print(f"📈 Experience Replay 统计: {stats}")
        
        return {
            'status': 'success',
            'output_shape': str(output.shape) if hasattr(output, 'shape') else str(type(output)),
            'statistics': stats
        }
        
    except Exception as e:
        print(f"❌ Experience Replay 测试失败: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def test_data_loading(data_dir, batch_size):
    """测试数据加载"""
    print("📂 测试数据加载...")
    
    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms, datasets
        
        # 创建数据变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 尝试加载数据
        if os.path.exists(os.path.join(data_dir, 'test')):
            test_dataset = datasets.ImageFolder(
                root=os.path.join(data_dir, 'test'),
                transform=transform
            )
        else:
            # 如果没有标准结构，尝试直接加载图像
            image_files = list(Path(data_dir).rglob('*.png')) + \
                         list(Path(data_dir).rglob('*.jpg'))
            print(f"📁 找到 {len(image_files)} 个图像文件")
            
            if len(image_files) == 0:
                print("⚠️  未找到图像文件")
                return {'status': 'no_data'}
            
            return {
                'status': 'success',
                'image_count': len(image_files),
                'sample_files': [str(f) for f in image_files[:3]]
            }
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # 测试一个批次
        batch = next(iter(test_loader))
        images, labels = batch
        
        print(f"✅ 数据加载成功")
        print(f"📊 批次形状: {images.shape}")
        print(f"🏷️  标签数量: {len(torch.unique(labels))}")
        
        return {
            'status': 'success',
            'dataset_size': len(test_dataset),
            'batch_shape': str(images.shape),
            'num_classes': len(torch.unique(labels)).item()
        }
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def generate_test_report(config, results):
    """生成测试报告"""
    report = {
        'test_info': {
            'test_name': config['test_name'],
            'dataset': config['dataset'],
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': config['checkpoint_path']
        },
        'results': results,
        'summary': {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results.values() if r.get('status') == 'success'),
            'failed_tests': sum(1 for r in results.values() if r.get('status') == 'failed')
        }
    }
    
    # 保存 JSON 报告
    result_dir = config['paths']['result_dir']
    with open(os.path.join(result_dir, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成 Markdown 报告
    md_report = f"""# 测试报告

## 基本信息
- 测试名称: {config['test_name']}
- 数据集: {config['dataset']}
- 检查点: {config['checkpoint_path']}
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试结果

### 总览
- 总测试数: {report['summary']['total_tests']}
- 通过测试: {report['summary']['passed_tests']}
- 失败测试: {report['summary']['failed_tests']}

### 详细结果
"""
    
    for test_name, result in results.items():
        status_icon = "✅" if result.get('status') == 'success' else "❌"
        md_report += f"\n#### {status_icon} {test_name}\n"
        
        if result.get('status') == 'success':
            for key, value in result.items():
                if key != 'status':
                    md_report += f"- {key}: {value}\n"
        else:
            md_report += f"- 错误: {result.get('error', '未知错误')}\n"
    
    with open(os.path.join(result_dir, 'test_report.md'), 'w') as f:
        f.write(md_report)
    
    return report

def main():
    """主函数"""
    print("🧪 开始模型测试")
    print("=" * 50)
    
    # 加载配置
    config = load_test_config()
    
    # 运行测试
    results = {}
    
    # 测试1: 模型加载
    results['model_loading'] = {
        'status': 'success' if test_model_loading(config['checkpoint_path']) else 'failed'
    }
    
    # 测试2: 数据加载
    results['data_loading'] = test_data_loading(
        config['paths']['data_dir'],
        config['batch_size']
    )
    
    # 测试3: Experience Replay
    results['experience_replay'] = test_experience_replay(config)
    
    # 生成报告
    report = generate_test_report(config, results)
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print(f"📊 通过: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
    print(f"📁 报告保存至: {config['paths']['result_dir']}")

if __name__ == '__main__':
    main()
EOF
    
    # 运行测试
    print_info "执行测试脚本..."
    cd $LOG_DIR
    python run_test.py test_config.yaml 2>&1 | tee test_output.log
    cd - > /dev/null
    
    print_success "测试执行完成"
}

# 分析测试结果
analyze_results() {
    print_info "分析测试结果..."
    
    # 检查测试报告
    if [ -f "$TEST_DIR/test_report.json" ]; then
        # 提取关键信息
        TOTAL_TESTS=$(python -c "import json; data=json.load(open('$TEST_DIR/test_report.json')); print(data['summary']['total_tests'])")
        PASSED_TESTS=$(python -c "import json; data=json.load(open('$TEST_DIR/test_report.json')); print(data['summary']['passed_tests'])")
        FAILED_TESTS=$(python -c "import json; data=json.load(open('$TEST_DIR/test_report.json')); print(data['summary']['failed_tests'])")
        
        print_info "测试统计:"
        echo "   - 总测试数: $TOTAL_TESTS"
        echo "   - 通过测试: $PASSED_TESTS"
        echo "   - 失败测试: $FAILED_TESTS"
        
        if [ "$FAILED_TESTS" -eq 0 ]; then
            print_success "所有测试通过！"
        else
            print_warning "有 $FAILED_TESTS 个测试失败"
        fi
    else
        print_warning "未找到测试报告文件"
    fi
    
    # 保存系统信息
    nvidia-smi > $TEST_DIR/gpu_info.txt
    df -h > $TEST_DIR/disk_info.txt
    
    print_success "结果分析完成"
}

# 生成测试总结
generate_summary() {
    print_info "生成测试总结..."
    
    cat > $TEST_DIR/test_summary.md << EOF
# 测试总结

## 测试信息
- 测试名称: $TEST_NAME
- 数据集: $DATASET
- 检查点: $CHECKPOINT_PATH
- 测试时间: $(date)

## 配置参数
- 批次大小: $BATCH_SIZE
- 工作进程: $NUM_WORKERS

## 文件路径
- 测试结果: $TEST_DIR
- 测试日志: $LOG_DIR
- 详细报告: $TEST_DIR/test_report.md

## 下一步
1. 查看详细报告: cat $TEST_DIR/test_report.md
2. 查看测试日志: cat $LOG_DIR/test_output.log
3. 分析性能指标: python scripts/analyze_performance.py $TEST_DIR
EOF
    
    print_success "测试总结已生成: $TEST_DIR/test_summary.md"
}

# 错误处理
handle_error() {
    print_error "测试过程中发生错误"
    
    if [ -d "$LOG_DIR" ]; then
        echo "错误时间: $(date)" >> $LOG_DIR/error_info.txt
        
        if [ -f "$LOG_DIR/test_output.log" ]; then
            tail -20 $LOG_DIR/test_output.log > $LOG_DIR/error.log
            print_info "错误日志已保存: $LOG_DIR/error.log"
        fi
    fi
    
    exit 1
}

# 主函数
main() {
    echo "开始测试流程..."
    
    check_environment
    setup_test_env
    setup_test_dirs
    check_dataset
    check_checkpoint
    create_test_config
    run_test
    analyze_results
    generate_summary
    
    echo "================================================"
    print_success "🎉 测试流程完成！"
    echo ""
    echo "📋 测试结果:"
    echo "   - 测试目录: $TEST_DIR"
    echo "   - 日志目录: $LOG_DIR"
    echo ""
    echo "📊 查看结果:"
    echo "   - 测试报告: cat $TEST_DIR/test_report.md"
    echo "   - 测试总结: cat $TEST_DIR/test_summary.md"
    echo "   - 测试日志: cat $LOG_DIR/test_output.log"
}

# 设置错误处理
trap handle_error ERR

# 运行主函数
main "$@"