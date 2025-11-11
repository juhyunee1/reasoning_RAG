#!/bin/bash
# 灵枢 - 神经科学推理链生成系统 Web Demo 启动脚本

echo "=================================="
echo "灵枢 Web Demo 启动脚本"
echo "=================================="

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python -c "import gradio" 2>/dev/null || {
    echo "警告: gradio 未安装，正在安装依赖..."
    pip install -r requirements.txt
}

# 检查数据库
if [ ! -d "./chroma_db" ]; then
    echo "错误: 未找到向量数据库 (./chroma_db)"
    echo "请先运行: python build_hybrid_reasoning_db.py"
    exit 1
fi

echo "启动 Web Demo..."
echo "访问地址: http://0.0.0.0:7201"
echo "=================================="

# 启动 Demo
python web_demo.py \
    --server-port 7860 \
    --server-name 0.0.0.0 \
    "$@"

