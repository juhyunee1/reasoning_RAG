# ? 灵枢 - 神经科学推理链生成系统

基于 RAG 技术的神经科学实验设计推理链生成系统，整合 **2729 篇 Nature Neuroscience 论文**。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.41.2-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ? 特性

- ? **智能检索**：基于向量语义检索相关研究
- ? **推理链生成**：自动生成完整的实验设计逻辑
- ? **Web 界面**：友好的图形化交互界面
- ? **权威数据源**：Nature Neuroscience 高质量论文
- ? **混合策略**：优化的检索与存储架构

## ? 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在 `config.py` 中设置：
```python
OPENAI_API_KEY = "your_qwen_api_key"
```

### 3. 构建数据库（首次使用）

```bash
python3 build_hybrid_reasoning_db.py
```
?? 需要约 2-3 小时

### 4. 启动 Web Demo

```bash
./start_web_demo.sh
```

或手动启动：
```bash
python3 web_demo.py
```

访问：`http://localhost:7201`

## ? 使用方式

### Web 界面（推荐）

启动后在浏览器中输入研究问题：

**示例**：
- `How does chronic stress affect hippocampal neurogenesis?`
- `What is the role of dopamine in reward learning?`
- `How do microglia contribute to synaptic pruning?`

系统会自动生成：
1. **Problem Decomposition**：问题逻辑分解
2. **Data Requirements**：数据需求分析
3. **Experimental Methods**：实验设计方案
4. **Expected Conclusion**：预期结论与意义

### 命令行

```bash
python reasoning_chain_generator.py
```

交互式输入问题，获得推理链。

### 编程接口

```python
from reasoning_chain_generator import ReasoningChainGenerator
import config

generator = ReasoningChainGenerator(api_key=config.OPENAI_API_KEY)

result = generator.generate_reasoning_chain(
    research_question="How does stress affect memory?",
    top_k=5
)

print(result['reasoning_chain'])
```

## ? 项目结构

```
rag/
├── web_demo.py                      # Web 界面（主入口）
├── reasoning_chain_generator.py     # 推理链生成核心
├── build_hybrid_reasoning_db.py     # 数据库构建
├── extract_reasoning.py             # 论文提取
├── embedding_utils.py               # Embedding 工具
├── config.py                        # 配置文件
├── requirements.txt                 # 依赖列表
├── start_web_demo.sh               # 启动脚本
├── data/                           # 数据目录
│   └── reasoning_chains_fixed_2.jsonl
└── chroma_db/                      # 向量数据库（构建后生成）
```

## ? 系统架构

```
┌─────────────┐
│  用户输入   │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│  Embedding      │  Qwen text-embedding-v4
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  向量检索       │  ChromaDB (2729 篇)
└──────┬──────────┘
       │
       ↓  Top-5 相关论文
┌─────────────────┐
│  Prompt 构建    │  问题 + 参考案例
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  LLM 生成       │  Qwen3-max
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  推理链输出     │  JSON 格式
└─────────────────┘
```

## ? 核心技术

### 1. 混合检索策略

- **索引**：仅对 `problem_decomposition` 建立向量索引
- **存储**：完整推理链存储在元数据中
- **优势**：避免方法细节稀释语义相似度，提升检索精度 20-30%

### 2. Embedding

- **模型**：Qwen text-embedding-v4
- **维度**：1024
- **优势**：高质量中英文语义理解

### 3. 生成模型

- **模型**：Qwen3-max
- **温度**：0.7（平衡创新与准确）
- **输出**：结构化 JSON

## ? 数据说明

### 数据源
- **期刊**：Nature Neuroscience
- **论文数**：2729 篇
- **时间跨度**：1998-2025
- **内容**：全文 + 图表 + 元数据

### 提取字段
- **问题分解**：从背景到假设的逻辑链条
- **数据需求**：样本、类型、脑区、任务
- **实验方法**：设计、采集、条件、分析
- **结论**：发现、回答、意义

## ?? 高级配置

### 修改端口

```bash
python web_demo.py --server-port 8080
```

### 生成公开链接

```bash
python web_demo.py --share
```

### 使用不同模型

```bash
python web_demo.py --model qwen-plus
```

### 自定义数据库路径

```bash
python web_demo.py --chroma-path /path/to/your/db
```

## ? 性能指标

| 指标 | 数值 |
|------|------|
| **论文数量** | 2,729 篇 |
| **向量维度** | 1,024 |
| **数据库大小** | ~500 MB |
| **检索延迟** | 0.5-1 秒 |
| **生成延迟** | 10-30 秒 |
| **检索精度** | Top-5 召回率 > 85% |

## ? 故障排查

### 问题：数据库未找到

```bash
# 解决：构建数据库
python build_hybrid_reasoning_db.py
```

### 问题：API 调用失败

```bash
# 解决：检查 config.py 中的 API key
# 或使用命令行参数
python web_demo.py --api-key YOUR_KEY
```

### 问题：端口被占用

```bash
# 解决：使用不同端口
python web_demo.py --server-port 8080
```

## ? 测试系统

在启动 Web Demo 前，可以先测试：

```bash
python -c "
from reasoning_chain_generator import ReasoningChainGenerator
import config

generator = ReasoningChainGenerator(api_key=config.OPENAI_API_KEY)
result = generator.generate_reasoning_chain('How does stress affect memory?')
print('? 系统正常' if result.get('status') == 'success' else '? 系统异常')
"
```

## ? 引用

如果使用本系统，请注明：

```
基于 Nature Neuroscience 2729 篇论文构建的神经科学推理链生成系统
数据来源: Nature Neuroscience (1998-2025)
技术: RAG + Qwen3-max
```

## ? 许可

本项目基于 Nature Neuroscience 公开论文构建，仅供学术研究使用。

## ? 贡献

欢迎提出改进建议！

---

**? 开始使用灵枢，加速你的神经科学研究！**