"""
配置文件
"""

import os
from pathlib import Path

# ==================== API配置 ====================
# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sktfaker")
OPENAI_MODEL = "qwen3-max"  # 阿里云通义千问模型（qwen3-max, qwen-plus, qwen-turbo）
EXTRACTION_MODEL = "qwen-plus"

# Embedding模型配置
EMBEDDING_MODEL = "text-embedding-v4"  # 或 "text-embedding-3-small"
EMBEDDING_DIMENSION = 2048  # large: 3072, small: 1536


# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "nature_neuroscience"  # 原始论文数据
VECTOR_DB_DIR = DATA_DIR / "vector_db"  # 向量数据库

# 确保目录存在
for dir_path in [DATA_DIR, PAPERS_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== 提取配置 ====================
# 最短段落长度（字符）
MIN_PARAGRAPH_LENGTH = 50

# 跳过的章节（不提取四元组）
SKIP_SECTIONS = [
    'references',
    'acknowledgments', 
    'acknowledgements',
    'data availability',
    'code availability',
    'author contributions',
    'competing interests',
    'supplementary information'
]

# LLM调用配置
MAX_RETRIES = 3  # 最大重试次数
TEMPERATURE = 0.1  # 温度（越低越稳定）
BATCH_SIZE = 5  # 批处理大小（控制API调用频率）


# ==================== 向量数据库配置 ====================
# 向量数据库类型：'chroma', 'pinecone', 'pgvector'
VECTOR_DB_TYPE = "chroma"

# ChromaDB配置
CHROMA_COLLECTION_NAME = "neuroscience"
CHROMA_PERSIST_DIRECTORY = str(VECTOR_DB_DIR / "chroma")

# Pinecone配置（如果使用）
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = "us-west1-gcp"
PINECONE_INDEX_NAME = "neuroscience-rag"


# ==================== RAG配置 ====================
# 检索配置
TOP_K_RETRIEVAL = 10  # 检索前K个最相关的SRUs
SIMILARITY_THRESHOLD = 0.7  # 相似度阈值

# 生成配置
GENERATION_MODEL = "gpt-4o"  # 生成时使用更强的模型
GENERATION_TEMPERATURE = 0.7  # 生成温度（允许更多创造性）
MAX_TOKENS = 2000  # 最大生成token数


# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "rag.log"
