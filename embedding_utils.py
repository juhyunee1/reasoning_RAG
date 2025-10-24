"""
Embedding 工具模块
支持 Qwen text-embedding-v4 模型
"""

import time
import asyncio
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm


class QwenEmbedder:
    """Qwen Embedding 封装类"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "text-embedding-v4",
        batch_size: int = 25,
        max_retries: int = 3
    ):
        """
        初始化 Embedder
        
        Args:
            api_key: Qwen API key
            model: embedding 模型名称
            batch_size: 批处理大小
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # 初始化客户端（使用同步客户端）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0  # 设置超时时间
        )
    
    def embed_single(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """
        对单个文本生成 embedding
        
        Args:
            text: 输入文本
            retry_count: 当前重试次数
            
        Returns:
            embedding 向量
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            # 确保获取实际的 embedding 数据
            embedding_data = response.data[0].embedding
            
            # 处理可能的异步结果
            if asyncio.iscoroutine(embedding_data):
                embedding_data = asyncio.run(embedding_data)
            
            # 检查是否是有效的 embedding
            if isinstance(embedding_data, list) and len(embedding_data) > 0:
                return embedding_data
            else:
                print(f"⚠️ 无效的 embedding 数据: {type(embedding_data)}")
                return None
            
        except Exception as e:
            if retry_count < self.max_retries:
                print(f"? Embedding 失败，重试 {retry_count + 1}/{self.max_retries}: {str(e)[:100]}")
                time.sleep(2 ** retry_count)  # 指数退避
                return self.embed_single(text, retry_count + 1)
            else:
                print(f"? Embedding 最终失败: {str(e)[:100]}")
                return None
    
    def embed_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        批量生成 embeddings
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            embedding 向量列表
        """
        embeddings = []
        
        # 分批处理
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="生成 Embeddings")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                embedding = self.embed_single(text)
                embeddings.append(embedding)
                
                # 避免触发速率限制
                time.sleep(0.1)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """获取 embedding 维度"""
        test_embedding = self.embed_single("test")
        return len(test_embedding) if test_embedding else 1536


def prepare_texts_for_embedding(reasoning_chain: Dict) -> Dict[str, str]:
    """
    准备推理链的各个字段用于 embedding
    
    Args:
        reasoning_chain: 单个推理链数据
        
    Returns:
        包含各字段文本的字典
    """
    return {
        'problem': reasoning_chain.get('problem_decomposition', ''),
        'data': reasoning_chain.get('data', ''),
        'method': reasoning_chain.get('method', ''),
        'conclusion': reasoning_chain.get('conclusion', ''),
        # 完整的推理链（用于综合检索）
        'full_chain': ' '.join([
            reasoning_chain.get('problem_decomposition', ''),
            reasoning_chain.get('data', ''),
            reasoning_chain.get('method', ''),
            reasoning_chain.get('conclusion', '')
        ])
    }

