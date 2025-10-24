"""
混合策略向量数据库构建器
核心思路：用 problem_decomposition 建立索引，但元数据存储完整推理链

适用场景：
- 用户输入宏观问题
- 需要精准匹配问题层面
- 但希望获取完整推理链作为生成参考
"""

import json
import chromadb
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import config
from embedding_utils import QwenEmbedder


class HybridReasoningDBBuilder:
    """混合策略数据库构建器"""
    
    def __init__(
        self,
        api_key: str,
        chroma_path: str = "./chroma_db",
        collection_name: str = "neuroscience"
    ):
        """
        初始化构建器
        
        Args:
            api_key: Qwen API key
            chroma_path: ChromaDB 持久化路径
            collection_name: 集合名称
        """
        self.embedder = QwenEmbedder(api_key=api_key)
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        
        # 初始化 ChromaDB
        print(f"\n初始化 ChromaDB: {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # 获取 embedding 维度
        print("获取 embedding 维度...")
        self.embedding_dim = self.embedder.get_embedding_dimension()
        print(f"✓ Embedding 维度: {self.embedding_dim}")
    
    def load_reasoning_chains(self, jsonl_file: Path) -> List[Dict]:
        """加载推理链数据"""
        print(f"\n加载推理链数据: {jsonl_file}")
        chains = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                if line.strip():
                    chains.append(json.loads(line))
        
        print(f"✓ 加载了 {len(chains)} 条推理链")
        return chains
    
    def create_collection(self, reset: bool = False):
        """创建或获取 collection"""
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"✓ 已删除旧的 collection: {self.collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "混合策略：用问题检索，返回完整推理链",
                "embedding_model": "text-embedding-v4",
                "embedding_dimension": self.embedding_dim,
                "strategy": "hybrid",
                "index_field": "problem_decomposition",
                "return_field": "full_chain"
            }
        )
        
        print(f"✓ Collection 已就绪: {self.collection_name}")
        print(f"  当前文档数: {self.collection.count()}")
    
    def build_database(self, reasoning_chains: List[Dict]):
        """
        构建混合策略数据库
        
        核心逻辑：
        1. 只对 problem_decomposition 进行 embedding（检索索引）
        2. 但在元数据中存储完整的推理链（返回内容）
        3. 检索时精准匹配问题，返回时获得完整推理链
        """
        print(f"\n开始构建向量数据库...")
        print(f"策略: 混合策略（问题索引 + 完整推理链存储）")
        print(f"总论文数: {len(reasoning_chains)}")
        
        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        
        failed_count = 0
        
        for chain in tqdm(reasoning_chains, desc="处理论文"):
            paper_id = chain.get('paper_id', 'unknown')
            title = chain.get('title', 'Unknown Title')
            
            # 提取四元组
            problem = chain.get('problem_decomposition', '')
            data = chain.get('data', '')
            method = chain.get('method', '')
            conclusion = chain.get('conclusion', '')
            
            # 检查完整性
            if not all([problem, data, method, conclusion]):
                tqdm.write(f"⚠ 跳过不完整的推理链: {title}")
                failed_count += 1
                continue
            
            # 🔑 关键：只对 problem_decomposition 进行 embedding
            # 构造检索文档（只包含问题部分）
            index_text = f"""Research Title: {title}

Problem Decomposition:
{problem}"""
            
            # 生成 embedding（只基于问题）
            embedding = self.embedder.embed_single(index_text)
            
            if embedding is None:
                tqdm.write(f"✗ Embedding 失败: {title}")
                failed_count += 1
                continue
            
            # 🔑 关键：在元数据中存储完整推理链
            reasoning_chain_dict = {
                'problem_decomposition': problem,
                'data': data,
                'method': method,
                'conclusion': conclusion
            }
            
            # 元数据（包含完整推理链）
            metadata = {
                'paper_id': paper_id,
                'title': title,
                'doi': chain.get('doi', ''),
                'year': chain.get('year', 0),
                'citation_count': chain.get('citation_count', 0),
                'journal': chain.get('journal', ''),
                'is_open_access': chain.get('is_open_access', False),
                # 完整的推理链（JSON 字符串）
                'reasoning_chain': json.dumps(reasoning_chain_dict, ensure_ascii=False)
            }
            
            all_ids.append(paper_id)
            all_embeddings.append(embedding)
            all_documents.append(index_text)  # 存储索引文本（用于显示）
            all_metadatas.append(metadata)
            
            # 批量插入
            if len(all_ids) >= 50:
                self.collection.add(
                    ids=all_ids,
                    embeddings=all_embeddings,
                    documents=all_documents,
                    metadatas=all_metadatas
                )
                all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []
        
        # 插入剩余的
        if all_ids:
            self.collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                documents=all_documents,
                metadatas=all_metadatas
            )
        
        print(f"\n✓ 构建完成！")
        print(f"  成功: {self.collection.count()} 篇论文")
        print(f"  失败: {failed_count} 篇论文")
        print(f"\n数据结构：")
        print(f"  - 索引向量: 基于 problem_decomposition")
        print(f"  - 元数据: 完整推理链（4个字段）")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="构建混合策略向量数据库（问题索引 + 完整推理链）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/reasoning_chains_fixed_2.jsonl",
        help="推理链 JSONL 文件路径"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="./chroma_db",
        help="ChromaDB 存储路径"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置已存在的数据库"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Qwen API Key"
    )
    
    args = parser.parse_args()
    
    # 获取 API key
    api_key = args.api_key or config.OPENAI_API_KEY
    print("="*80)
    print("策略说明：")
    print("  1. 索引：只对 problem_decomposition 建立向量索引")
    print("  2. 存储：元数据中保存完整的四元组")
    print("  3. 检索：用户问题精准匹配 problem，避免被方法稀释")
    print("  4. 返回：获取完整推理链作为生成参考")
    print("="*80)
    
    # 创建构建器
    builder = HybridReasoningDBBuilder(
        api_key=api_key,
        chroma_path=args.chroma_path
    )
    
    # 加载数据
    chains = builder.load_reasoning_chains(Path(args.input))
    
    # 创建 collection
    builder.create_collection(reset=args.reset)
    
    # 构建数据库
    builder.build_database(chains)
    
    print("\n" + "="*80)
    print("✓ 向量数据库构建完成！")
    print("="*80)
    print(f"\n数据库位置: {args.chroma_path}")
    print(f"Collection 名称: neuroscience")
    print(f"\n下一步：")
    print(f"  使用 reasoning_chain_generator.py 生成推理链")
    print(f"  （修改 chroma_path 参数为 './chroma_db'）")


if __name__ == "__main__":
    main()
