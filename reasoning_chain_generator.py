"""
推理链生成器
根据用户输入的研究问题，检索相关推理链并生成新的四元组
"""

import json
import asyncio
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from embedding_utils import QwenEmbedder


class ReasoningChainGenerator:
    """推理链生成器"""
    
    def __init__(
        self,
        api_key: str,
        chroma_path: str = "./chroma_db",
        collection_name: str = "neuroscience",
        generation_model: str = "qwen3-max",
        temperature: float = 0.7
    ):
        # 初始化 embedder
        self.embedder = QwenEmbedder(api_key=api_key)
        
        # 连接到向量数据库
        self.client = chromadb.PersistentClient(path=str(Path(chroma_path)))
        self.collection = self.client.get_collection(name=collection_name)
        
        # 初始化生成模型（使用同步客户端）
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0  # 设置超时时间
        )
        self.generation_model = generation_model
        self.temperature = temperature
        
        print(f"? 推理链生成器已就绪")
        print(f"  向量数据库: {chroma_path}")
        print(f"  推理链数量: {self.collection.count()}")
        print(f"  生成模型: {generation_model}")
    
    def generate_reasoning_chain(
        self,
        research_question: str,
        top_k: int = 5,
        return_references: bool = True
    ) -> Dict:
        """
        根据研究问题生成完整的四元组推理链
        
        Args:
            research_question: 用户的研究问题
            top_k: 检索 top-k 个相关推理链作为参考
            return_references: 是否返回参考的推理链
            
        Returns:
            生成的推理链和参考来源
        """
        print(f"\n研究问题: {research_question}")
        print("-" * 80)
        
        # 第一步：检索相关推理链
        print(f"? 检索 top-{top_k} 相关推理链...")
        retrieved = self._retrieve_similar_chains(research_question, top_k)
        
        # 处理可能的异步结果
        if asyncio.iscoroutine(retrieved):
            retrieved = asyncio.run(retrieved)
        
        if not retrieved:
            return {
                'status': 'error',
                'message': '没有找到相关的推理链参考',
                'reasoning_chain': None
            }
        
        print(f"? 找到 {len(retrieved)} 个相关推理链")
        
        # 第二步：构建 prompt
        prompt = self._build_generation_prompt(research_question, retrieved)
        
        # 第三步：生成四元组
        print("? 生成推理链...")
        generated_chain = self._call_llm(prompt)
        
        # 处理可能的异步结果
        if asyncio.iscoroutine(generated_chain):
            generated_chain = asyncio.run(generated_chain)
        
        # 第四步：解析生成结果
        parsed_chain = self._parse_generated_chain(generated_chain)
        
        result = {
            'status': 'success',
            'research_question': research_question,
            'reasoning_chain': parsed_chain,
            'raw_output': generated_chain
        }

        if return_references:
            result['references'] = self._format_references(retrieved)

        print("DEBUG: 返回类型：", type(result))
        try:
            json.dumps(result)
            print("✅ JSON 可序列化")
        except Exception as e:
            print("❌ JSON 序列化失败:", e)
            
        return result
    
    def _retrieve_similar_chains(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:
        """检索相关推理链"""
        # 生成查询 embedding
        query_embedding = self.embedder.embed_single(query)
        
        if query_embedding is None:
            print("? 查询 embedding 生成失败")
            return []
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        retrieved = []
        for i in range(len(results['ids'][0])):
            # 解析存储的推理链
            reasoning_chain_json = results['metadatas'][0][i].get('reasoning_chain', '{}')
            reasoning_chain = json.loads(reasoning_chain_json)
            
            retrieved.append({
                'title': results['metadatas'][0][i]['title'],
                'year': results['metadatas'][0][i]['year'],
                'citation_count': results['metadatas'][0][i]['citation_count'],
                'similarity': 1 - results['distances'][0][i],
                'reasoning_chain': reasoning_chain,
                'full_text': results['documents'][0][i]
            })
        
        return retrieved
    
    def _build_generation_prompt(
        self,
        research_question: str,
        retrieved_chains: List[Dict]
    ) -> str:
        """Build generation prompt"""
        
        # Build reference examples
        examples = []
        for i, item in enumerate(retrieved_chains[:5], 1):  # Use top 5 most relevant
            chain = item['reasoning_chain']
            examples.append(f"""
Example {i} (Similarity: {item['similarity']:.3f}, Citations: {item['citation_count']}):
Study: {item['title']} ({item['year']})

Problem Decomposition:
{chain['problem_decomposition']}

Data Requirements:
{chain['data']}

Experimental Methods:
{chain['method']}

Conclusion:
{chain['conclusion']}
""")
        
        examples_text = "\n".join(examples)
        
        prompt = f"""You are an expert in neuroscience experimental design. Your task is to generate a complete scientific reasoning chain (quadruple structure) based on the user's research question, referencing similar successful research cases.

User's Research Question:
{research_question}

Relevant Research Cases (for reference):
{examples_text}

Based on the above reference cases, design a complete scientific reasoning chain for the user's research question, including the following four components:

1. **Problem Decomposition**:
   - Start from the broad context and progressively focus on specific research hypotheses
   - Include: background question → mechanistic gap → core hypothesis
   - Express in a coherent paragraph of 3-5 sentences

2. **Data Requirements**:
   - Clearly specify what data is needed to test the hypothesis
   - Include: sample source, data type, sampling characteristics, brain regions, task conditions
   - Express in a coherent paragraph of 3-4 sentences

3. **Experimental Methods**:
   - Describe in detail how to design experiments to acquire the data
   - Include: experimental design, data acquisition methods, experimental conditions, analytical pipeline
   - Express in a coherent paragraph of 4-5 sentences

4. **Conclusion**:
   - Based on the hypothesis, predict potential findings and their scientific significance
   - Include: expected findings, how they answer the question, scientific significance
   - Express in a coherent paragraph of 4-6 sentences

**Important Requirements**:
- Reference the scientific logic and expression style from the examples, but innovate for the user's specific question
- Ensure logical coherence across the four components: problem → data → method → conclusion
- Express each component as a natural, fluent paragraph (not bullet points)
- Output in JSON format

Output Format:
{{
    "problem_decomposition": "...",
    "data": "...",
    "method": "...",
    "conclusion": "..."
}}

Output only JSON, no other explanations.
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for generation"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a neuroscience experimental design expert, skilled at extracting methodological insights from related research and designing new experimental protocols."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=3000
            )

            print("DEBUG: response type =", type(response))

            if hasattr(response, "result"):
                response = response.result

            if hasattr(response, "choices") and len(response.choices) > 0:
                message_content = response.choices[0].message.content
                return message_content or "LLM 响应为空"
            else:
                return "LLM 响应 choices 为空"
            
        except Exception as e:
            print(f"❌ LLM 调用失败: {str(e)}")
            return f"Generation failed: {str(e)}"
    
    def _parse_generated_chain(self, raw_output: str) -> Optional[Dict]:
        """解析生成的推理链"""
        try:
            # 尝试提取 JSON
            start = raw_output.find('{')
            end = raw_output.rfind('}') + 1
            
            if start == -1 or end == 0:
                print("? 未找到 JSON 格式，返回原始输出")
                return None
            
            json_str = raw_output[start:end]
            chain = json.loads(json_str)
            
            # 验证必需字段
            required = ['problem_decomposition', 'data', 'method', 'conclusion']
            if all(k in chain for k in required):
                return chain
            else:
                print("? JSON 缺少必需字段")
                return None
                
        except json.JSONDecodeError as e:
            print(f"? JSON 解析失败: {e}")
            return None
    
    def _format_references(self, retrieved: List[Dict]) -> List[Dict]:
        """格式化参考文献"""
        return [
            {
                'title': item['title'],
                'year': item['year'],
                'citation_count': item['citation_count'],
                'similarity': item['similarity']
            }
            for item in retrieved
        ]
    
    def print_reasoning_chain(self, chain: Dict, show_references: bool = True):
        """打印推理链（格式化输出）"""
        if not chain or chain.get('status') == 'error':
            print(f"? {chain.get('message', '生成失败')}")
            return
        
        reasoning = chain['reasoning_chain']
        
        if reasoning is None:
            print("\n" + "="*80)
            print("生成的推理链（原始输出）")
            print("="*80)
            print(chain['raw_output'])
            return
        
        print("\n" + "="*80)
        print("生成的科研推理链")
        print("="*80)
        
        print(f"\n【研究问题】")
        print(chain['research_question'])
        
        print(f"\n【1. Problem Decomposition - 问题分解】")
        print(reasoning['problem_decomposition'])
        
        print(f"\n【2. Data Requirements - 数据需求】")
        print(reasoning['data'])
        
        print(f"\n【3. Experimental Methods - 实验方法】")
        print(reasoning['method'])
        
        print(f"\n【4. Conclusion - 预期结论】")
        print(reasoning['conclusion'])
        
        if show_references and 'references' in chain:
            print(f"\n【参考文献】")
            for i, ref in enumerate(chain['references'][:5], 1):
                print(f"  {i}. {ref['title']} ({ref['year']})")
                print(f"     相似度: {ref['similarity']:.4f}, 引用数: {ref['citation_count']}")


def main():
    """交互式界面"""
    import config
    
    # 初始化生成器
    generator = ReasoningChainGenerator(api_key=config.OPENAI_API_KEY)
    
    print("\n" + "="*80)
    print("神经科学推理链生成器")
    print("="*80)
    print("输入你的研究问题，系统会生成完整的四元组推理链")
    print("（输入 'quit' 退出）")
    print("="*80)
    
    while True:
        question = input("\n>>> 研究问题: ").strip()
        if not question:
            continue
        
        if question.lower() == 'quit':
            break
        
        # 生成推理链
        result = generator.generate_reasoning_chain(
            research_question=question,
            top_k=5,
            return_references=True
        )
        
        # 打印结果
        generator.print_reasoning_chain(result)


if __name__ == "__main__":
    main()
