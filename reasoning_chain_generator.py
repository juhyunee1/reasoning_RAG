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
        generation_model: str = "qwen3-max", # gemini-2.5-pro
        temperature: float = 0.3,  # 0.2 ~ 0.4
        top_p: float = 0.9  # 0.85 ~ 0.95
    ):
        # 初始化 embedder（使用768维，与数据库构建时保持一致）
        self.embedder = QwenEmbedder(api_key=api_key, dimensions=768)
        
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
        self.top_p = top_p
        
        print(f"? 推理链生成器已就绪")
        print(f"  向量数据库: {chroma_path}")
        print(f"  推理链数量: {self.collection.count()}")
        print(f"  生成模型: {generation_model}")
    
    def generate_reasoning_chain(
        self,
        research_question: str,
        top_k: int = 15,
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
        
        # 第四步：解析生成结果（包含完整内容和摘要）
        parsed_result = self._parse_generated_chain(generated_chain)
        
        if parsed_result is None:
            return {
                'status': 'error',
                'message': '无法解析生成的推理链',
                'reasoning_chain': None,
                'raw_output': generated_chain
            }
        
        # 提取完整内容和摘要
        reasoning_chain = parsed_result.get('reasoning_chain')
        summary = parsed_result.get('summary')
        
        result = {
            'status': 'success',
            'research_question': research_question,
            'reasoning_chain': reasoning_chain,  # 完整内容
            'summary': summary,  # 摘要
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
                'full_text': results['documents'][0][i],
                'authors': results['metadatas'][0][i].get('authors', '[]'),
                'journal': results['metadatas'][0][i].get('journal', 'Unknown Journal')
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
        for i, item in enumerate(retrieved_chains[:15], 1):  # Use top 15 most relevant
            chain = item['reasoning_chain']
            
            # Check if this is an abstract-only entry (SFN abstracts) or full reasoning chain
            if 'abstract' in chain:
                # SFN abstract format
                examples.append(f"""
Example {i} (Similarity: {item['similarity']:.3f}, Citations: {item['citation_count']}):
Study: {item['title']} ({item['year']})

Abstract:
{chain['abstract']}
""")
            else:
                # Full reasoning chain format
                examples.append(f"""
Example {i} (Similarity: {item['similarity']:.3f}, Citations: {item['citation_count']}):
Study: {item['title']} ({item['year']})

Problem Decomposition:
{chain.get('problem_decomposition', 'N/A')}

Data Requirements:
{chain.get('data', 'N/A')}

Experimental Methods:
{chain.get('method', 'N/A')}

Conclusion:
{chain.get('conclusion', 'N/A')}
""")
        
        examples_text = "\n".join(examples)
        
        prompt = f"""You are an expert in neuroscience experimental design. Your task is to generate a mechanistically grounded, hypothesis-driven experimental plan—not a generic description—based on the user's research question and retrieved scientific examples.

User's Research Question:
{research_question}

Relevant Research Cases (for reference):
{examples_text}

Each retrieved paper contains a concise scientific reasoning chain with four components:
- problem_decomposition: background → knowledge gap → hypothesis
- data: what data are required and why
- method: how the data are acquired and why these methods fit the question
- conclusion: expected findings and scientific significance

---

You MUST ensure:
1) The experimental paradigm is SPECIFIC and WELL-DEFINED  
   - No naming paradigms without explaining their structure (e.g., not just "fear conditioning"; instead: trial structure, cues, reinforcement schedule, timing logic, behavioral readouts)

2) Every method directly maps to answering the core hypothesis  
   - NO method stacking
   - For each major technique, explicitly state what question it tests and what data it contributes

1. problem_decomposition
- Begin by identifying what is already known about this biological process or circuit.  
- Then clearly state what remains unknown or unresolved — highlight the specific gap that the user’s question targets.  
- Finally, state what this experiment aims to determine or test, i.e., the specific variable, mechanism, or causal link that must be experimentally probed.  
- Keep focus on what this design needs to solve, not a general field problem.
- Must explicitly state WHAT mechanistic variable drives the expected behavior/neural state change

2. data
- Sample specifications: species, strain, age range, sex balance, expected sample size rationale, inclusion/exclusion logic
- Data modalities: neural activity (spiking, LFP, calcium, EEG), behavior metrics, molecular/anatomical markers if relevant
- Recording characteristics: spatial and temporal resolution, brain regions, timescales, sampling frequency, expected signal quality
- Task paradigms: Behavioral paradigms, experimental manipulations, control conditions, training protocols
- Describe the structure of the behavior task

3. method
- Experimental logic: Define the independent variable(s), dependent variable(s), and causal contrast, specify how the manipulation directly tests the proposed mechanism
- Behavior-neural linkage: Explain how neural measurements align with task variables
- Controls & counterfactuals: Negative / positive controls, counterbalanced group logic, sham conditions, how you rule out confounds (sensory, motor, arousal, learning rate, etc.)
- Perturbations (if used): Conceptual description of perturbation strategy, state purpose: test necessity / sufficiency / circuit path
- Analysis reasoning: Core analytic approach (not code), what statistical comparison answers the hypothesis
- Quality & ethics: Blind conditions, replication, exclusion logic, ethical approvals

4. conclusion
- Answer to research question: How predicted results directly address the core hypothesis
- State how these findings resolve the unknown and fill the knowledge gap
- Summarizes the logical chain: Data -> Method -> Conclusion -> Scientific Findings 

Critical Instructions:
- NO vague paradigm naming (must describe task logic)
- Each field should be ONE COMPREHENSIVE PARAGRAPH (not lists, not multiple paragraphs)
- Write like a PI planning a grant, not a student listing methods
- Draw methodological insights from the reference cases while innovating for the specific question
- Use detailed, technical language appropriate for scientific publication
- No word limits - prioritize completeness and scientific rigor
- Output in English

Output Format (JSON):
{{
    "problem_decomposition": "COMPREHENSIVE detailed paragraph",
    "problem_summary": "Concise summary (50-100 words)",
    "data": "COMPREHENSIVE detailed paragraph",
    "data_summary": "Concise summary (50-100 words)",
    "method": "COMPREHENSIVE detailed paragraph",
    "method_summary": "Concise summary (50-100 words)",
    "conclusion": "COMPREHENSIVE detailed paragraph",
    "conclusion_summary": "Concise summary (50-100 words)"
}}

Return ONLY the JSON object with 8 fields, no additional text.
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for generation with timeout and retry"""
        max_retries = 3
        timeout = 120  
        
        for attempt in range(max_retries):
            try:
                print(f"? LLM调用尝试 {attempt + 1}/{max_retries}...")
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
                    top_p=self.top_p,
                    timeout=timeout  # 设置超时时间
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
                print(f"❌ LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return f"Generation failed after {max_retries} attempts: {str(e)}"
                else:
                    print(f"? 等待2秒后重试...")
                    import time
                    time.sleep(2)
        
        return "Generation failed: Max retries exceeded"
    
    def _parse_generated_chain(self, raw_output: str) -> Optional[Dict]:
        """解析生成的推理链（包含完整内容和摘要）"""
        try:
            # 尝试提取 JSON
            start = raw_output.find('{')
            end = raw_output.rfind('}') + 1
            
            if start == -1 or end == 0:
                print("? 未找到 JSON 格式，返回原始输出")
                return None
            
            json_str = raw_output[start:end]
            chain = json.loads(json_str)
            
            # 验证必需字段（8个字段）
            required_full = ['problem_decomposition', 'data', 'method', 'conclusion']
            required_summary = ['problem_summary', 'data_summary', 'method_summary', 'conclusion_summary']
            
            if all(k in chain for k in required_full) and all(k in chain for k in required_summary):
                # 分离完整内容和摘要
                reasoning_chain = {
                    'problem_decomposition': chain['problem_decomposition'],
                    'data': chain['data'],
                    'method': chain['method'],
                    'conclusion': chain['conclusion']
                }
                
                summary = {
                    'problem_decomposition': chain['problem_summary'],
                    'data': chain['data_summary'],
                    'method': chain['method_summary'],
                    'conclusion': chain['conclusion_summary']
                }
                
                return {
                    'reasoning_chain': reasoning_chain,
                    'summary': summary
                }
            else:
                print("? JSON 缺少必需字段")
                return None
                
        except json.JSONDecodeError as e:
            print(f"? JSON 解析失败: {e}")
            return None
    
    def _format_references(self, retrieved: List[Dict]) -> List[Dict]:
        """格式化参考文献"""
        formatted_refs = []
        for item in retrieved:
            # 解析作者信息
            authors = []
            if 'authors' in item and item['authors']:
                try:
                    authors_data = json.loads(item['authors']) if isinstance(item['authors'], str) else item['authors']
                    # 提取作者姓名（处理 [{"name": "作者1"}, {"name": "作者2"}] 格式）
                    if isinstance(authors_data, list) and len(authors_data) > 0:
                        if isinstance(authors_data[0], dict) and 'name' in authors_data[0]:
                            authors = [author['name'] for author in authors_data]
                        else:
                            authors = authors_data
                except:
                    authors = []
            
            # 格式化作者姓名（取前3个作者，超过3个用"等"）
            if authors:
                if len(authors) <= 3:
                    author_str = ', '.join(authors)
                else:
                    author_str = ', '.join(authors[:3]) + '等'
            else:
                author_str = 'Unknown Authors'
            
            formatted_refs.append({
                'title': item['title'],
                'year': item['year'],
                'citation_count': item['citation_count'],
                'similarity': item['similarity'],
                'authors': author_str,
                'journal': item.get('journal', 'Unknown Journal')
            })
        
        return formatted_refs
    
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
            for i, ref in enumerate(chain['references'][:15], 1):
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
            top_k=15,
            return_references=True
        )
        
        # 打印结果
        generator.print_reasoning_chain(result)


if __name__ == "__main__":
    main()
