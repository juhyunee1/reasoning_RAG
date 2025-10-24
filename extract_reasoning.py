"""
直接从论文提取科研推理链
生成reasoning_chains.jsonl格式
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm

class DirectReasoningExtractor:
    """直接提取科研推理链"""
    
    def __init__(self, api_key: str, model: str = "qwen3-max"):
        """初始化提取器"""
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 关键章节（用于提取推理链）
        # 注意：Nature Neuroscience的Introduction章节叫做"Main"
        self.key_sections = ['abstract', 'introduction', 'main', 'results', 'discussion', 'conclusion', 'methods']
    
    def load_metadata(self, metadata_file: Path) -> Dict[str, Dict]:
        """加载所有论文的元数据"""
        print("加载元数据...")
        metadata_dict = {}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取metadata"):
                if line.strip():
                    meta = json.loads(line)
                    metadata_dict[meta['id']] = meta
        
        print(f"✓ 加载了 {len(metadata_dict)} 篇论文的元数据")
        return metadata_dict
    
    def load_media(self, media_file: Path) -> Dict[str, List]:
        """
        加载所有论文的媒体数据（figures, tables等）
        
        注意：media.jsonl中每一行是一个图表，同一篇论文有多行
        需要按paper_id聚合
        """
        print("加载媒体数据...")
        media_dict = {}
        
        with open(media_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取media"):
                if line.strip():
                    media_item = json.loads(line)
                    paper_id = media_item.get('id')
                    
                    if paper_id:
                        # 将每个media item添加到对应paper的列表中
                        if paper_id not in media_dict:
                            media_dict[paper_id] = []
                        
                        # 提取关键信息
                        media_dict[paper_id].append({
                            'type': media_item.get('type'),
                            'label': media_item.get('label'),
                            'caption': media_item.get('caption'),
                            'legend': media_item.get('legend'),
                            'name': media_item.get('name')
                        })
        
        print(f"✓ 加载了 {len(media_dict)} 篇论文的媒体数据")
        
        # 统计信息
        total_media = sum(len(items) for items in media_dict.values())
        print(f"✓ 总计 {total_media} 个图表")
        
        return media_dict
    
    def parse_contents_file(self, contents_file: Path, max_papers: int = None) -> List[Dict]:
        """
        解析contents.jsonl文件（使用状态机处理字符串中的括号）
        """
        print(f"\n解析contents文件...")
        papers = []
        
        with open(contents_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pos = 0
        paper_count = 0
        failed_count = 0
        
        with tqdm(desc="解析论文") as pbar:
            while pos < len(content):
                # 跳过空白字符
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
                
                if pos >= len(content):
                    break
                
                # 找到JSON对象的起始位置
                if content[pos] != '{':
                    pos += 1
                    continue
                
                start = pos
                brace_count = 0
                in_string = False
                escape = False
                
                # 状态机：正确处理字符串中的括号
                for i in range(start, len(content)):
                    char = content[i]
                    
                    if escape:
                        escape = False
                        continue
                    
                    if char == '\\':
                        escape = True
                        continue
                    
                    if char == '"':
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                
                                try:
                                    paper = json.loads(content[start:end])
                                    papers.append(paper)
                                    paper_count += 1
                                    pbar.update(1)
                                    
                                    if max_papers and paper_count >= max_papers:
                                        print(f"✓ 已达到设定的数量限制: {max_papers} 篇")
                                        print(f"✓ 成功解析 {len(papers)} 篇论文")
                                        return papers
                                        
                                except json.JSONDecodeError as e:
                                    failed_count += 1
                                    if failed_count <= 5:  # 只显示前5个错误
                                        tqdm.write(f"⚠ 跳过解析失败的对象 #{paper_count+failed_count}: {str(e)[:80]}")
                                
                                pos = end
                                break
                else:
                    # 如果没找到匹配的右括号，说明文件结束或格式错误
                    break
        
        if failed_count > 0:
            print(f"⚠ 共有 {failed_count} 个对象解析失败（已跳过）")
        print(f"✓ 成功解析 {len(papers)} 篇论文")
        return papers
    
    def extract_key_content(self, paper_content: Dict, media_data: List = None, max_total_chars: int = 20000) -> str:
        """
        智能提取论文的关键内容用于推理链分析
        
        策略：
        1. 按章节优先级分配字符配额
        2. Figures插入到Methods和Results之间
        3. 保证Results和Discussion不被截断
        
        Args:
            paper_content: 论文内容
            media_data: 媒体数据（figures, tables等）
            max_total_chars: 最大字符数限制
            
        Returns:
            关键内容的文本
        """
        # 字符配额分配（确保关键章节不被截断）
        char_quotas = {
            'introduction': 3000,  # Introduction完整保留
            'main': 3000,          # Nature Neuroscience的Introduction
            'methods': 10000,       # Methods核心部分
            'results': 10000,       # Results完整保留
            'discussion': 6000     # Discussion完整保留
        }
        
        # 分别提取各章节内容
        sections_dict = {}
        
        for section in paper_content.get('sections', []):
            section_title = section.get('section_title', '')
            section_title_lower = section_title.lower()
            section_texts = section.get('section_text', [])
            
            # 检查是否是关键章节
            matched_key = None
            for key in char_quotas.keys():
                if key in section_title_lower:
                    matched_key = key
                    break
            
            if matched_key:
                section_content = []
                char_count = 0
                quota = char_quotas[matched_key]
                
                # 提取内容直到达到配额
                for text_item in section_texts:
                    if isinstance(text_item, dict):
                        text = text_item.get('content', '')
                    else:
                        text = str(text_item)
                    
                    text = text.strip()
                    if text:
                        # 检查是否超过配额
                        if char_count + len(text) > quota:
                            # 如果是Results或Discussion，强制包含
                            if matched_key in ['results', 'discussion']:
                                section_content.append(text)
                                char_count += len(text)
                            else:
                                break
                        else:
                            section_content.append(text)
                            char_count += len(text)
                
                if section_content:
                    # 根据章节类型添加标记
                    if matched_key in ['introduction', 'main']:
                        label = "[IMPORTANT FOR PROBLEM ANALYSIS]"
                    elif matched_key == 'methods':
                        label = "[IMPORTANT FOR DATA & METHODS]"
                    elif matched_key == 'results':
                        label = "[IMPORTANT FOR CONCLUSION - FINDINGS]"
                    elif matched_key == 'discussion':
                        label = "[IMPORTANT FOR CONCLUSION - SIGNIFICANCE]"
                    else:
                        label = ""
                    
                    sections_dict[matched_key] = f"\n## {section_title} {label}:\n" + "\n".join(section_content)
        
        # 提取Figures和Tables的captions
        figures_content = ""
        if media_data:
            figure_captions = []
            for media_item in media_data[:15]:  # 最多提取15个图表
                media_type = media_item.get('type', '').lower()
                caption = media_item.get('caption', '')
                legend = media_item.get('legend', '')
                name = media_item.get('name', '')
                
                if caption or legend or name:
                    caption_text = f"{media_type.upper()}: "
                    if name:
                        caption_text += f"{name}. "
                    if caption:
                        caption_text += caption
                    if legend:
                        caption_text += f" [Legend: {legend}]"
                    figure_captions.append(caption_text.strip())
            
            if figure_captions:
                figures_content = f"\n## Figures and Tables [IMPORTANT FOR DATA & METHODS]:\n" + "\n".join(figure_captions)
        
        # 按优先顺序组装内容：Introduction → Methods → Figures → Results → Discussion
        final_parts = []
        
        # 1. Introduction/Main
        if 'introduction' in sections_dict:
            final_parts.append(sections_dict['introduction'])
        elif 'main' in sections_dict:
            final_parts.append(sections_dict['main'])
        
        # 2. Methods
        if 'methods' in sections_dict:
            final_parts.append(sections_dict['methods'])
        
        # 3. Figures（插入到Methods和Results之间）
        if figures_content:
            final_parts.append(figures_content)
        
        # 4. Results（最重要，完整保留）
        if 'results' in sections_dict:
            final_parts.append(sections_dict['results'])
        
        # 5. Discussion（最重要，完整保留）
        if 'discussion' in sections_dict:
            final_parts.append(sections_dict['discussion'])
        
        return "\n".join(final_parts)
    
    def build_reasoning_prompt(self, paper_title: str, key_content: str) -> str:
        prompt = f"""You are an expert in scientific reasoning and research methodology in neuroscience. Your task is to extract the CORE SCIENTIFIC REASONING CHAIN from this paper in a CONCISE, STREAMLINED format.

Paper Title
{paper_title}

Key Content from Paper
{key_content[:30000]}  

Your Task
Extract the CORE reasoning logic in 4 CONCISE PARAGRAPHS. Focus on the ESSENCE, not details.

1. problem_decomposition: Core Research Question and Its Logical Breakdown
Read the Introduction/Main section strategically (early → middle → late paragraphs) and write ONE CONCISE PARAGRAPH (3-5 sentences) that captures:
- Early paragraphs: The broad background context and macroscopic problem
- Middle paragraphs: The mechanistic/phenotypic gap or unresolved question
- Late paragraphs: The specific hypothesis or core research question this paper addresses
Synthesize these three layers into a coherent logical flow: background → gap → hypothesis.

2. data: Data Sources and Requirements
Read the Methods section and Figure/Table captions and write ONE CONCISE PARAGRAPH (3-4 sentences) describing what data this research depends on. Include:
- Sample source: What subjects/models were used (species, age, strain, sample size)
- Data type: What was measured (neural recordings, behavior, imaging, molecular data)
- Sampling characteristics: Recording methods, temporal/spatial resolution, duration and brain regions
- Task conditions: What behavioral paradigms or experimental manipulations were applied
Use natural, flowing language to describe the data foundation of this study.

3. method: Experimental Design and Data Acquisition Methods
Read the Methods, Figure captions, and Results sections and write ONE CONCISE PARAGRAPH (4-5 sentences) describing HOW the required data were obtained. Include:
- Experimental design: Control vs. experimental groups, within/between-subject design, sample size rationale
- Data acquisition methods: Specific techniques and instruments used to collect each data type (e.g., recording setup, behavioral tracking, imaging parameters)
- Experimental conditions: Manipulated variables, control conditions, timing/sequence of interventions
- Analytical pipeline: Key processing and analysis steps linking raw data to testable predictions
Focus on the DESIGN LOGIC: what methods were chosen to obtain what data, and how the experimental design allows testing the hypothesis.

4. conclusion: Key Findings, Answer, and Significance
Read the Results and Discussion sections and write ONE CONCISE PARAGRAPH (4-6 sentences) that:
- Summarizes 2-3 main empirical findings
- States how these findings answer the core question
- Explains the broader scientific significance

Critical Instructions:
- In Nature Neuroscience papers, the "Main" section IS the Introduction
- Each field should be ONE CONCISE PARAGRAPH (not lists, not multiple paragraphs)
- Focus on CORE REASONING LOGIC, not exhaustive details
- Write in natural, flowing language suitable for embedding models
- Output in English

Output Format (JSON):
{{
    "problem_decomposition": "...",
    "data": "...",
    "method": "...",
    "conclusion": "..."
}}

Example Output:
{{
    "problem_decomposition": "Synapse development requires coordinated assembly of pre- and postsynaptic components at precise subcellular locations. While neuroligin-neurexin complexes are established synaptogenic pairs, the diversity of synaptic connections suggests additional trans-synaptic adhesion systems remain to be discovered. The mechanistic gap lies in identifying novel heterophilic receptor-ligand pairs that not only mediate cell-cell adhesion but also recruit intracellular scaffolds like PSD-95 to organize functional synapses. This leads to the hypothesis that NGL (netrin-G ligand) proteins, as postsynaptic adhesion molecules binding both presynaptic netrin-G and postsynaptic PSD-95, constitute a bidirectional synaptogenic system regulating excitatory synapse formation.",
    "data": "The study used cultured rat hippocampal neurons (embryonic day 18) and transfected HEK293T cells as experimental models, with n=30-50 neurons per condition across three independent cultures. Protein localization was assessed via confocal immunofluorescence and postembedding immunoelectron microscopy targeting synaptic markers (PSD-95, synapsin I, VGlut1). Functional measurements included whole-cell patch-clamp recordings of miniature excitatory postsynaptic currents (mEPSCs) to quantify synapse number and strength. Key manipulations included lentiviral overexpression, siRNA-mediated knockdown of endogenous NGL-2, and synaptogenic bead assays where NGL-coated microspheres were applied to axons to test presynaptic differentiation capacity.",
    "method": "The research employed a multi-level experimental strategy integrating molecular, structural, and functional analyses. NGL-2 was first identified as a PSD-95 interactor via yeast two-hybrid screening and validated by coimmunoprecipitation from brain lysates. To test synaptogenic function, neurons were transfected with NGL-2 or control vectors at DIV7 and analyzed at DIV14 for changes in synaptic puncta density and mEPSC frequency. Loss-of-function experiments used lentiviral shRNA to knock down endogenous NGL-2 from DIV5-14, followed by blind quantification of dendritic spine density and electrophysiology. Cell-surface presentation assays tested sufficiency: NGL-2-expressing HEK293 cells or antibody-coated beads were cocultured with neurons to induce presynaptic differentiation visualized by synapsin clustering. Statistical comparisons used one-way ANOVA with post-hoc Tukey tests.",
    "conclusion": "NGL-2 localizes specifically to excitatory postsynaptic sites and mediates trans-synaptic adhesion by binding presynaptic netrin-G2 in an isoform-specific manner while simultaneously recruiting PSD-95 via its C-terminal PDZ-binding motif. Overexpression increased excitatory synapse density and mEPSC frequency by 40%, whereas siRNA knockdown reduced synapse number by 35% without affecting inhibitory synapses, demonstrating selective regulation of excitatory synaptogenesis. Application of soluble NGL-2 ectodomain competitively disrupted existing synapses, confirming its necessity for synapse maintenance. These findings establish the NGL-netrin-G complex as a novel trans-synaptic organizing system parallel to neurexin-neuroligin, revealing molecular diversity in synapse specification with implications for understanding circuit wiring, synaptic plasticity, and neurodevelopmental disorders like autism spectrum disorders."
}}

Return ONLY the JSON object with 4 fields, no additional text.
"""
        return prompt
    
    def extract_reasoning_chain(self, paper_content: Dict, metadata: Dict, media_data: List = None) -> Optional[Dict]:
        """
        从单篇论文直接提取推理链
        
        Args:
            paper_content: 论文内容
            metadata: 论文元数据
            media_data: 媒体数据（figures, tables等）
            
        Returns:
            推理链数据（包含research_reasoning）
        """
        paper_title = paper_content.get('title', 'Unknown')
        
        # 提取关键内容（包括media captions）
        key_content = self.extract_key_content(paper_content, media_data)
        
        if not key_content or len(key_content) < 200:
            print(f"  ✗ 内容太少，跳过")
            return None
        
        # 构建Prompt
        prompt = self.build_reasoning_prompt(paper_title, key_content)
        
        # 调用LLM提取推理链
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in scientific reasoning extraction. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                reasoning = json.loads(response.choices[0].message.content)
                
                # 验证必要字段
                required_fields = ['problem_decomposition', 'data', 'method', 'conclusion']
                if not all(field in reasoning for field in required_fields):
                    raise ValueError(f"Missing required fields in LLM response")
                
                # 构建完整数据（LLM 直接返回段落格式）
                result = {
                    "paper_id": paper_content.get('id'),
                    "doi": paper_content.get('doi'),
                    "title": paper_title,
                    "journal": metadata.get('journal'),
                    "year": metadata.get('publish_year', '').split('/')[0] if metadata.get('publish_year') else '',
                    "citation_count": metadata.get('citation_count'),
                    "is_open_access": metadata.get('is_open_access'),
                    "authors": metadata.get('authors', []),
                    "article_url": metadata.get('article_url'),
                    
                    # 核心推理链（精炼的段落格式）
                    "problem_decomposition": reasoning['problem_decomposition'],
                    "data": reasoning['data'],
                    "method": reasoning['method'],
                    "conclusion": reasoning['conclusion']
                }
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  ✗ 提取失败: {e}")
                    return None
        
        return None
    
    def _load_processed_papers(self, output_file: Path) -> set:
        """
        从输出文件中加载已处理的论文ID
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            已处理的paper_id集合
        """
        processed_ids = set()
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'paper_id' in data:
                                processed_ids.add(data['paper_id'])
                        except:
                            continue
            except Exception as e:
                print(f"⚠ 警告: 读取已有输出文件失败: {e}")
        
        return processed_ids
    
    def _save_single_result(self, result: Dict, output_file: Path):
        """
        增量保存单个结果（追加模式）
        
        Args:
            result: 单个论文的推理链
            output_file: 输出文件路径
        """
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _save_failed_log(self, failed_list: list, output_file: Path):
        """
        保存失败日志
        
        Args:
            failed_list: 失败的论文列表 [{paper_id, title, error}, ...]
            output_file: 输出文件路径
        """
        log_file = output_file.parent / "failed_papers.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_failed": len(failed_list),
                "failed_papers": failed_list,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def batch_process(
        self,
        data_dir: Path,
        output_file: Path,
        max_papers: int = None,
        resume: bool = True
    ):
        """
        批量处理论文（支持断点续传）
        
        Args:
            data_dir: 数据目录
            output_file: 输出文件（reasoning_chains.jsonl）
            max_papers: 最多处理多少篇
            resume: 是否启用断点续传（默认True）
        """
        data_dir = Path(data_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("🧠 提取科研推理链（支持断点续传）")
        print("=" * 80)
        
        # 检查断点续传
        processed_ids = set()
        if resume:
            processed_ids = self._load_processed_papers(output_file)
            if processed_ids:
                print(f"✓ 发现已处理的论文: {len(processed_ids)} 篇")
                print(f"✓ 将跳过已处理的论文，从断点继续")
        
        # 加载元数据
        metadata_dict = self.load_metadata(data_dir / "metadata.jsonl")
        
        # 加载媒体数据
        media_dict = {}
        media_file = data_dir / "media.jsonl"
        if media_file.exists():
            media_dict = self.load_media(media_file)
        else:
            print("⚠ 警告: 未找到media.jsonl，将不提取图表信息")
        
        # 解析contents
        papers = self.parse_contents_file(
            data_dir / "contents.jsonl",
            max_papers=max_papers
        )
        
        if not papers:
            print("✗ 错误: 未找到论文数据")
            return
        
        # 过滤已处理的论文
        if resume and processed_ids:
            papers_to_process = [p for p in papers if p.get('id') not in processed_ids]
            print(f"\n✓ 总论文数: {len(papers)} 篇")
            print(f"✓ 已处理: {len(processed_ids)} 篇")
            print(f"✓ 待处理: {len(papers_to_process)} 篇")
        else:
            papers_to_process = papers
            print(f"\n✓ 将处理 {len(papers_to_process)} 篇论文")
        
        if not papers_to_process:
            print("\n✓ 所有论文已处理完成！")
            return
        
        print("=" * 80)
        
        # 处理每篇论文（增量保存）
        processed = 0
        failed = []
        
        for paper in tqdm(papers_to_process, desc="提取进度"):
            try:
                paper_id = paper.get('id')
                paper_title = paper.get('title', 'Unknown')
                metadata = metadata_dict.get(paper_id, {})
                media_data = media_dict.get(paper_id, [])
                
                print(f"\n处理: {paper_title[:60]}...")
                if media_data:
                    print(f"  找到 {len(media_data)} 个图表")
                
                # 提取推理链
                result = self.extract_reasoning_chain(paper, metadata, media_data)
                
                if result:
                    # 立即保存（增量写入）
                    self._save_single_result(result, output_file)
                    print(f"  ✓ 提取成功并已保存")
                    processed += 1
                else:
                    failed.append({
                        "paper_id": paper_id,
                        "title": paper_title,
                        "error": "提取失败"
                    })
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n✗ 处理失败: {paper.get('title', 'Unknown')[:40]}")
                print(f"   错误: {error_msg}")
                failed.append({
                    "paper_id": paper.get('id'),
                    "title": paper.get('title', 'Unknown'),
                    "error": error_msg
                })
                continue
        
        # 保存失败日志
        if failed:
            self._save_failed_log(failed, output_file)
            print(f"\n⚠ 失败日志已保存: {output_file.parent / 'failed_papers.json'}")
        
        # 总结
        total_processed = len(processed_ids) + processed
        print("\n" + "=" * 80)
        print("📊 处理总结")
        print("=" * 80)
        print(f"✓ 本次处理: {processed}/{len(papers_to_process)} 篇")
        print(f"✓ 累计已处理: {total_processed}/{len(papers)} 篇论文")
        print(f"✗ 本次失败: {len(failed)} 篇")
        
        if failed:
            print(f"\n失败的论文:")
            for fail_info in failed[:5]:
                print(f"  - {fail_info.get('title', 'Unknown')[:50]}")
                print(f"    ID: {fail_info.get('paper_id', 'N/A')}")
                print(f"    错误: {fail_info.get('error', 'Unknown')[:80]}")
            if len(failed) > 5:
                print(f"  ... 还有 {len(failed)-5} 篇（详见 failed_papers.json）")
        
        # 显示示例（从文件读取最后一个）
        if processed > 0 and output_file.exists():
            print("\n" + "=" * 80)
            print("📝 推理链示例（最后处理的一篇）")
            print("=" * 80)
            
            # 读取最后一行
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        example = json.loads(lines[-1].strip())
                        print(f"\n论文: {example.get('title', 'N/A')[:60]}...")
                        print(f"DOI: {example.get('doi', 'N/A')}")
                        print(f"期刊: {example.get('journal', 'N/A')}")
                        print(f"年份: {example.get('year', 'N/A')}")
                        
                        # 扁平化的推理链字段
                        print(f"\n🎯 问题拆解 (前200字符):")
                        problem_text = example.get('problem_decomposition', 'N/A')
                        print(f"  {problem_text[:200]}...")
                        
                        print(f"\n📊 数据需求 (前200字符):")
                        data_text = example.get('data', 'N/A')
                        print(f"  {data_text[:200]}...")
                        
                        print(f"\n🔬 研究方法 (前200字符):")
                        method_text = example.get('method', 'N/A')
                        print(f"  {method_text[:200]}...")
                        
                        print(f"\n✨ 结论 (前200字符):")
                        conclusion_text = example.get('conclusion', 'N/A')
                        print(f"  {conclusion_text[:200]}...")
            except Exception as e:
                print(f"⚠ 无法显示示例: {e}")
        
        print("\n" + "=" * 80)
        print("✅ 完成！")
        print("=" * 80)
        
        if resume and papers_to_process:
            print("\n💡 提示:")
            print("  - 如果中途中断，再次运行将自动从断点继续")
            print("  - 已处理的论文会被自动跳过")
            print(f"  - 输出文件: {output_file}")
        
        print("=" * 80)
        
        # 保存日志
        log_file = output_file.parent / "reasoning_extraction_log.json"
        log_data = {
            "total_papers": len(papers),
            "previously_processed": len(processed_ids),
            "this_batch_processed": processed,
            "cumulative_processed": total_processed,
            "this_batch_failed": len(failed),
            "resume_enabled": resume,
            "output_file": str(output_file),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    import argparse
    import config
    
    parser = argparse.ArgumentParser(description="直接提取科研推理链")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./nature_neuroscience",
        help="数据目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/reasoning_chains.jsonl",
        help="输出文件"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="最多处理多少篇论文（0=全部）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=config.OPENAI_API_KEY,
        help="API密钥"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.OPENAI_MODEL,
        help="模型名称"
    )
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = DirectReasoningExtractor(
        api_key=args.api_key,
        model=args.model
    )
    
    # 批量处理
    max_papers = None if args.max_papers == 0 else args.max_papers
    
    extractor.batch_process(
        data_dir=args.data_dir,
        output_file=args.output,
        max_papers=max_papers
    )


if __name__ == "__main__":
    main()

