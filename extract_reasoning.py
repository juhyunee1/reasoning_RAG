"""
直接从论文提取科研推理链
生成reasoning_chains.jsonl格式
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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
        self.key_sections = ['abstract', 'introduction', 'main', 'results', 'discussion', 'conclusion', 'methods']
    
    def load_metadata(self, metadata_file: Path) -> Dict[str, Dict]:
        """
        加载所有论文的元数据
        
        同时通过 id 和 doi 建立索引，支持双重匹配
        """
        print("加载元数据...")
        metadata_dict = {}
        metadata_dict_by_doi = {}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取metadata"):
                if line.strip():
                    meta = json.loads(line)
                    # 通过 id 索引
                    if 'id' in meta:
                        metadata_dict[meta['id']] = meta
                    # 通过 doi 索引（作为备选）
                    if 'doi' in meta and meta['doi']:
                        metadata_dict_by_doi[meta['doi']] = meta
        
        print(f"✓ 加载了 {len(metadata_dict)} 篇论文的元数据")
        print(f"✓ 其中 {len(metadata_dict_by_doi)} 篇有 DOI 信息")
        
        # 返回双重索引字典
        return {
            'by_id': metadata_dict,
            'by_doi': metadata_dict_by_doi
        }
    
    def load_media(self, media_file: Path) -> Dict[str, List]:
        """
        加载所有论文的媒体数据（figures, tables等）
        
        注意：media.jsonl中每一行是一个图表，同一篇论文有多行
        需要按paper_id或doi聚合，支持双重匹配
        """
        print("加载媒体数据...")
        media_dict_by_id = {}
        media_dict_by_doi = {}
        
        with open(media_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取media"):
                if line.strip():
                    media_item = json.loads(line)
                    
                    # 提取关键信息
                    media_info = {
                        'type': media_item.get('type'),
                        'label': media_item.get('label'),
                        'caption': media_item.get('caption'),
                        'legend': media_item.get('legend'),
                        'name': media_item.get('name')
                    }
                    
                    # 通过 id 索引
                    paper_id = media_item.get('id')
                    if paper_id:
                        if paper_id not in media_dict_by_id:
                            media_dict_by_id[paper_id] = []
                        media_dict_by_id[paper_id].append(media_info)
                    
                    # 通过 doi 索引（作为备选）
                    paper_doi = media_item.get('doi')
                    if paper_doi:
                        if paper_doi not in media_dict_by_doi:
                            media_dict_by_doi[paper_doi] = []
                        media_dict_by_doi[paper_doi].append(media_info)
        
        total_papers_by_id = len(media_dict_by_id)
        total_papers_by_doi = len(media_dict_by_doi)
        total_media = sum(len(items) for items in media_dict_by_id.values())
        
        print(f"✓ 通过 ID 索引: {total_papers_by_id} 篇论文")
        print(f"✓ 通过 DOI 索引: {total_papers_by_doi} 篇论文")
        print(f"✓ 总计 {total_media} 个图表")
        
        # 返回双重索引字典
        return {
            'by_id': media_dict_by_id,
            'by_doi': media_dict_by_doi
        }
    
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
    
    def extract_key_content(self, paper_content: Dict, media_data: List = None, max_total_chars: int = None, is_review: bool = False) -> str:
        """
        提取论文的完整关键内容用于推理链分析
        
        策略：
        - 研究论文模式：提取所有关键章节的完整内容（introduction, main, methods, results, discussion, conclusion）
        - 综述模式：保留所有正文内容，排除References、Rights And Permissions、Author Information等部分
        
        Args:
            paper_content: 论文内容
            media_data: 媒体数据（figures, tables等）
            max_total_chars: 最大字符数（暂未使用）
            is_review: 是否是综述论文（默认False）
        Returns:
            关键内容的完整文本
        """
        sections_list = paper_content.get('sections', [])
        
        # 综述模式：保留所有正文内容，排除不需要的章节
        if is_review:
            # 需要排除的章节关键词
            exclude_keywords = [
                'references', 'reference',
                'rights and permissions', 'rights & permissions',
                'author information', 'author contributions', 'authors',
                'acknowledgments', 'acknowledgement',
                'supplementary', 'supplemental',
                'data availability', 'code availability',
                'competing interests', 'conflict of interest',
                'ethics', 'ethical',
                'peer review', 'reviewer',
                'publisher', 'copyright',
                'about this article', 'about the article',
                'article history', 'publication history'
            ]
            
            final_parts = []
            
            # 遍历所有章节，排除不需要的章节
            for section in sections_list:
                section_title = section.get('section_title', '')
                section_title_lower = section_title.lower()
                
                # 检查是否需要排除
                should_exclude = False
                for keyword in exclude_keywords:
                    if keyword in section_title_lower:
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                # 提取章节内容
                section_texts = section.get('section_text', [])
                section_content = []
                
                for text_item in section_texts:
                    if isinstance(text_item, dict):
                        text = text_item.get('content', '')
                    else:
                        text = str(text_item)
                    
                    text = text.strip()
                    if text:
                        section_content.append(text)
                
                if section_content:
                    final_parts.append(f"\n## {section_title}:\n" + "\n".join(section_content))
            
            # 添加图表信息
            if media_data:
                figure_captions = []
                for media_item in media_data:
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
                    final_parts.append(f"\n## Figures and Tables:\n" + "\n".join(figure_captions))
            
            return "\n".join(final_parts)
        
        # 研究论文模式：提取关键章节
        # 关键章节列表
        key_section_keys = ['introduction', 'main', 'methods', 'results', 'discussion', 'conclusion']
        
        # 不同期刊的章节命名规范：
        # - nature: Abstract, Main, Discussion, Methods, 无 Results
        # - nature neuroscience: Abstract, Main, Results, Discussion, Methods
        # - nature communication: Abstract, Introduction, Results, Discussion, Methods 
        # - neuron: Summary, Keywords, Introduction, Results, Discussion, Experimental Procedures
        # - cell: Summary, Kerwords, Introduction, Results, Discussion, Experimental Procedures
        # - nature reviews neuroscience: Abstract, Main, Results, Discussion, Methods
        # - science: Abstract, No Results, Discussion
        
        # Methods 章节的各种命名（需要特别处理）
        methods_keywords = ['methods', 'method', 'experimental procedures', 'experimental procedure', 
                           'materials and methods', 'materials & methods', 'procedures']
        
        sections_dict = {}
        
        # 特殊处理：Nature 期刊没有 Results 标题，而是用 Main 和 Discussion 之间的章节作为 Results
        # 先找出 Main 和 Discussion 的位置
        main_idx = None
        discussion_idx = None
        
        for idx, section in enumerate(sections_list):
            section_title = section.get('section_title', '')
            section_title_lower = section_title.lower()
            
            # 识别 Main（注意：Main 标题通常很短，避免误匹配包含"main"的其他标题）
            if 'main' in section_title_lower and len(section_title) < 20:
                main_idx = idx
            # 识别 Discussion
            if 'discussion' in section_title_lower:
                discussion_idx = idx
        
        # 提取 Main 和 Discussion 之间的章节作为 Results（仅当两者都存在且之间有章节时）
        nature_results_sections = []
        if main_idx is not None and discussion_idx is not None and discussion_idx > main_idx + 1:
            for idx in range(main_idx + 1, discussion_idx):
                nature_results_sections.append(idx)
        
        # 遍历所有章节进行匹配
        for idx, section in enumerate(sections_list):
            section_title = section.get('section_title', '')
            section_title_lower = section_title.lower()
            section_texts = section.get('section_text', [])
            
            # 检查是否是关键章节
            matched_key = None
            
            # 特殊处理 1: 如果是 Nature 的 Results 章节（Main 和 Discussion 之间）
            if idx in nature_results_sections:
                matched_key = 'results'
            # 特殊处理 2: 先检查是否是 Methods 相关章节
            # Neuron 使用 "Experimental Procedures"，需要匹配到 methods
            elif any(keyword in section_title_lower for keyword in methods_keywords):
                matched_key = 'methods'
            # 特殊处理 3: 其他章节的匹配
            else:
                for key in key_section_keys:
                    if key in section_title_lower:
                        matched_key = key
                        break
            
            if matched_key:
                section_content = []
                
                # 完整提取所有内容，不截断
                for text_item in section_texts:
                    if isinstance(text_item, dict):
                        text = text_item.get('content', '')
                    else:
                        text = str(text_item)
                    
                    text = text.strip()
                    if text:
                        section_content.append(text)
                
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
                    
                    # 对于 Results，如果已有内容，则追加（Nature 期刊可能有多个 Results 章节）
                    if matched_key == 'results' and matched_key in sections_dict:
                        # 追加到已有的 Results 内容
                        existing_content = sections_dict[matched_key]
                        new_content = f"\n## {section_title}:\n" + "\n".join(section_content)
                        sections_dict[matched_key] = existing_content + "\n" + new_content
                    else:
                        sections_dict[matched_key] = f"\n## {section_title} {label}:\n" + "\n".join(section_content)
        
        # 提取Figures和Tables的captions（完整提取，无数量限制）
        figures_content = ""
        if media_data:
            figure_captions = []
            # 提取所有图表，不再限制数量
            for media_item in media_data:
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
{key_content}  

Your Task
Extract the CORE reasoning logic in 4 CONCISE PARAGRAPHS. Focus on the ESSENCE, not details.

1. problem_decomposition: Core Research Question and Its Logical Breakdown
Read the Introduction/Main section strategically (early → middle → late paragraphs) and write ONE CONCISE PARAGRAPH  that captures:
- Early paragraphs: The broad background context and macroscopic problem
- Middle paragraphs: The mechanistic/phenotypic gap or unresolved question
- Late paragraphs: The specific hypothesis or core research question this paper addresses
Synthesize these three layers into a coherent logical flow: background → gap → hypothesis.

2. data: Data Sources and Requirements
Read the Methods section and Figure/Table captions and write ONE CONCISE PARAGRAPH  describing what data this research depends on. Include:
- Sample source: What subjects/models were used (species, age, strain, sample size)
- Data type: What was measured (neural recordings, behavior, imaging, molecular data)
- Sampling characteristics: Recording methods, temporal/spatial resolution, duration and brain regions
- Task conditions: What behavioral paradigms or experimental manipulations were applied, describe the structure of the behavior task(most important part)
Use natural, flowing language to describe the data foundation of this study.

3. method: Experimental Design and Data Acquisition Methods
Read the Methods, Figure captions, and Results sections and write ONE CONCISE PARAGRAPH  describing HOW the required data were obtained. Include:
- Experimental design: Control vs. experimental groups, within/between-subject design, sample size rationale
- Data acquisition methods: Specific techniques and instruments used to collect each data type (e.g., recording setup, behavioral tracking, imaging parameters)
- Experimental conditions: Manipulated variables, control conditions, timing/sequence of interventions
- Analytical pipeline: Key processing and analysis steps linking raw data to testable predictions
Focus on the DESIGN LOGIC: what methods were chosen to obtain what data, and how the experimental design allows testing the hypothesis.

4. conclusion: Key Findings, Answer, and Significance
Read the Results and Discussion sections and write ONE CONCISE PARAGRAPH that:
- States how these experiments and data analysis results lead to the findings
- Summarizes the logical chain: Data -> Method -> Conclusion -> Scientific findings(most important part)

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
    
    def build_survey_prompt(self, paper_title: str, key_content: str) -> str:
        """构建综述论文的推理链提取prompt"""
        prompt = f"""You are an expert in scientific reasoning and research methodology in neuroscience. Your task is to extract the CORE SCIENTIFIC REASONING CHAIN from this REVIEW/SURVEY paper in a CONCISE, STREAMLINED format.

Paper Title
{paper_title}

Key Content from Paper
{key_content}  

Your Task
Extract the CORE reasoning logic in 4 CONCISE PARAGRAPHS. Focus on the ESSENCE of how the review synthesizes the field, not details.

1. problem_decomposition: Field Problem Landscape and Key Questions
Read the Abstract and Introduction strategically (early → mid → late paragraphs) and write ONE CONCISE PARAGRAPH that captures:
- Early paragraphs: Broad neuroscientific context and overarching domain question
- Middle paragraphs: Key mechanistic unknowns, theoretical debates, or unresolved gaps
- Late paragraphs: Specific sub-questions or frameworks the review is organized around
Focus on the problem landscape of the field, not a single experimental hypothesis.

2. evidence: Empirical Foundation Integrated by the Review
Read early-to-mid Main and figure captions focusing on empirical studies and write ONE CONCISE PARAGRAPH describing the empirical foundation this review integrates:
- Species/systems studied (human, rodent, primate, cell prep, computational models, etc.)
- Key data modalities (electrophysiology, imaging, behavior, genetics, computational modeling, clinical studies)
- Major experimental paradigms and brain regions represented
- Cross-scale linkage (molecular → circuit → systems → behavior)
Emphasize what empirical sources the field uses, not specific datasets from this paper.

3. framework: Knowledge Organization and Synthesis Strategy
Read late Main, figure captions, and Discussion and write ONE CONCISE PARAGRAPH describing how the review organizes and synthesizes knowledge:
- Theoretical or mechanistic frameworks compared or integrated
- Competing/alternative models and how the review reconciles them
- Cross-study synthesis strategy (causal chain, computation-to-circuit mapping, cross-species convergence, etc.)
- Key conceptual diagrams, schemas, or explanatory mechanisms emphasized
Focus on the logic that structures the field and how the review creates coherence across diverse findings.

4. conclusion: Field Consensus, Gaps, and Future Directions
Read the Conclusion / Discussion and write ONE PARAGRAPH that:
- States current consensus or leading mechanistic view(s)
- Summarizes major open questions and controversies
- Identifies methodological or conceptual limitations
- Highlights future research directions or proposed experimental/theoretical roadmaps
Summarize the logical path: evidence → synthesis → field status → what's next

Critical Instructions:
- This is a REVIEW/SURVEY paper, focus on field-level synthesis, not single experiments
- Each field should be ONE CONCISE PARAGRAPH (not lists, not multiple paragraphs)
- Focus on CORE REASONING LOGIC and field-level insights
- Write in natural, flowing language suitable for embedding models
- Output in English

Output Format (JSON):
{{
    "problem_decomposition": "...",
    "evidence": "...",
    "framework": "...",
    "conclusion": "..."
}}

Return ONLY the JSON object with 4 fields, no additional text.
"""
        return prompt
    
    def extract_reasoning_chain(self, paper_content: Dict, metadata: Dict, media_data: List = None, is_review: bool = False) -> Optional[Dict]:
        """
        从单篇论文直接提取推理链
        
        Args:
            paper_content: 论文内容
            metadata: 论文元数据
            media_data: 媒体数据（figures, tables等）
            is_review: 是否使用综述提取模式（默认False，使用研究论文模式）
            
        Returns:
            推理链数据（包含research_reasoning）
        """
        paper_title = paper_content.get('title', 'Unknown')
        
        # 提取关键内容（包括media captions）
        key_content = self.extract_key_content(paper_content, media_data, is_review=is_review)
                
        # 显示内容长度信息（用于监控）
        content_length_k = len(key_content) / 1000
        print(f"  📄 内容长度: {content_length_k:.1f}K 字符")
        
        # 根据参数选择提取模式
        if is_review:
            print(f"  📚 使用综述提取模式")
            # 构建综述Prompt
            prompt = self.build_survey_prompt(paper_title, key_content)
            required_fields = ['problem_decomposition', 'evidence', 'framework', 'conclusion']
        else:
            print(f"  🔬 使用研究论文提取模式")
            # 构建研究论文Prompt
            prompt = self.build_reasoning_prompt(paper_title, key_content)
            required_fields = ['problem_decomposition', 'data', 'method', 'conclusion']
        
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
                    "conclusion": reasoning['conclusion']
                }
                
                # 根据论文类型添加不同的字段
                if is_review:
                    # 综述论文：使用 evidence 和 framework
                    result["evidence"] = reasoning['evidence']
                    result["framework"] = reasoning['framework']
                else:
                    # 研究论文：使用 data 和 method
                    result["data"] = reasoning['data']
                    result["method"] = reasoning['method']
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  ✗ 提取失败: {e}")
                    return None
        
        return None
    
    def _match_paper_data(
        self,
        paper: Dict,
        metadata_dict: Dict,
        media_dict: Dict
    ) -> Tuple[Dict, List]:
        """
        匹配论文的元数据和媒体数据
        
        优先使用 id 匹配，如果失败则使用 doi 匹配
        
        Args:
            paper: 论文内容字典
            metadata_dict: 元数据字典（包含 by_id 和 by_doi）
            media_dict: 媒体数据字典（包含 by_id 和 by_doi）
            
        Returns:
            (metadata, media_data) 元组
        """
        paper_id = paper.get('id')
        paper_doi = paper.get('doi')
        
        # 优先通过 id 匹配元数据
        metadata = {}
        if paper_id and 'by_id' in metadata_dict:
            metadata = metadata_dict['by_id'].get(paper_id, {})
        
        # 如果 id 匹配失败，尝试通过 doi 匹配
        if not metadata and paper_doi and 'by_doi' in metadata_dict:
            metadata = metadata_dict['by_doi'].get(paper_doi, {})
        
        # 优先通过 id 匹配媒体数据
        media_data = []
        if paper_id and 'by_id' in media_dict:
            media_data = media_dict['by_id'].get(paper_id, [])
        
        # 如果 id 匹配失败，尝试通过 doi 匹配
        if not media_data and paper_doi and 'by_doi' in media_dict:
            media_data = media_dict['by_doi'].get(paper_doi, [])
        
        return metadata, media_data
    
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
        resume: bool = True,
        is_review: bool = False
    ):
        """
        批量处理论文（支持断点续传）
        
        Args:
            data_dir: 数据目录
            output_file: 输出文件（reasoning_chains.jsonl）
            max_papers: 最多处理多少篇
            resume: 是否启用断点续传（默认True）
            is_review: 是否使用综述提取模式（默认False，使用研究论文模式）
        """
        data_dir = Path(data_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("🧠 提取科研推理链（支持断点续传）")
        print("=" * 80)
        if is_review:
            print("📚 模式: 综述提取模式")
        else:
            print("🔬 模式: 研究论文提取模式")
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
            media_dict = {'by_id': {}, 'by_doi': {}}
        
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
                
                # 使用双重匹配获取元数据和媒体数据
                metadata, media_data = self._match_paper_data(
                    paper, metadata_dict, media_dict
                )
                
                print(f"\n处理: {paper_title[:60]}...")
                if not metadata:
                    print(f"  ⚠ 警告: 未找到匹配的元数据 (ID: {paper_id})")
                if media_data:
                    print(f"  找到 {len(media_data)} 个图表")
                
                # 提取推理链
                result = self.extract_reasoning_chain(paper, metadata, media_data, is_review=is_review)
                
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
        default="./data/annual_review",
        help="数据目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/annual_review/reasoning_chains.jsonl",
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
        default=config.EXTRACTION_MODEL,
        help="模型名称"
    )
    parser.add_argument(
        "--is-review",
        action="store_true",
        help="使用综述提取模式（默认False，使用研究论文模式）"
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
        max_papers=max_papers,
        is_review=args.is_review
    )

if __name__ == "__main__":
    main()
