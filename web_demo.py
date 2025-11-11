# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""灵枢 - 神经科学推理链生成系统 Web Demo"""
import os
# os.environ["GRADIO_TEMP_DIR"] = "/home/cyy/rag/.gradio_tmp"

import asyncio
import tempfile
import json
from datetime import datetime
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

from reasoning_chain_generator import ReasoningChainGenerator
import config


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None,
                        help="Qwen API key (default from config.py)")
    parser.add_argument("--chroma-path", type=str, default="./chroma_db",
                        help="ChromaDB path")
    parser.add_argument("--model", type=str, default="qwen3-max",
                        help="Generation model name")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=7201,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _summarize_text(text: str, max_length: int = 100) -> str:
    """总结文本内容，保留前半部分"""
    if not text or text == 'N/A':
        return 'N/A'
    
    # 如果文本较短，直接返回
    if len(text) <= max_length:
        return text
    
    # 截取前半部分并添加省略号
    return text[:max_length] + "..."

def _generate_full_text_file(result: dict, query: str) -> str:
    """生成包含完整推理链的文本文件"""
    if not result or result.get('status') != 'success':
        return None
    
    reasoning = result.get('reasoning_chain', {})
    if not reasoning:
        return None
    
    # 创建临时文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reasoning_chain_{timestamp}.txt"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # 写入完整内容
    with open(filepath, 'w', encoding='utf-8') as f:
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"研究问题: {query}\n")
        f.write("-" * 80 + "\n\n")
        
        # 1. Problem Decomposition (完整内容)
        f.write("1. Problem Decomposition (完整内容)\n")
        f.write("-" * 80 + "\n")
        f.write(reasoning.get('problem_decomposition', 'N/A') + "\n\n")
        
        # 2. Data Requirements (完整内容)
        f.write("2. Data Requirements (完整内容)\n")
        f.write("-" * 80 + "\n")
        f.write(reasoning.get('data', 'N/A') + "\n\n")
        
        # 3. Experimental Methods (完整内容)
        f.write("3. Experimental Methods (完整内容)\n")
        f.write("-" * 80 + "\n")
        f.write(reasoning.get('method', 'N/A') + "\n\n")
        
        # 4. Conclusion (完整内容)
        f.write("4. Expected Conclusion (完整内容)\n")
        f.write("-" * 80 + "\n")
        f.write(reasoning.get('conclusion', 'N/A') + "\n\n")
        
        # References
        if 'references' in result and result['references']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("参考文献\n")
            f.write("=" * 80 + "\n\n")
            
            for i, ref in enumerate(result['references'][:15], 1):
                authors = ref.get('authors', 'Unknown Authors')
                title = ref.get('title', 'Unknown Title')
                journal = ref.get('journal', 'Unknown Journal')
                year = ref.get('year', 'Unknown Year')
                
                f.write(f"[{i}] {authors}. {title}[J]. {journal}, {year}.\n")
                f.write(f"    (相关性: {ref['similarity']:.3f} | 引用数: {ref['citation_count']})\n\n")
    
    return filepath
    
def _format_reasoning_chain(result: dict) -> str:
    """格式化推理链为 Markdown（使用LLM生成的摘要）"""
    if result.get('status') == 'error':
        return f"❌ **错误**: {result.get('message', '生成失败')}"
    
    # 优先使用LLM生成的摘要
    summary = result.get('summary')
    if summary:
        # 构建格式化输出（使用LLM摘要）
        output = []
        output.append(f"## 🔬 科研推理链（概要）\n")
        
        # 1. Problem Decomposition - LLM摘要
        output.append(f"### 1️⃣ Problem Decomposition")
        output.append(f"{summary.get('problem_decomposition', 'N/A')}\n")
        
        # 2. Data Requirements - LLM摘要
        output.append(f"### 2️⃣ Data Requirements")
        output.append(f"{summary.get('data', 'N/A')}\n")
        
        # 3. Experimental Methods - LLM摘要
        output.append(f"### 3️⃣ Experimental Methods")
        output.append(f"{summary.get('method', 'N/A')}\n")
        
        # 4. Conclusion - LLM摘要
        output.append(f"### 4️⃣ Expected Conclusion")
        output.append(f"{summary.get('conclusion', 'N/A')}\n")
    
    else:
        # 备选方案：使用原始推理链的截断版本
        reasoning = result.get('reasoning_chain')
        if not reasoning:
            return f"⚠️ **无法解析生成结果**\n\n原始输出:\n```\n{result.get('raw_output', 'N/A')}\n```"
        
        # 构建格式化输出（简短版本）
        output = []
        output.append(f"## 🔬 科研推理链（概要）\n")
        
        # 1. Problem Decomposition - 简短总结
        problem = reasoning.get('problem_decomposition', 'N/A')
        output.append(f"### 1️⃣ Problem Decomposition")
        output.append(f"{_summarize_text(problem, max_length=150)}\n")
        
        # 2. Data Requirements - 简短总结
        data = reasoning.get('data', 'N/A')
        output.append(f"### 2️⃣ Data Requirements")
        output.append(f"{_summarize_text(data, max_length=150)}\n")
        
        # 3. Experimental Methods - 简短总结
        method = reasoning.get('method', 'N/A')
        output.append(f"### 3️⃣ Experimental Methods")
        output.append(f"{_summarize_text(method, max_length=150)}\n")
        
        # 4. Conclusion - 简短总结
        conclusion = reasoning.get('conclusion', 'N/A')
        output.append(f"### 4️⃣ Expected Conclusion")
        output.append(f"{_summarize_text(conclusion, max_length=150)}\n")
    
    return "\n".join(output)


def _launch_demo(args, generator):
    """启动 Gradio Demo"""

    def predict(_query, _chatbot, _task_history):
        """处理用户输入，生成推理链"""
        if not _query.strip():
            yield _chatbot, _task_history, None
            return
        
        print(f"User Query: {_query}")
        
        # 添加用户消息并显示加载状态
        _chatbot.append((_query, "🔍 正在进行问题分析与生成...\n"))
        yield _chatbot, _task_history, None
        
        try:
            # 生成推理链
            result = generator.generate_reasoning_chain(
                research_question=_query,
                top_k=15,
                return_references=True
            )
            
            # 处理可能的异步结果
            if asyncio.iscoroutine(result):
                print("DEBUG: Detected coroutine result, running asyncio...")
                result = asyncio.run(result)
            elif hasattr(result, "result"):  # 针对 AsyncRequest
                print("DEBUG: Detected AsyncRequest, resolving...")
                result = result.result()


            # 格式化输出（简短总结）
            formatted_response = _format_reasoning_chain(result)
            
            # 生成完整文本文件
            full_text_file = _generate_full_text_file(result, _query)
            
            print(f"Generation completed successfully")
            print(f"Response length: {len(formatted_response)} chars")
            print("="*80)
            print("生成的推理链内容:")
            print("="*80)
            print(formatted_response)
            print("="*80)
            
            if full_text_file:
                print(f"完整内容文件: {full_text_file}")
                print(f"文件是否存在: {os.path.exists(full_text_file)}")
                if os.path.exists(full_text_file):
                    print(f"文件大小: {os.path.getsize(full_text_file)} bytes")
            else:
                print("⚠️ 无法生成完整文本文件")
            
            # 更新聊天界面
            _chatbot[-1] = (_query, formatted_response)
            _task_history.append((_query, formatted_response))
            
            print(f"Chatbot updated, current length: {len(_chatbot)}")
            print(f"Task history length: {len(_task_history)}")
            
            # 返回更新后的状态和文件（显示下载组件）
            if full_text_file:
                yield _chatbot, _task_history, gr.update(value=full_text_file, visible=True)
            else:
                yield _chatbot, _task_history, gr.update(visible=False)
            
        except Exception as e:
            error_msg = f"❌ **生成失败**: {str(e)}"
            _chatbot[-1] = (_query, error_msg)
            _task_history.append((_query, error_msg))
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            yield _chatbot, _task_history, gr.update(visible=False)

    def regenerate(_chatbot, _task_history):
        """重新生成最后一个回答"""
        if not _task_history:
            yield _chatbot, _task_history, gr.update(visible=False)
            return
        
        # 移除最后一轮对话
        last_query = _task_history[-1][0]
        _task_history.pop(-1)
        _chatbot.pop(-1)
        
        # 重新生成（使用相同的predict函数）
        for result in predict(last_query, _chatbot, _task_history):
            yield result

    def reset_user_input():
        """清空输入框"""
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        """清空对话历史"""
        _task_history.clear()
        _chatbot.clear()
        return [], [], gr.update(visible=False)

    with gr.Blocks(title="灵枢 - 神经科学推理链生成系统") as demo:

        gr.Markdown("""<center><font size=8>🧠 灵枢</font></center>""")
        gr.Markdown(
            """\
<center><font size=3>科学问题生成与实验设计大模型</font></center>
<center><font size=2>基于 CNS 海量文献构建</font></center>""")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='问题生成与实验设计', height=600)
                # 文件下载组件 - 初始隐藏，生成内容后显示
                download_file = gr.File(
                    label="📥 下载完整报告", 
                    visible=False,
                    height=50,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 💡 使用说明")
                gr.Markdown("""
                1. 输入：研究问题
                2. 输出：
                   - 问题生成
                   - 数据需求
                   - 实验方法
                   - 潜在结论

                **示例问题**：
                - How does stress affect hippocampal neurogenesis?
                - What is the role of dopamine in reward learning?
                """)
        
        query = gr.Textbox(
            lines=3, 
            label='输入你的研究问题 (Input Research Question)',
            placeholder="例如: How does chronic stress affect hippocampal neurogenesis?"
        )
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 清除历史 (Clear)")
            submit_btn = gr.Button("🚀 生成推理链 (Generate)", variant="primary")
            regen_btn = gr.Button("🔄 重新生成 (Regenerate)")
        
        # 事件绑定
        submit_btn.click(
            predict, 
            [query, chatbot, task_history], 
            [chatbot, task_history, download_file], 
            show_progress=True,
            queue=True
        )
        submit_btn.click(reset_user_input, [], [query])
        
        # 支持回车提交
        query.submit(
            predict,
            [query, chatbot, task_history],
            [chatbot, task_history, download_file],
            show_progress=True,
            queue=True
        )
        query.submit(reset_user_input, [], [query])
        
        empty_btn.click(
            reset_state, 
            [chatbot, task_history], 
            outputs=[chatbot, task_history, download_file], 
            show_progress=True
        )
        regen_btn.click(
            regenerate, 
            [chatbot, task_history], 
            [chatbot, task_history, download_file], 
            show_progress=True,
            queue=True
        )

        gr.Markdown("""\
<center><font size=2>本系统基于 RAG 技术，结合向量检索与大模型生成 | 数据来源: 海量 CNS 文献</font></center>
<center><font size=2>💡 提示：界面显示为简要总结，完整内容请下载详细报告</font></center>""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    """主函数"""
    args = _get_args()

    # 获取 API key
    api_key = args.api_key or config.OPENAI_API_KEY
    if not api_key:
        raise ValueError("请提供 API key（通过 --api-key 或在 config.py 中配置）")

    print("="*80)
    print("灵枢 - 神经科学推理链生成系统")
    print("="*80)
    
    # 初始化推理链生成器
    try:
        generator = ReasoningChainGenerator(
            api_key=api_key,
            chroma_path=args.chroma_path,
            collection_name="neuroscience",
            generation_model=args.model,
            temperature=0.3
        )
        print("✓ 初始化成功！")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    print("\n正在启动 Web 界面...")
    print(f"  访问地址: http://{args.server_name}:{args.server_port}")
    if args.share:
        print(f"  公开链接: 将在启动后生成")
    print("="*80)
    
    # 启动 Demo
    _launch_demo(args, generator)


if __name__ == '__main__':
    main()
