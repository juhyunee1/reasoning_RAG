# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""灵枢 - 神经科学推理链生成系统 Web Demo"""
import os
os.environ["GRADIO_TEMP_DIR"] = "/home/cyy/rag/.gradio_tmp"

import asyncio
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


def _format_reasoning_chain(result: dict) -> str:
    """格式化推理链为 Markdown"""
    if result.get('status') == 'error':
        return f"❌ **错误**: {result.get('message', '生成失败')}"
    
    reasoning = result.get('reasoning_chain')
    if not reasoning:
        return f"⚠️ **无法解析生成结果**\n\n原始输出:\n```\n{result.get('raw_output', 'N/A')}\n```"
    
    # 构建格式化输出
    output = []
    output.append(f"## 🔬 科研推理链\n")
    
    # 1. Problem Decomposition
    output.append(f"### 1️⃣ Problem Decomposition")
    output.append(f"{reasoning.get('problem_decomposition', 'N/A')}\n")
    
    # 2. Data Requirements
    output.append(f"### 2️⃣ Data Requirements")
    output.append(f"{reasoning.get('data', 'N/A')}\n")
    
    # 3. Experimental Methods
    output.append(f"### 3️⃣ Experimental Methods")
    output.append(f"{reasoning.get('method', 'N/A')}\n")
    
    # 4. Conclusion
    output.append(f"### 4️⃣ Expected Conclusion")
    output.append(f"{reasoning.get('conclusion', 'N/A')}\n")
    
    # References
    if 'references' in result and result['references']:
        output.append(f"---\n### 📚 参考文献")
        for i, ref in enumerate(result['references'][:3], 1):
            output.append(f"{i}. **{ref['title']}** ({ref['year']})")
            output.append(f"   - 相似度: {ref['similarity']:.3f} | 引用数: {ref['citation_count']}")
    
    return "\n".join(output)


def _launch_demo(args, generator):
    """启动 Gradio Demo"""

    def predict(_query, _chatbot, _task_history):
        """处理用户输入，生成推理链"""
        if not _query.strip():
            return _chatbot, _task_history
        
        print(f"User Query: {_query}")
        
        # 添加用户消息并显示加载状态
        _chatbot.append((_query, "🔍 正在检索相关研究...\n"))
        yield _chatbot, _task_history
        
        try:
            # 生成推理链
            result = generator.generate_reasoning_chain(
                research_question=_query,
                top_k=5,
                return_references=True
            )
            
            # 处理可能的异步结果
            if asyncio.iscoroutine(result):
                print("DEBUG: Detected coroutine result, running asyncio...")
                result = asyncio.run(result)
            elif hasattr(result, "result"):  # 针对 AsyncRequest
                print("DEBUG: Detected AsyncRequest, resolving...")
                result = result.result()


            # 格式化输出
            formatted_response = _format_reasoning_chain(result)
            
            print(f"Generation completed successfully")
            print(f"Response length: {len(formatted_response)} chars")
            print("="*80)
            print("生成的推理链内容:")
            print("="*80)
            print(formatted_response)
            print("="*80)
            
            # 更新聊天界面
            _chatbot[-1] = (_query, formatted_response)
            _task_history.append((_query, formatted_response))
            
            print(f"Chatbot updated, current length: {len(_chatbot)}")
            print(f"Task history length: {len(_task_history)}")
            
        except Exception as e:
            error_msg = f"❌ **生成失败**: {str(e)}"
            _chatbot[-1] = (_query, error_msg)
            _task_history.append((_query, error_msg))
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("DEBUG: predict() reached return point")
        # 返回更新后的状态
        yield _chatbot, _task_history

    def regenerate(_chatbot, _task_history):
        """重新生成最后一个回答"""
        if not _task_history:
            yield _chatbot, _task_history
            return
        
        # 移除最后一轮对话
        last_query = _task_history[-1][0]
        _task_history.pop(-1)
        _chatbot.pop(-1)
        
        # 重新生成
        yield from predict(last_query, _chatbot, _task_history)

    def reset_user_input():
        """清空输入框"""
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        """清空对话历史"""
        _task_history.clear()
        _chatbot.clear()
        return [], []

    with gr.Blocks(title="灵枢 - 神经科学推理链生成系统") as demo:

        gr.Markdown("""<center><font size=8>🧠 灵枢</font></center>""")
        gr.Markdown(
            """\
<center><font size=3>神经科学实验设计推理链生成系统</font></center>
<center><font size=2>基于 Nature Neuroscience 2729 篇论文构建 | RAG + LLM</font></center>""")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='推理链生成', height=600)
            with gr.Column(scale=1):
                gr.Markdown("### 💡 使用说明")
                gr.Markdown("""
                1. **输入研究问题**（英文）
                2. 系统自动检索相关研究
                3. 生成完整推理链：
                   - 问题分解
                   - 数据需求
                   - 实验方法
                   - 预期结论
                
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
            [chatbot, task_history], 
            show_progress=True,
            queue=True
        )
        submit_btn.click(reset_user_input, [], [query])
        
        # 支持回车提交
        query.submit(
            predict,
            [query, chatbot, task_history],
            [chatbot, task_history],
            show_progress=True,
            queue=True
        )
        query.submit(reset_user_input, [], [query])
        
        empty_btn.click(
            reset_state, 
            [chatbot, task_history], 
            outputs=[chatbot, task_history], 
            show_progress=True
        )
        regen_btn.click(
            regenerate, 
            [chatbot, task_history], 
            [chatbot, task_history], 
            show_progress=True
        )

        gr.Markdown("""\
<center><font size=2>本系统基于 RAG 技术，结合向量检索与大模型生成 | 数据来源: Nature Neuroscience</font></center>""")

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
    print(f"正在初始化...")
    print(f"  数据库路径: {args.chroma_path}")
    print(f"  生成模型: {args.model}")
    
    # 初始化推理链生成器
    try:
        generator = ReasoningChainGenerator(
            api_key=api_key,
            chroma_path=args.chroma_path,
            collection_name="neuroscience",
            generation_model=args.model,
            temperature=0.7
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
