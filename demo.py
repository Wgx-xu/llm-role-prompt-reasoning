import gradio as gr
import time
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# ============ 配置区 ============
API_KEY = "sk-5f27f4c1918f4a84b1022a44edd6a230"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
TEMPERATURE = 0.0

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 角色定义
CONDITIONS = {
    "baseline": ("无角色 (Baseline)", ""),
    "mathematician": ("数学家", "你是一个数学家。"),
    "efficient_math": (
    "高效数学家", "你是一个极其注重效率的数学家。请用最精简的步骤推导，不需要任何多余的解释或过渡句，直接给出结论。"),
    "careless_student": ("粗心学生", "你是一个经常粗心大意的学生。")
}

USER_TEMPLATE = (
    "{prefix}请逐步推理以下数学题，"
    "最后一行以 #### 答案 格式输出最终数值。\n"
    "题目：{question}\n解答："
)


# ============ 核心修改：更聪明的答案提取器 ============
def extract_answer(text):
    if not text:
        return "未提取到"

    # 1. 优先匹配严格格式： #### 38
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m: return m.group(1).strip()

    # 2. 匹配常见中文引导词： "答案 38", "答案：38", "原数为 38"
    m = re.search(r"(?:答案|原数|结果)[:：\s]*([-+]?\d+(?:\.\d+)?)", text)
    if m: return m.group(1).strip()

    # 3. 兜底策略：如果最后一行只有一个纯数字（如截图中的 "38"）
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # 检查最后一行是不是一个纯粹的数字
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)", last_line)
        if m: return m.group(1).strip()

    return "未提取到，请人工核对"


def call_single_api(cond_name, prefix, question):
    """请求单个角色的 API"""
    prompt = USER_TEMPLATE.format(prefix=prefix, question=question)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        latency = time.time() - t0
        content = resp.choices[0].message.content
        tokens = resp.usage.completion_tokens if resp.usage else 0
        pred = extract_answer(content)

        # 如果未提取到，为了好看，可以标红提示
        pred_display = f"`{pred}`" if pred != "未提取到，请人工核对" else f"<span style='color:red'>`{pred}`</span>"

        # 构建 Markdown 格式的展示结果
        display_text = f"""**提取答案：** {pred_display}
**耗时：** `{latency:.2f} 秒`
**消耗 Token：** `{tokens}`
---
**完整推理过程：**
{content}"""
        return cond_name, display_text

    except Exception as e:
        latency = time.time() - t0
        return cond_name, f"❌ API 请求失败: {e}"


def analyze_question(question):
    """并发请求四个条件，返回结果"""
    if not question.strip():
        return [("请输入题目...")] * 4

    results = [None] * 4
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(call_single_api, cond, prefix, question): i
            for i, (cond, (label, prefix)) in enumerate(CONDITIONS.items())
        }

        for future in future_to_idx:
            idx = future_to_idx[future]
            _, display_text = future.result()
            results[idx] = display_text

    return results


# ============ 构建 Gradio 界面 ============
with gr.Blocks(title="角色扮演对LLM推理效率的影响") as demo:
    gr.Markdown(
        """
        # 🧠 大模型角色扮演提示词对推理效率的影响
        ### 实验核心：对比四种角色设定下的 **答案准确度**、**推理耗时** 与 **Token消耗**。
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="输入数学题目",
                placeholder="例如：一辆车每小时行驶60公里，3小时行驶多少公里？",
                lines=3,
                value="一个两位数，十位数字与个位数字之和为11，若将两个数字对调，所得新数比原数大45，求原数。"
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("🚀 开始推理", variant="primary", scale=2)

    gr.Markdown("---")
    gr.Markdown("### 📊 四种角色推理结果对比")

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### 1. {CONDITIONS['baseline'][0]}")
            out_base = gr.Markdown("等待输入...")
        with gr.Column():
            gr.Markdown(f"### 2. {CONDITIONS['mathematician'][0]}")
            out_math = gr.Markdown("等待输入...")
        with gr.Column():
            gr.Markdown(f"### 3. {CONDITIONS['efficient_math'][0]}")
            out_eff_math = gr.Markdown("等待输入...")
        with gr.Column():
            gr.Markdown(f"### 4. {CONDITIONS['careless_student'][0]}")
            out_careless = gr.Markdown("等待输入...")

    submit_btn.click(
        fn=analyze_question,
        inputs=[question_input],
        outputs=[out_base, out_math, out_eff_math, out_careless]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)