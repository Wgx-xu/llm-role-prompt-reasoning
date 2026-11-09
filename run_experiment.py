import os
import re
import json
import time
import csv
from openai import OpenAI

# ============ 配置区 ============
API_KEY = "sk-5f27f4c1918f4a84b1022a44edd6a230"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
TEMPERATURE = 0.0

DATA_FILES = [
    ("data/simple.json", "simple"),
    ("data/medium.json", "medium"),
    ("data/hard.json",   "hard"),
]

OUTPUT_CSV = "results/raw_results.csv"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============ 核心修改：丰富角色条件 ============
CONDITIONS = {
    "baseline": "",
    "mathematician": "你是一个数学家。",
    "efficient_math": "你是一个极其注重效率的数学家。请用最精简的步骤推导，不需要任何多余的解释或过渡句，直接给出结论。",
    "careless_student": "你是一个经常粗心大意的学生。"
}

USER_TEMPLATE = (
    "{prefix}请逐步推理以下数学题，"
    "最后一行以 #### 答案 格式输出最终数值。\n"
    "题目：{question}\n解答："
)

# ============ 工具函数 ============
def extract_answer(text):
    if not text:
        return None
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def normalize(x):
    if x is None:
        return None
    try:
        v = float(x)
        if v.is_integer():
            return str(int(v))
        return f"{v:.2f}"
    except Exception:
        return str(x).strip()

def call_api(prompt):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    latency = time.time() - t0
    content = resp.choices[0].message.content
    usage = resp.usage
    return content, latency, usage

# ============ 主流程 ============
def load_data():
    all_data = []
    for path, difficulty in DATA_FILES:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            item["difficulty"] = difficulty
        all_data.extend(items)
    return all_data

def run():
    os.makedirs("results", exist_ok=True)
    data = load_data()

    rows = []
    total = len(data) * len(CONDITIONS)
    done = 0

    for item in data:
        for cond, prefix in CONDITIONS.items():
            prompt = USER_TEMPLATE.format(prefix=prefix, question=item["question"])
            try:
                content, latency, usage = call_api(prompt)
            except Exception as e:
                content, latency, usage = f"ERROR: {e}", 0.0, None

            pred = extract_answer(content)
            correct = normalize(pred) == normalize(item["gold"])

            rows.append({
                "id": item["id"],
                "difficulty": item["difficulty"],
                "condition": cond,
                "question": item["question"],
                "gold": item["gold"],
                "pred": pred,
                "correct": int(correct),
                "latency": round(latency, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                "raw_output": content  # 保存原始输出以便核查
            })

            done += 1
            print(f"[{done}/{total}] {item['id']} | {item['difficulty']} | {cond} | correct={correct}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n结果已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    run()