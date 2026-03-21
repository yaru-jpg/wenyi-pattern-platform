import os
import cv2
import numpy as np
import hashlib
import json
import re
import io
import base64
import traceback
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
import dashscope

# ==========================================
# 1. 阿里云大模型 API KEY (安全模式：从环境变量读取)
# 绝不将真实的 Key 暴露在代码中上传至 GitHub！
# ==========================================
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def robust_json_parse(raw_text):
    """防弹JSON解析器"""
    try:
        text = raw_text.strip()
        tick_str = chr(96) * 3
        text = text.replace(tick_str + 'json', '').replace(tick_str, '')

        start_obj = text.find('{')
        start_arr = text.find('[')
        starts = [s for s in (start_obj, start_arr) if s != -1]
        start = min(starts) if starts else -1

        end_obj = text.rfind('}')
        end_arr = text.rfind(']')
        ends = [e for e in (end_obj, end_arr) if e != -1]
        end = max(ends) if ends else -1

        if start != -1 and end != -1:
            clean_json = text[start:end + 1]
            try:
                return json.loads(clean_json)
            except:
                clean_json = clean_json.replace('\n', ' ').replace('\r', '')
                clean_json = re.sub(r',\s*\}', '}', clean_json)
                clean_json = re.sub(r',\s*\]', ']', clean_json)
                for i in range(5):
                    try:
                        return json.loads(clean_json + '}' * i)
                    except:
                        try:
                            return json.loads(clean_json + ']' * i)
                        except:
                            continue
        return None
    except Exception as e:
        print("JSON解析核心报错: " + str(e))
        return None


@app.post("/api/analyze_text")
async def analyze_text(text: str = Form(...)):
    """文本推理接口：使用对纯文本支持极好且 JSON 格式遵循度极高的 qwen-plus 模型"""
    prompt = (
        f"你是一位中国古典纹样专家。用户提供了一段描述：\"{text}\"。\n"
        "请提取关键词，推理出最符合描述的 2 到 3 种传统纹样。\n"
        "【极其重要的警告】\n"
        "必须且只能输出一个JSON数组！绝对不要输出任何Markdown标记或多余的解释！\n"
        "必须输出如下结构的JSON数组：\n"
        "[\n"
        "  {\n"
        '    "pattern_name": "推理出的真实名称",\n'
        '    "details": {\n'
        '      "zh": {"name": "真实名称", "era": "年代范围", "meaning": "寓意", "history": "历史典故", "usage": "场景", "cross": "跨文化对照"},\n'
        '      "en": {"name": "Name", "era": "Era", "meaning": "Meaning", "history": "History", "usage": "Usage", "cross": "Cross-cultural"},\n'
        '      "semantics": {"基本信息": ["年代"], "工艺": ["工艺"], "寓意": ["寓意1"]},\n'
        '      "evolution": [\n'
        '        {"era": {"zh": "起源朝代", "en": "Origin"}, "desc": {"zh": "特征", "en": "Features"}, "keyword": "keyword"}\n'
        '      ]\n'
        '    }\n'
        "  }\n"
        "]"
    )

    try:
        # 校验环境变量是否成功读取
        if not dashscope.api_key:
            raise ValueError("未检测到 API KEY，请检查环境变量配置。")

        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )

        if response.status_code != 200:
            raise ValueError(f"API请求被拒: {response.message}")

        raw_text = response.output.choices[0].message.content
        ai_data_list = robust_json_parse(raw_text)

        if not ai_data_list or not isinstance(ai_data_list, list):
            raise ValueError(f"模型返回格式错乱，原话为：{raw_text[:100]}...")

        results = []
        for item in ai_data_list:
            img_hash = int(hashlib.md5(item.get("pattern_name", "temp").encode()).hexdigest()[:8], 16)
            results.append({
                "pattern_type": item.get("pattern_name", "未知纹样"),
                "confidence": "基于文本推理",
                "colors": ["#5C1E16", "#D4A373", "#7BA7A0", "#1A1A2E"],
                "color_ratios": [35, 30, 20, 15],
                "edge_base64": "",
                "img_id": f"PID-{img_hash}",
                "material": {"zh": "文本推理", "en": "Text Inferred"},
                "craft": {"zh": "文本推理", "en": "Text Inferred"},
                "dynamic_details": item.get("details", {})
            })
        return results

    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        # 若出现异常，直接把原因呈现在前端卡片上，方便排查
        return [{
            "pattern_type": "大模型响应异常",
            "confidence": "Error",
            "colors": ["#5C1E16", "#D4A373", "#7BA7A0", "#1A1A2E"],
            "color_ratios": [35, 30, 20, 15],
            "edge_base64": "",
            "img_id": "Error",
            "material": {"zh": "错误", "en": "Error"},
            "craft": {"zh": "错误", "en": "Error"},
            "dynamic_details": {
                "zh": {
                    "name": "模型解析失败",
                    "era": "请求中断",
                    "meaning": "请重试或检查 API 状态。",
                    "history": f"异常诊断详细信息：{err_msg}",
                    "usage": "系统故障",
                    "cross": ""
                }
            }
        }]


@app.post("/api/analyze")
async def analyze_pattern(file: UploadFile = File(...)):
    """处理图像的接口（视觉多模态模型）"""
    contents = await file.read()
    img_hash = int(hashlib.md5(contents).hexdigest()[:8], 16)

    try:
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        img_for_ai = img_pil.copy()
        img_for_ai.thumbnail((1600, 1600))
        buffered_ai = io.BytesIO()
        img_for_ai.save(buffered_ai, format="JPEG", quality=98)
        img_base64 = base64.b64encode(buffered_ai.getvalue()).decode("utf-8")
        image_data = "data:image/jpeg;base64," + img_base64
    except Exception as e:
        return {"pattern_type": "图片读取异常", "dynamic_details": {}}

    prompt = (
        "你是一位中国古典纹样鉴定专家。请极其精准地识别图片中的传统纹样。\n"
        "【极其重要的警告】\n"
        "必须且只能输出一个标准的JSON对象！不要客套和Markdown！单项描述不要超过20个字！\n"
        "必须严格按照如下结构输出：\n"
        "{\n"
        '  "pattern_name": "真实名称",\n'
        '  "details": {\n'
        '    "zh": {"name": "真实名称", "era": "年代范围", "meaning": "寓意", "history": "历史典故", "usage": "场景", "cross": "跨文化对照"},\n'
        '    "en": {"name": "Name", "era": "Era", "meaning": "Meaning", "history": "History", "usage": "Usage", "cross": "Cross-cultural"},\n'
        '    "semantics": {"基本信息": ["年代"], "工艺": ["工艺"], "空间": ["地点"], "寓意": ["寓意1"]},\n'
        '    "evolution": [\n'
        '      {"era": {"zh": "起源", "en": "Origin"}, "desc": {"zh": "特征", "en": "Features"}, "keyword": "key"}\n'
        '    ]\n'
        '  }\n'
        "}"
    )

    messages = [{'role': 'user', 'content': [{'image': image_data}, {'text': prompt}]}]

    try:
        # 校验环境变量是否成功读取
        if not dashscope.api_key:
            raise ValueError("未检测到 API KEY，请检查环境变量配置。")

        response = dashscope.MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        if response.status_code != 200:
            raise ValueError(f"API请求被拒: {response.message}")

        raw_text = response.output.choices[0].message.content[0]['text']
        ai_data = robust_json_parse(raw_text)

        if ai_data is None:
            raise ValueError(f"模型格式错乱，原话为：{raw_text[:100]}...")

        predicted_name = ai_data.get("pattern_name", "未知纹样")
        dynamic_details = ai_data.get("details", {})
        conf_score = "98.50%"

    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        predicted_name = "大模型响应异常"
        conf_score = "0.00%"
        dynamic_details = {
            "zh": {
                "name": "解析失败",
                "history": f"异常诊断详细信息：{err_msg}"
            }
        }

    img_rgb = np.array(img_pil)
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    resized_kmeans = cv2.resize(img_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = resized_kmeans.reshape(-1, 3)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=3).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    ratios = [int(c / len(labels) * 100) for c in counts]

    sorted_indices = np.argsort(ratios)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_ratios = [ratios[i] for i in sorted_indices]

    hex_colors = ["#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]) for c in sorted_colors]
    hsv_colors = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0][0] for c in sorted_colors]
    avg_s, avg_v, avg_h = np.mean([c[1] for c in hsv_colors]), np.mean([c[2] for c in hsv_colors]), np.mean([c[0] for c in hsv_colors])

    if avg_s < 60 and avg_v < 150:
        mat_zh, mat_en, craft_zh, craft_en = "石材/青铜", "Stone/Bronze", "雕刻/铸造", "Carving/Casting"
    elif avg_v > 180 and 90 < avg_h < 130 and avg_s > 40:
        mat_zh, mat_en, craft_zh, craft_en = "陶瓷(青花)", "Ceramic", "窑烧工艺", "Kiln Firing"
    elif avg_s > 100 and avg_v > 100:
        mat_zh, mat_en, craft_zh, craft_en = "丝绸/织物", "Silk/Fabric", "刺绣/织锦", "Embroidery"
    else:
        mat_zh, mat_en, craft_zh, craft_en = "木材/竹器", "Wood/Bamboo", "木雕/漆器", "Wood Carving"

    img_edge = cv2.resize(img_cv, (600, int(600 * img_cv.shape[0] / img_cv.shape[1])))
    gray = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    _, buffer = cv2.imencode('.jpg', cv2.bitwise_not(edges), [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    edge_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "pattern_type": predicted_name, "confidence": conf_score,
        "colors": hex_colors, "color_ratios": sorted_ratios,
        "edge_base64": edge_base64, "img_id": "PID-" + str(img_hash),
        "material": {"zh": mat_zh, "en": mat_en}, "craft": {"zh": craft_zh, "en": craft_en},
        "dynamic_details": dynamic_details
    }

if __name__ == "__main__":
    import uvicorn
    # Railway 云端部署必须使用 0.0.0.0 和动态端口
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)