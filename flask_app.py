"""
高校心理危机预警与社工介入智能体 - Flask应用

提供完整的全屏视觉反馈系统
"""

import os
import json
import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from coze_coding_utils.runtime_ctx.context import new_context
from coze_coding_dev_sdk import SearchClient

app = Flask(__name__)
CORS(app)

# 配置
CONFIG_PATH = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
LLM_CONFIG_FILE = f"{CONFIG_PATH}/config/agent_llm_config.json"


def load_config():
    """加载LLM配置"""
    with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_llm():
    """获取LLM实例"""
    cfg = load_config()
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    return ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg['config'].get('temperature', 0.7),
        streaming=True,
        timeout=cfg['config'].get('timeout', 600),
    )


def detect_mental_state(text: str) -> dict:
    """智能检测心理状态"""
    text_lower = text.lower()

    critical_keywords = [
        "自杀", "自残", "跳楼", "割腕", "不想活了", "活着没意思",
        "活着太累", "死了一了百了", "轻生", "了结自己",
        "已经计划", "准备", "药", "工具", "绝望到", "无法承受",
        "没有活下去", "结束生命", "死亡", "了结", "离开这个世界"
    ]

    sub_healthy_keywords = [
        "难过", "伤心", "压力大", "焦虑", "失眠", "睡不着",
        "迷茫", "困惑", "无助", "孤独", "失落", "沮丧",
        "抑郁", "悲观", "绝望", "没有动力", "疲惫",
        "烦躁", "不安", "紧张", "害怕", "恐惧", "担心", "压抑",
        "没意思", "没希望", "累", "痛苦", "撑不住", "扛不住"
    ]

    critical_count = sum(1 for kw in critical_keywords if kw in text_lower)
    sub_healthy_count = sum(1 for kw in sub_healthy_keywords if kw in text_lower)

    if critical_count >= 1:
        return {"state": "critical", "risk_level": 4, "confidence": 0.95, "keywords": [kw for kw in critical_keywords if kw in text_lower]}
    elif sub_healthy_count >= 3:
        return {"state": "sub_healthy", "risk_level": 2, "confidence": 0.8, "keywords": [kw for kw in sub_healthy_keywords if kw in text_lower]}
    elif sub_healthy_count >= 1:
        return {"state": "sub_healthy", "risk_level": 1, "confidence": 0.6, "keywords": [kw for kw in sub_healthy_keywords if kw in text_lower]}
    else:
        return {"state": "healthy", "risk_level": 0, "confidence": 0.85, "keywords": []}


def get_personalized_message(state: str) -> str:
    """根据状态返回个性化消息"""
    messages = {
        "critical": "💔 我听到了你的痛苦\n\n无论你现在正在经历什么，请记住：\n• 你并不孤单\n• 痛苦是暂时的\n• 求助是勇敢的表现\n\n📞 请立即拨打：400-161-9995\n如紧急请拨打 110 或 120\n\n你的生命比任何事情都重要。💙",
        "sub_healthy": "🌼 我在这里陪伴你\n\n我能感受到你正在经历困难的日子。\n• 这完全正常\n• 你已经迈出了重要的一步\n• 让我们一起度过这个难关\n\n📞 需要帮助请拨打：400-161-9995\n\n求助是勇敢的表现，不是软弱。💛",
        "healthy": "🌟 你做得很好！\n\n能够保持心理健康是非常了不起的事情！\n• 继续保持规律作息\n• 坚持运动\n• 与朋友保持联系\n\n心理健康和身体健康同样重要，继续加油！💚"
    }
    return messages.get(state, messages["healthy"])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    mental_state = detect_mental_state(user_message)
    personal_msg = get_personalized_message(mental_state["state"])

    try:
        llm = get_llm()
        cfg = load_config()
        system_prompt = cfg.get("sp", "")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content

        full_response = f"{personal_msg}\n\n---\n\n{full_response}"

        return jsonify({
            'response': full_response,
            'mental_state': mental_state
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'mental_state': mental_state,
            'response': f"{personal_msg}\n\n系统遇到问题，请拨打热线：400-161-9995"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
