import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量（这会自动读取 .env 文件里的配置）
load_dotenv()

# 2. 初始化 OpenAI 客户端，连接到阿里云 DashScope 服务
client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),  # 自动获取
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 固定地址
)

# 3. 封装一个调用函数
def ask_qwen(prompt, model="qwen-plus"):
    """向指定的Qwen模型发送问题，并获取答案。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,      # 控制创造性
            max_tokens=128,       # 限制回复长度
        )
        # 返回模型的回复内容
        return response.choices[0].message.content
    except Exception as e:
        print(f"哎呀，调用出错了：{e}")
        return None

# 4. 真正运行的部分
if __name__ == "__main__":
    # 你的问题
    question = "好的学习方法有哪些？"
    print(f"🤔 问：{question}")
    
    # 调用函数
    answer = ask_qwen(question)
    
    # 打印结果
    if answer:
        print(f"🤖 答：{answer}")