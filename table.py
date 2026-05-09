#!/usr/bin/env python
# coding: utf-8

# # 处理表格（pandas vs 大模型 vs function calling）
# 
# 运行前请在项目根目录配置 `.env`：`DASHSCOPE_API_KEY=你的密钥`

# In[9]:


import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 是一个在 python 中非常常用的函数，出自第三方库 python-dotenv，用于把 .env 文件的内容注入进来
load_dotenv()

# OpenAI 是提供了一个协议，通过 OpenAI 与大模型服务建立连接
client = OpenAI(
    #  os.getenv 负责从系统环境中提取需要的值
    api_key=os.getenv('DASHSCOPE_API_KEY'),  # 自动获取
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 自定义封装的一个辅助函数，用于简化调用大模型 LLM 生成回答的过程
def generate_responses(prompt, model="qwen-plus"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7, # 控制创造性
        # max_tokens=128, # 限制回复长度
    )

    return response.choices[0].message.content


# ## 1. pandas处理表：`import pandas as pd`
# 
# 下面演示把库导入为别名 **`pd`** 后，如何造表、汇总和筛选（与调用大模型无关，属于常用数据分析写法）。

# In[10]:


# 造一张「本周饮品销量」小表
df = pd.DataFrame(
    {
        "饮品": ["美式", "拿铁", "橙汁", "美式"],
        "销量_杯": [30, 18, 12, 12],
        "门店": ["A店", "A店", "B店", "B店"],
    }
)

print("原始表：")
print(df)

print("\n按饮品汇总销量：")
print(df.groupby("饮品", as_index=False)["销量_杯"].sum())

print("\n销量冠军（饮品名）：")
print(df.groupby("饮品")["销量_杯"].sum().idxmax())


# # 2. 用大模型问答方式分析表格
# 本案例依赖的是大模型本身的语言+推理能力（属于零样本/少样本学习）。Function calling 是另一种设计模式，让模型去调用外部精确计算工具，适合对准确性要求高的场景
# 
# ## 默认的输出格式如何修改？
# 
# 1. 添加输出答案要求 
# 2. 添加示例格式

# In[11]:


# 造一张「本周饮品销量」小表
df = pd.DataFrame({
    "饮品": ["美式", "拿铁", "橙汁", "美式"],
    "销量_杯": [30, 18, 12, 12],
    "门店": ["A店", "A店", "B店", "B店"],
})

# 把表格转成文本形式，方便模型“看到”
table_text = df.to_string(index=False)

print("原始表：")
print(table_text)

# 问题1：按饮品汇总销量
query1 = "按饮品汇总销量（即每种饮品的总销量），请输出一个列表。"
prompt1 = f"""
根据下面的表格数据回答问题。只输出答案，不输出无关内容。

输出答案要求：
- 列名是“饮品”和“销量_杯”
- 行号从0开始
- 按销量降序排列
- 不要输出其他注释

示例格式：
   饮品  销量_杯
0  拿铁    18
1  橙汁    12
2  美式    42

请输出：

表格：
{table_text}

问题：{query1}
"""
print( query1)
print(generate_responses(prompt1))


# 问题2：销量最高的饮品是什么？
query2 = "销量最高的饮品是什么？（只输出饮品名）"
prompt2 = f"""
表格：
{table_text}

问题：{query2}
"""

print(query2)
print(generate_responses(prompt2))


# # 3. pandas 精确计算 + 大模型润色成自然语言
# 
# 
# 最佳实践：先用 pandas 算出精确结果，再让大模型帮你润色成自然语言回答
# 
# ❌这不是function calling，整个过程中，模型没有“调用”任何外部工具，它只是“被动接收”了你已经算好的结果。所有的计算都由你手动写在代码里完成。
# 
# 
# ✅这个例子是一种手工拆解的模式：
# 
# 我们先用 Python 代码（pandas）在本地算出精确结果。
# 
# 然后把结果（数字、文本）塞进 prompt 里。
# 
# 最后让大模型只负责润色，把它变成一段自然语言。
# 

# In[12]:


# 原始数据（示例：本周饮品销量）
df = pd.DataFrame({
    "饮品": ["美式", "拿铁", "橙汁", "美式"],
    "销量_杯": [30, 18, 12, 12],
    "门店": ["A店", "A店", "B店", "B店"],
})

print("原始表格：")
print(df)
print("\n" + "="*40 + "\n")

# 5. 用 pandas 做精确计算
# 5.1 按饮品汇总销量
grouped = df.groupby("饮品", as_index=False)["销量_杯"].sum()
# 按销量降序排列，让答案更清晰
grouped_sorted = grouped.sort_values("销量_杯", ascending=False)

# 5.2 找出销量冠军（饮品名）
champion = grouped_sorted.iloc[0]["饮品"]
champion_sales = grouped_sorted.iloc[0]["销量_杯"]

# 6. 把精确结果转成文本，让大模型润色
# 6.1 把按饮品的汇总表转成易读的文本
summary_text = grouped_sorted.to_string(index=False)

prompt = f"""
请把下面这段“按饮品销量汇总”的数据，用一段自然流畅的中文描述出来。
要求：
- 不要改变数据，不要计算，只做润色。
- 把表格里的数据说清楚，可以提到销量冠军和具体数字。

数据：
{summary_text}

请输出最终的中文段落：
"""

response = generate_responses(prompt)

print("📊 精确计算结果（pandas）：")
print(grouped_sorted)
print(f"\n销量冠军：{champion}（{champion_sales}杯）")

print("💬 大模型润色后的自然语言回答：")
print(response)


# # 4.1 Function Calling实战——饮品销量
# 
# ## 总结
# 
# 不用 Function Calling：适合你完全清楚要算什么的场景，代码简单高效。
# 
# 用 Function Calling：适合搭建对话式数据分析助手，用户可以随意提问，模型自动选择合适的函数去执行。
# 
# ## 与你原版的对比
# | 维度 | 原版 | Function Calling 版 |
# | :--- | :--- | :--- |
# | 调用方式 | 你手动调用 `groupby` 并构建 prompt | 模型自己决定调用 `get_sales_summary` |
# | 任务能力 | 只完成“汇总+冠军”两个固定任务 | 用户问“门店销量冠军”“某个饮品销量多少”等，模型自动匹配对应函数 |
# | 实现复杂度 | 无需处理函数调度 | 需要写 `tools` 定义 + 处理 `tool_calls` 循环 |
# | 性能表现 | 运行结果稳定、快 | 多一次网络请求，稍慢，但更灵活 |

# In[ ]:


import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ---------- 定义真实的数据查询函数（后端执行） ----------
# 使用 pandas 创建数据表
df = pd.DataFrame({
    "饮品": ["美式", "拿铁", "橙汁", "美式"],
    "销量_杯": [30, 18, 12, 12],
    "门店": ["A店", "A店", "B店", "B店"],
})

def get_sales_summary(by: str = "饮品"):
    """
    获取销量汇总数据。
    by: 分组依据，目前支持 "饮品" 或 "门店"
    """
    if by == "饮品":
        grouped = df.groupby("饮品")["销量_杯"].sum().reset_index()
        grouped = grouped.sort_values("销量_杯", ascending=False)
        # 转为字典列表，方便模型阅读
        result = grouped.to_dict(orient="records")
    elif by == "门店":
        grouped = df.groupby("门店")["销量_杯"].sum().reset_index()
        grouped = grouped.sort_values("销量_杯", ascending=False)
        result = grouped.to_dict(orient="records")
    else:
        result = {"error": f"不支持的聚合方式: {by}"}
    return json.dumps(result, ensure_ascii=False)

def get_top_sales(by: str = "饮品"):
    """获取销量冠军"""
    if by == "饮品":
        grouped = df.groupby("饮品")["销量_杯"].sum()
        top_item = grouped.idxmax()
        top_value = int(grouped.max())
        return json.dumps({"冠军": top_item, "销量": top_value}, ensure_ascii=False)
    elif by == "门店":
        grouped = df.groupby("门店")["销量_杯"].sum()
        top_item = grouped.idxmax()
        top_value = int(grouped.max())
        return json.dumps({"冠军": top_item, "销量": top_value}, ensure_ascii=False)
    else:
        return json.dumps({"error": f"不支持的冠军查询: {by}"})

# ---------- 定义工具描述（给大模型看的说明书） ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_summary",
            "description": "按饮品或门店获取销量汇总，返回排序后的列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "by": {
                        "type": "string",
                        "enum": ["饮品", "门店"],
                        "description": "分组维度，例如'饮品'或'门店'"
                    }
                },
                "required": ["by"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_sales",
            "description": "获取销量冠军（销量最高的饮品或门店）",
            "parameters": {
                "type": "object",
                "properties": {
                    "by": {
                        "type": "string",
                        "enum": ["饮品", "门店"],
                        "description": "冠军类型，例如'饮品'或'门店'"
                    }
                },
                "required": ["by"]
            }
        }
    }
]

# ---------- 主对话循环 ----------
def ask_with_function_calling(user_question):
    messages = [{"role": "user", "content": user_question}]
    
    # 第一次调用：让模型判断是否需要调用函数
    first_response = client.chat.completions.create(
        model="qwen-plus",  # 或 qwen-turbo
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = first_response.choices[0].message
    messages.append(response_message)
    
    # 模型要求调用函数吗？
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"🔧 模型调用: {func_name}, 参数: {func_args}")
            
            # 执行真实的 Python 函数
            if func_name == "get_sales_summary":
                result = get_sales_summary(**func_args)
            elif func_name == "get_top_sales":
                result = get_top_sales(**func_args)
            else:
                result = json.dumps({"error": "未知函数"})
            
            # 将工具结果加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        
        # 第二次调用：模型根据工具结果生成最终回答
        second_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
        )
        final_answer = second_response.choices[0].message.content
        return final_answer
    else:
        # 没有工具调用，直接返回模型回答
        return response_message.content

# ---------- 运行示例 ----------
if __name__ == "__main__":
    # 问题1：按饮品汇总销量
    q1 = "按饮品汇总销量，并告诉我销量最高的饮品是什么"
    print(f"🙋 用户: {q1}")
    ans1 = ask_with_function_calling(q1)
    print(f"🤖 AI: {ans1}\n")
    
    # 问题2：销量冠军门店
    q2 = "哪个门店的销量最高？"
    print(f"🙋 用户: {q2}")
    ans2 = ask_with_function_calling(q2)
    print(f"🤖 AI: {ans2}\n")


# # 4.2 Function Calling实战——天气
# 
# Function calling（也叫工具调用）是一种交互模式，流程如下：
# 
# 1、你给模型预定义一组函数（比如 get_sales_summary(by='drink')），告诉模型这些函数能做什么。
# 
# 2、用户问自然语言问题，比如“按饮品汇总销量，然后自然地说出来”。
# 
# 3、大模型自己决定该不该调用函数、调用哪个函数、传什么参数。
# 
# 4、模型返回的不是最终答案，而是一个函数调用指令（例如 get_sales_summary(by='drink')）。
# 
# 5、你的代码收到这个指令后，实际去执行 pandas 计算（或查数据库、调API等），然后把精确的计算结果再发回给模型。
# 
# 6、模型根据结果生成最终的自然语言回答。
# 
# 整个过程中，模型在主动决策“我需要调用工具来帮忙”，而不仅仅是接收现成数据。

# In[14]:


import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载配置和初始化客户端
load_dotenv()

client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 定义真实的Python函数
def get_current_weather(location: str):
    """一个模拟的天气查询函数，在实际应用中这里应该调用真实的天气API"""
    temperature = 10
    condition = "晴天，微风"
    if '上海' in location:
        temperature = 36
        condition = "多云"
    elif '深圳' in location:
        temperature = 37
        condition = "晴朗"

    # 返回JSON字符串化的结果
    return json.dumps({
        "location": location,
        "temperature": temperature,
        "condition": condition,
        "unit": "摄氏度"
    })

# 3. 向大模型描述这个工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "获取指定城市的实时天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如'大连'或'上海'",
                }
            },
            "required": ["location"],
        },
    },
}]

# 4. 核心逻辑：让大模型自主决定并调用工具
def chat_with_ai(messages, tools):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        # tool_choice="auto", # 'auto'代表让AI自主决定是否调用
    )
    return response

# 初始提问
user_question = "大连的天气怎么样？"
messages = [{"role": "user", "content": user_question}]
print(f"🤔 用户：{user_question}")

# 5. 第一次调用：让AI判断
response = chat_with_ai(messages, tools)
assistant_output = response.choices[0].message

# 将AI的初始回复加入对话记录
messages.append(assistant_output)

# 6. 判断AI是否要求调用工具
if assistant_output.tool_calls is None:
    # 没有要求调用，直接输出AI的回答
    print(f"🤖 AI：{assistant_output.content}")
else:
    # AI要求调用工具
    for tool_call in assistant_output.tool_calls:
        # 解析函数名和参数
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(f"🔧 AI 要求调用工具：{function_name}，参数：{function_args}")

        # 执行真实的函数（你的Python代码）
        if function_name == "get_current_weather":
            function_response = get_current_weather(**function_args)

        # 将工具执行结果添加回对话记录
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": function_response,
        })

    # 7. 第二次调用：将工具结果发给AI，让它组织语言回答
    second_response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    final_answer = second_response.choices[0].message.content
    print(f"🤖 AI (最终回答)：{final_answer}")

