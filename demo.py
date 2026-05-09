#!/usr/bin/env python
# coding: utf-8

# # Prompt 示例（DashScope / OpenAI 兼容模式）
# 
# 运行前请在项目根目录配置 `.env`：`DASHSCOPE_API_KEY=你的密钥`

# In[37]:


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


# ## 1. 定义清楚提示词 prompt

# In[38]:


# 清楚的提示词
prompt = f"""
根据下面的上下文回答问题。保持答案简短且准确。如果不确定答案，请回答“不确定答案”。

Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceutical。\
在那里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小鼠中提取的，\
能够结合到T细胞的表面，并限制它们的细胞杀伤潜力。在1986年，它被批准用于帮助预防肾脏移植后的\
器官排斥，成为首个被允许用于人类的治疗性抗体。

问题：OKT3最初是从什么来源提取的？
"""

# 不清楚的提示词
# prompt = f"""问题：OKT3最初是从什么来源提取的？""" 

response = generate_responses(prompt)
print(response)


# ## 2. 结构化，定义变量用 {}

# In[39]:


instruction = """
根据下面的上下文回答问题。保持答案简短且准确。如果不确定答案，请回答“不确定答案”。
"""

context = """
Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceutical。 \
在那里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小鼠中提取的， \
能够结合到T细胞的表面，并限制它们的细胞杀伤潜力。在1986年，它被批准用于帮助预防肾脏移植后的 \
器官排斥，成为首个被允许用于人类的治疗性抗体。
"""

query = """
OKT3最初是从什么来源提取的？
"""
prompt = f"""
{instruction}

### 上下文
{context}

### 问题:
{query}
"""

response = generate_responses(prompt)
print(response)


# ## 3. 添加输出格式（JSON 输出说明）

# In[40]:


# 添加格式

instruction = """
根据下面的上下文回答问题。保持答案简短且准确。如果不能确定答案，请回答“不确定答案”。

以Json格式输出：
{"[具体问题]":"[答案]"},
"""

context = """
Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceutical。 \
在那里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小鼠中提取的， \
能够结合到T细胞的表面，并限制它们的细胞杀伤潜力。在1986年，它被批准用于帮助预防肾脏移植后的 \
器官排斥，成为首个被允许用于人类的治疗性抗体。
"""

query = """
OKT3最初是从什么来源提取的？
"""

prompt = f"""
{instruction}

### 上下文
{context}

### 问题:
{query}
"""

response = generate_responses(prompt)
print(response)


# ## 4. Few-shot learning 示例
# 
# - **zero-shot learning**：不给任何的 examples
# - **one-shot learning**：只给一个 example
# - **few-shot learning**：多个 examples
# 
# 参考：<https://www.promptingguide.ai/zh>
# 
# ### Zero-shot：情感分类

# In[41]:


prompt = """
将文本分类为中性、负面或正面。
文本：我认为这次假期还可以。
情感：
"""

response = generate_responses(prompt)
print(response)


# ### One-shot：先看一条示范，再判断新句子（只输出正面/中性/负面）

# In[42]:


prompt = """
任务：情感分类。你只输出三个字之一：正面、中性、负面（不要解释）。

示范（告诉你输出长什么样）：
句子：这部电影从头到尾都很拖沓。
输出：负面

现在判断下面这一句：
句子：外卖准时送到，味道也挺满意。
输出：
"""

response = generate_responses(prompt)
print(response)


# ### Few-shot：虚构词与例句（续写）

# In[43]:


prompt = """
“whatpu”是坦桑尼亚的一种小型毛茸茸的动物。一个使用whatpu这个词的句子的例子是：
我们在非洲旅行时看到了这些非常可爱的whatpus。
“farduddle”是指快速跳上跳下。一个使用farduddle这个词的句子的例子是：
"""

response = generate_responses(prompt)
print(response)

