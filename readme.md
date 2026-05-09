项目开始

# 针对 Python3

如果是其他版本可以自行 AI 搜索对应的命令

```bash
python3 --version

# openai 库
python3 -m pip install -U openai

# python-dotenv 库
python3 -m pip install python-dotenv

# pandas 库 用于表的处理
python3 -m pip install -q pandas
```

```bash
# 升级 pip 版本
python3 -m pip install --upgrade pip
```

# 运行 main.py 文件

```bash
python3 main.py
```

# 运行 demo.ipynb 文件

你需要在本机执行一次：

```bash
/usr/bin/python3 -m pip install ipykernel -U --user
/usr/bin/python3 -m ipykernel install --user --name python3 --display-name "Python 3"
```

然后选择你安装的内核比如：Python 3.9.6

然后，点击全部运行

然后单独修改某个块中内容之后再点击单个左侧的运行。

# .ipynb 和 .py 文件的区别

| 特性     | `.ipynb` 文件 (Jupyter Notebook)             | `.py` 文件 (Python 脚本)           |
| :------- | :------------------------------------------- | :--------------------------------- |
| 文件格式 | JSON 格式，包含代码、输出、富文本            | 纯文本，仅包含代码和注释           |
| 执行方式 | 交互式，可逐块运行代码并即时查看结果         | 整体运行，一次性执行整个脚本       |
| 适用场景 | 数据探索、教学、原型设计、实验报告           | 项目开发、生产环境部署、自动化脚本 |
| 形象比喻 | 草稿本：可以随时写、随时算、随时改，记录过程 | 正式作文：从头写到尾，一次性提交   |

两者可以相互转换，转换方法如下👇

# python3 将 demo.ipynb 文件转换为 py 文件怎么操作

在项目目录下执行：

```bash
# 进入到项目目录下
cd /path/to/prompt-demo

# 1）安装转换工具（只需一次）
python3 -m pip install nbconvert

# 2）推荐：用模块方式调用 nbconvert（不依赖 jupyter 子命令是否在 PATH 里）
python3 -m nbconvert --to script demo.ipynb
```

转换成功后，同目录会出现 **`demo.py`**（Markdown 单元会变成三引号字符串注释）。

若执行 `python3 -m jupyter nbconvert ...` 时出现 **`Jupyter command jupyter-nbconvert not found`**，多半是本机只装了 `jupyter` 核心包、没有把 `nbconvert` 装进可被识别的环境，或脚本目录未加入 PATH。解决办法仍然是：**先 `pip install nbconvert`，再用上面的 `python3 -m nbconvert --to script demo.ipynb`**，一般即可绕过该问题。

转换得到的脚本若直接 `python3 demo.py`，会按顺序执行笔记本里所有代码单元（含多次调用 API），请注意用量与费用。

**注意：** `demo.ipynb` 里包含多个示例单元；转成 `demo.py` 后也会全部出现在一个文件里，可按需删减或注释不需要的部分再运行。

# 运行 demo.py 文件

```bash
python3 demo.py
```

# .env 和 load_dotenv() 和 os.getenv() 的关系

## 比喻

.env 相当于集装箱，负责存放一些敏感信息或者环境配置

load_dotenv() 相当于搬运工，负责将 .env 中的东西搬运到你的仓库中

os.getenv() 相当于取货员，负责从仓库中拿到具体的货物，比如：DASHSCOPE_API_KEY

## 来源

load_dotenv() 是第三方库 python-dotenv 提供的，专门用来解析文本文件。

os.getenv() 是 Python 标准库 os 提供的。

原因是 Python 代码在设计时，通常只依赖标准库来读取配置（os.getenv）。这样，无论你的配置是来自 .env 文件、Docker 容器、还是云服务器的后台设置，你的代码都不需要修改，只需要更换“搬运工”即可。

# pandas 处理表 VS 大模型处理表 VS function calling 处理表

pandas：计算准确，一次搞定，精确求和、均值、最大值，对表格做复杂筛选或连接

大模型处理表：用自然语言，无需写代码逻辑，模型擅长文字概括，把数据“总结”成一段话

## function calling 处理表

流程如下：

1、你给模型预定义一组函数（比如 get_sales_summary(by='drink')），告诉模型这些函数能做什么。

2、用户问自然语言问题，比如“按饮品汇总销量，然后自然地说出来”。

3、大模型自己决定该不该调用函数、调用哪个函数、传什么参数。

4、模型返回的不是最终答案，而是一个函数调用指令（例如 get_sales_summary(by='drink')）。

5、你的代码收到这个指令后，实际去执行 pandas 计算（或查数据库、调API等），然后把精确的计算结果再发回给模型。

6、模型根据结果生成最终的自然语言回答。

整个过程中，模型在主动决策“我需要调用工具来帮忙”，而不仅仅是接收现成数据。

# 如果做成真正的产品，各阶段是谁负责

假设你要开发一个“AI 数据助手”网页应用（用户输入问题，后端调用大模型 + Function Calling，返回回答）。

| 角色                    | 负责内容                               | 具体工作                                                                                                               |
| :---------------------- | :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| 前端工程师（你）        | 用户界面（UI）与交互逻辑               | 设计聊天界面，收集用户输入，调用后端 API，展示返回的结果                                                               |
| 后端工程师              | 业务逻辑、大模型调用、Function Calling | 接收前端请求，调用大模型 API，根据模型返回的 `tool_calls` 执行具体函数（查数据库、调用内部服务），将最终结果返回给前端 |
| AI/算法工程师（或后端） | 设计 tools 定义、prompt 优化、模型选型 | 决定定义哪些函数（比如 `get_sales_summary`），写好函数描述，调优提示词                                                 |
