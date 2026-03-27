import os
import sys
from dataclasses import dataclass


def _print_env_info() -> None:
    print("executable:", sys.executable)
    print("prefix:", sys.prefix)
    print("paths sample:", sys.path[:5])


# ---- Planning Agent ----
from pydantic import BaseModel, Field


class WebSearchItem(BaseModel):
    """网页搜索项（单条搜索指令）"""

    query: str = Field(..., description="用于网页搜索的关键词，只写一个关键词")
    reason: str = Field(..., description="该搜索项对回答原问题的必要性说明")


class WebSearchPlan(BaseModel):
    """网页搜索计划（包含多条搜索指令）"""

    searches: list[WebSearchItem] = Field(
        ..., description="为精准回答原问题需执行的网页搜索列表"
    )


class SynthesizeData(BaseModel):
    short_summary: str
    """研究结果的简短总结，200 字以内。"""

    markdown_report: str
    """最终报告"""

    follow_up_questions: list[str]
    """建议进一步研究的主题"""


from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_community.chat_models import ChatTongyi

# 你需要在运行前确保 model 变量可用（例如 DeepSeek Chat 模型）
# Notebook 里通常在更早的单元格定义了 model/api_key/base_url。
# 这里提供两种方式：
# 1) 直接在此处初始化你的主模型；
# 2) 从环境变量读取 key。

try:
    from langchain_deepseek import ChatDeepSeek

    _DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    _DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    if not _DEEPSEEK_API_KEY:
        raise RuntimeError(
            "缺少环境变量 DEEPSEEK_API_KEY（脚本需要主模型 `model`）。"
        )

    model = ChatDeepSeek(model="deepseek-chat", api_key=_DEEPSEEK_API_KEY, base_url=_DEEPSEEK_BASE_URL)
except Exception as exc:
    raise RuntimeError(
        "未能初始化主模型 `model`。请设置 DEEPSEEK_API_KEY（以及可选 DEEPSEEK_BASE_URL），"
        "或按你的实际情况在脚本里初始化 `model`。"
    ) from exc


planning_system_prompt = '''
你是专业研究助手，需基于用户查询完成以下核心任务，确保搜索精准高效：
1. 设计 10 个不重复的网页搜索关键词，关键词需贴合查询核心需求，兼顾精准度与覆盖性；
2. 为每个关键词单独说明搜索原因，说明该关键词如何助力解答用户查询，原因需具体可落地，不笼统。
'''

planning_agent = create_agent(
    model=model,
    system_prompt=planning_system_prompt,
    response_format=WebSearchPlan,
)


# ---- Execute Agent ----
# 建议把 key 通过环境变量提供，避免把 key 写进代码
_DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not _DASHSCOPE_API_KEY:
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY（用于通义千问联网搜索模型）。")

qwen_search_model = ChatTongyi(
    model="qwen-max",
    api_key=_DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_kwargs={"enable_search": True},
)

execute_prompt = '''
你是一名专业的研究助理。给定一个搜索关键词后，需通过网络搜索该关键词，并生成简洁的搜索结果摘要。
摘要需包含 2-3 个段落，字数控制在 500 字以内，需涵盖核心要点。
表述需简洁凝练，无需使用完整句子或注重语法规范。
该摘要将供他人整合报告使用，因此务必提炼核心信息、剔除无关内容。除摘要本身外，不得添加任何额外评论。
'''

execute_agent = create_agent(
    system_prompt=execute_prompt,
    model=qwen_search_model,
)


# ---- Synthesize Agent ----
synthesize_prompt = '''
你是一名资深研究员，负责为一项研究查询撰写结构连贯的报告。
你将收到原始查询需求以及研究助理完成的初步研究成果。
请首先制定报告大纲，明确报告的结构框架与逻辑脉络。
随后基于大纲撰写完整报告，并将其作为最终输出。
内容有有结构有段落，最底层的结构不要用1. 或者- 分点描述，直接用文字描述。
最终报告需采用 Markdown 格式，内容需详尽深入。
字数不少于 2000 字。
'''

summary_prompt = '''
你是一名资深研究员，负责为一项研究查询撰写结构连贯的报告。
目前有一份报告，你需要在不要过多的删减的情况下从现有报告中精确提炼重点。
'''

synthesize_agent = create_agent(
    system_prompt=synthesize_prompt,
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 20000),
            summary_prompt=summary_prompt,
        )
    ],
    response_format=SynthesizeData,
)


def clean_markdown_fence(text: str) -> str:
    return text.strip("```").strip("markdown").strip()


class DeepResearchManager:

    def __init__(self) -> None:
        print("初始化已完成，欢迎使用。\n使用前请确认相关模型能够被顺利调用。")

    def run(self, query: str) -> None:
        """执行一次完整的深度调研。

        参数：
        - query：用户的研究问题/主题。

        输出：
        - 控制台打印报告与 follow-up 问题
        - 在当前工作目录保存 Markdown 文件
        """
        print("Starting research...")

        search_plan_response = self.plan_searches(query)
        search_results = self.perform_searches(search_plan_response)
        markdown_content, follow_up_questions = self.write_report(query, search_results)

        print("\n\n=====REPORT=====\n\n")
        print(markdown_content)

        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        print("\n".join(follow_up_questions))

        self.save_report_as_md(query, markdown_content)

    def plan_searches(self, query: str):
        """规划搜索计划。

        这里调用 planning_agent，让模型把 query 拆解成若干条“可执行搜索词”。
        返回值是 LangChain/LangGraph 的 invoke 原始响应（dict），其中：
        - response["structured_response"] 是 WebSearchPlan（由 response_format 约束得到）
        """
        print("Planning searches...")
        response = planning_agent.invoke({"messages": [{"role": "user", "content": query}]})
        print("Search plan structured response:", response["structured_response"])
        return response

    def perform_searches(self, search_plan_response) -> list[str]:
        """根据搜索计划逐条执行联网搜索，并收集摘要。

        参数：
        - search_plan_response：plan_searches 的返回结果（invoke 原始响应 dict）

        过程：
        - 从 structured_response 取出 WebSearchPlan
        - 遍历每条 query，把 query 作为用户消息喂给 execute_agent
        - 从 execute_agent 的最终消息中取出 content 作为该条搜索的摘要

        返回：
        - results：每个搜索词对应的一段摘要文本（list[str]）
        """
        print("Starting searching...")
        num_completed = 0

        plan: WebSearchPlan = search_plan_response["structured_response"]
        tasks = [item.query for item in plan.searches]

        results: list[str] = []
        for task in tasks:
            search_agent_res = execute_agent.invoke(
                {"messages": [{"role": "user", "content": task}]}
            )
            # 约定：最后一条 message 是模型最终输出（摘要文本）
            content = search_agent_res["messages"][-1].content
            if content:
                results.append(content)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")

        return results

    def write_report(self, query: str, search_results: list[str]) -> tuple[str, list[str]]:
        """综合所有搜索结果，生成最终 Markdown 报告与后续研究问题。

        参数：
        - query：用户研究问题
        - search_results：perform_searches 收集到的多条摘要

        过程：
        - 把“用户需求 + 汇总摘要”拼成一个长提示词
        - 调用 synthesize_agent，让模型产出结构化结果 SynthesizeData
        - 对 markdown_report 做轻量清洗：去掉可能的 ```markdown 代码围栏

        返回：
        - markdown_content：最终 Markdown 报告正文
        - follow_up：建议进一步研究的问题列表
        """
        print("Thinking about report...")
        write_report_query = f"用户的需求是：{query}\n汇总的搜索结果如下: {search_results}"

        synthesize_agent_response = synthesize_agent.invoke(
            {"messages": [{"role": "user", "content": write_report_query}]}
        )

        report = synthesize_agent_response["structured_response"].markdown_report
        markdown_content = clean_markdown_fence(report)
        follow_up = synthesize_agent_response["structured_response"].follow_up_questions
        return markdown_content, follow_up

    def save_report_as_md(self, query: str, markdown_content: str) -> None:
        """把报告保存为 Markdown 文件。

        文件名策略：用 query 作为文件名的一部分，并做简单清洗以避免非法字符。
        保存路径：当前工作目录（os.getcwd()）。
        """
        sanitized_query = (
            query.replace(" ", "_")
            .replace("：", "")
            .replace(":", "")
            .replace("?", "")
            .replace("？", "")
        )

        file_name = f"关于{sanitized_query}调研报告.md"
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        print(f"Report saved as: {file_path}")


def main() -> None:
    _print_env_info()

    manager = DeepResearchManager()
    test_query = "AI在教育中的应用"
    manager.run(test_query)


if __name__ == "__main__":
    main()
