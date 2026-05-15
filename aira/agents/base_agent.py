# agents/base_agent.py

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from loguru import logger

from aira.tools.rag_tool import RAGTool
from aira.tools.search import SearchTool
from aira.tools.calculator import CalculatorTool
from aira.core.dependencies import get_retriever


# ── ReAct prompt ────────────────────────────────────────────────────────────
# The agent reads this on every step to decide what to do next.
# Must contain: {tools}, {tool_names}, {input}, {agent_scratchpad}
# These are filled in automatically by LangChain at runtime.

REACT_PROMPT = PromptTemplate.from_template("""
You are AIra, an intelligent AI assistant with access to tools.
Use the tools below to answer the user's question accurately.

Always follow this format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Rules:
- Always use document_search BEFORE web_search for domain-specific questions
- Use calculator for any math or numeric computation
- Use web_search only for current events or general knowledge not in documents
- Never make up answers — use tools to find them
- Keep Final Answer concise and direct

Available tools:
{tools}

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


def build_tools(retriever) -> list:
    """
    Instantiates all three tools and wraps them as LangChain Tool objects.
    LangChain Tool requires: name, func, description.
    """
    rag = RAGTool(retriever)
    search = SearchTool(max_results=5)
    calculator = CalculatorTool()

    tools = [
        Tool(
            name=rag.name,                  # "document_search"
            func=rag.run,
            description=rag.description,    # dynamic — reads kb_metadata
        ),
        Tool(
            name=search.name,               # "web_search"
            func=search.run,
            description=search.description,
        ),
        Tool(
            name=calculator.name,           # "calculator"
            func=calculator.run,
            description=calculator.description,
        ),
    ]

    logger.info(f"Tools registered: {[t.name for t in tools]}")
    return tools


def build_agent(llm) -> AgentExecutor:
    """
    Builds and returns a LangChain AgentExecutor.

    Flow on each user question:
    1. Agent reads the ReAct prompt + question
    2. LLM writes a Thought and picks a tool (Action)
    3. Tool runs and returns Observation
    4. Repeat until LLM writes Final Answer
    5. AgentExecutor returns answer + all intermediate steps
    """
    retriever = get_retriever()
    tools = build_tools(retriever)

    # create_react_agent wires: llm + tools + prompt → runnable agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,           # prevents infinite loops
        verbose=True,               # logs each Thought/Action/Observation
        handle_parsing_errors=True, # recovers gracefully if LLM output is malformed
        return_intermediate_steps=True,  # exposes tool call trace to api/agent.py
    )

    logger.info("AgentExecutor built successfully")
    return executor