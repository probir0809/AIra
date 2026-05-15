# agents/tool_agent.py

from loguru import logger
from typing import Dict, List, Any

from aira.agents.base_agent import build_agent


class ToolAgent:
    """
    Production wrapper around the LangChain AgentExecutor.

    Responsibilities:
    - Initializes the agent with the LLM
    - Exposes a clean .run(question) interface
    - Formats intermediate steps (tool call trace) for the API
    - Handles errors gracefully without crashing the server
    """

    def __init__(self, llm):
        logger.info("Initializing ToolAgent...")
        self.agent = build_agent(llm)
        logger.info("ToolAgent ready")

    def run(self, question: str) -> Dict[str, Any]:
        """
        Runs the agent on a question and returns a structured response.

        Returns:
        {
            "answer": "the final answer string",
            "tool_calls": [
                {
                    "tool": "document_search",
                    "input": "what is RAG",
                    "output": "Retrieved 5 chunks..."
                },
                ...
            ]
        }
        """
        logger.info(f"ToolAgent received question: '{question}'")

        try:
            result = self.agent.invoke({"input": question})

            answer = result.get("output", "I could not generate an answer.")
            intermediate_steps = result.get("intermediate_steps", [])

            tool_calls = self._format_steps(intermediate_steps)

            logger.info(
                f"ToolAgent completed — "
                f"answer length: {len(answer)} chars, "
                f"tool calls made: {len(tool_calls)}"
            )

            return {
                "answer": answer,
                "tool_calls": tool_calls,
            }

        except Exception as e:
            logger.error(f"ToolAgent error: {e}")
            return {
                "answer": f"Agent encountered an error: {str(e)}",
                "tool_calls": [],
            }

    def _format_steps(self, intermediate_steps: List) -> List[Dict[str, str]]:
        """
        Converts LangChain's raw intermediate_steps into a clean list.

        LangChain returns: [(AgentAction, observation_string), ...]
        We return:         [{"tool": ..., "input": ..., "output": ...}, ...]
        """
        tool_calls = []

        for action, observation in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": str(observation),
            })
            logger.debug(
                f"Tool call — "
                f"tool={action.tool} "
                f"input='{str(action.tool_input)[:80]}' "
                f"output='{str(observation)[:80]}'"
            )

        return tool_calls