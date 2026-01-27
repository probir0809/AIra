from langchain.chains import LLMChain
from aira.core.prompt_manager import PromptManager
import re


class BasicChain:
    """
    a simple LangChain wrapper that connects:
    promptTemplate → LLM → Response
    """

    def __init__(self, llm):
        self.prompt = PromptManager.get_basic_chat_prompt()
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            # verbose=True   # for debugging
        )

    def clean(self,text: str) -> str:

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        tokens_to_remove = [
            "<|assistant|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>"
        ]

        for token in tokens_to_remove:
            text = text.replace(token, "")

        return text.strip()

    def run(self, question: str) -> str:
        """ Executes the chain with a user question"""
        return self.clean(self.chain.run(question=question))
    