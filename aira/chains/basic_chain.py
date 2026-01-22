from langchain.chains import LLMChain
from aira.core.prompt_manager import PromptManager


class BasicChain:
    """
    A simple LangChain wrapper that connects:
    PromptTemplate → LLM → Response
    """

    def __init__(self, llm):
        self.prompt = PromptManager.get_basic_chat_prompt()
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=True
        )

    def run(self, question: str) -> str:
        """
        Executes the chain with a user question.
        """
        return self.chain.run(question=question)
