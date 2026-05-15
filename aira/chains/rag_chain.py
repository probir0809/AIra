# rag_chain.py

from langchain.chains import LLMChain
from aira.core.prompt_manager import PromptManager
from typing import List
from langchain.schema import Document
import re

class RAGChain:
    """
    Retrieval-Augmented Generation Chain
    """

    def __init__(self, llm):
        self.prompt = PromptManager.get_rag_prompt()
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt
        )

    def _format_context(self, documents: List[Document]) -> str:
        """
        Converts retrieved documents into a single context string
        """
        return "\n\n".join(doc.page_content for doc in documents)

    def clean(self, response) -> str:

        text = response.get("text") 



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

    def run(self, question: str, documents: List[Document]) -> str:
        context = self._format_context(documents)

        #response = self.chain.run(
        #     context=context,
        #     question=question
        # )
        response = self.chain.invoke(
            {"context": context, "question": question}
        )

        return self.clean(response)
    

