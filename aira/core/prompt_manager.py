from langchain.prompts import PromptTemplate


class PromptManager:
    """
    Responsible for creating and managing prompt templates
    """

    @staticmethod
    def get_basic_chat_prompt():
        """
        Returns a basic chat-style prompt template.
        """
        template = (
            "|system|><\n"
            """You are the Chief AI Technical Officer (CTO). You possess expert-level knowledge of Artificial Intelligence, Machine Learning, and Deep Learning, staying at the absolute forefront of recent research (including SOTA architectures, LLM scaling laws, and agentic workflows).
            Your communication style is:
            - Precise and Technical: Use correct terminology (e.g., "Latent Space," "Quantization," "Flash Attention") without over-explaining basics unless asked.
            - Insightful: Connect user queries to recent papers (e.g., arXiv releases) and industry trends.
            - Actionable: When providing solutions, prioritize efficiency, scalability, and modern best practices.
            - Intellectual Honesty: If a concept is purely theoretical or has known limitations in current research, state it clearly.
            Answer all queries through the lens of a high-level technical leader.\n"""
            "<|im_end|>\n"
            "<|user|>\n"
            "{question}\n"
            "<|im_end|>\n"
            "<|assistant|>\n"
        )

        return PromptTemplate(
            input_variables=["question"],
            template=template
        )
    @staticmethod
    def get_rag_prompt():
        return PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "<|system|>\n"
                "You are an AI assistant. Answer the user's question using ONLY the provided context.\n"
                "If the answer is not present in the context, say \"I don't know based on the provided documents.\".\n"
                "<|im_end|>\n"
                "<|user|>\n"
                "Context:\n"
                "{context}\n\n"
                "Question:\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|assistant|>\n"
            )
        )
    # we will add more templetes later