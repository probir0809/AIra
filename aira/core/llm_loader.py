from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from .config import MODEL_NAME, DEVICE, MAX_TOKENS, TEMPERATURE, TOP_P


class AIraModel:
    """
    Core interface to load and expose the language model.
    """
    def __init__(self):
        logger.info(f"Loading model '{MODEL_NAME}' on {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if DEVICE == "cuda" else None
        )
        self.model.eval()

        logger.info("Model loaded successfully!")

        self._llm = self._build_langchain_llm()

    def _build_langchain_llm(self):
        """
        Wrap HuggingFace model into a LangChain-compatible LLM.
        """
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    @property
    def llm(self):
        """
        Exposes LangChain-compatible LLM.
        """
        return self._llm
