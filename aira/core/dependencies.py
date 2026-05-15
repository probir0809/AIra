from aira.core.llm_loader import AIraModel
from aira.chains.basic_chain import BasicChain

def get_basic_chain():
    llm = AIraModel().llm
    return BasicChain(llm)
