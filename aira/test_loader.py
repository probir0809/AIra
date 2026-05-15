from aira.rag.loader import PDFLoader
from aira.rag.chunker import TextChunker

loader = PDFLoader()
docs = loader.load_directory("/home/gaian/Desktop/rag")

chunker = TextChunker()
chunks = chunker.split(docs)

print("Total chunks:", len(chunks))
print("\nSample chunk:\n")
print(chunks[0].page_content[:500])
print("\nMetadata:", chunks[0].metadata)
