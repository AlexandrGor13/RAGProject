import asyncio
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rag import DB_FAISS, Query2LLM

""" Пример для демонстрации """


async def main():
    model = "t-lite-quantized"
    url = "http://localhost:11434"
    # Загружает БД с векторными представлениями фрагментов текста
    db = DB_FAISS(
        OllamaEmbeddings(model=model, base_url=url), name_source="faiss_index_tlite"
    )
    try:
        db = db.load()
    except:
        await db.from_json_file("products.json")

    query_tlite = Query2LLM(model=OllamaLLM(model=model, temperature=0.2, base_url=url))
    while True:
        retriever = db.retriever()
        result = await query_tlite.invoke(input("Напишите свой вопрос: "), retriever)
        print(result, end="\n\n")


if __name__ == "__main__":
    asyncio.run(main())
