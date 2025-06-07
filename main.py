import asyncio
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rag import DB_FAISS, Query2LLM


async def main():
    model = "t-lite-quantized"
    url = "http://localhost:11434"
    # Загружает БД с векторными представлениями фрагментов
    db_tlite = DB_FAISS(
        OllamaEmbeddings(model=model, base_url=url), name_source="faiss_index_tlite1"
    )
    try:
        db_tlite = db_tlite.load()
    except:
        await db_tlite.from_json_file("products1.json")

    query_tlite = Query2LLM(model=OllamaLLM(model=model, temperature=0.2, base_url=url))
    while True:
        retriever = db_tlite.retriever()
        result = await query_tlite.invoke(input("Ваш вопрос: "), retriever)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
