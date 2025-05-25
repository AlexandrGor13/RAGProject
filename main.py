import asyncio

from rag import DB_FAISS, Query2LLM


async def main():
    # Загружает БД с векторными представлениями фрагментов
    # db = DB_FAISS().load()
    # await db.delete("worked_.txt")

    db = DB_FAISS()
    await db.add_file("worked.txt")
    await db.add_file("worked_.txt")

    query = Query2LLM()
    while True:
        result = await query.invoke(input("Ваш вопрос: "), db.retriever())
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
