import asyncio

from rag import *


async def main():
    # Загружает БД с векторными представлениями фрагментов
    db = DB_FAISS().load()
    # await db.delete("worked.txt")
    # print(db.faiss.index_to_docstore_id)
    # db = DB_FAISS()
    await db.add_file("worked.txt")

    query = Query2LLM()
    # while True:
    result = await query.invoke(input("Ваш вопрос: "), db.retriever())
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
