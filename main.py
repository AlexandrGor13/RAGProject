import asyncio

from rag.faiss_db import DB_FAISS
from rag.search import QueryPrompt


async def main():
    # Загружает БД с векторными представлениями фрагментов
    db = DB_FAISS().load()
    # print(db.faiss.index_to_docstore_id)
    # db = DB_FAISS()
    # db.add_file("worked.txt")

    query_prompt = QueryPrompt()
    # while True:
    result = await query_prompt.invoke(input("Ваш вопрос: "), db.retriever())
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
