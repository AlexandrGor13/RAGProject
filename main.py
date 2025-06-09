import asyncio
from rag import DB_FAISS

""" Пример для демонстрации """


async def main():
    model = "t-lite-quantized"
    url = "http://localhost:11434"
    # Загружает БД с векторными представлениями фрагментов текста
    db = DB_FAISS(model=model, base_url=url, name_source="faiss_index_tlite")
    try:
        db = db.load()
    except:
        await db.from_json_file("products.json")

    while True:
        result = await db.invoke(input("Напишите свой вопрос: "))
        print(result, end="\n\n")


if __name__ == "__main__":
    asyncio.run(main())
