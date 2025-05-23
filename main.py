from rag.faiss_db import DB_FAISS
from rag.search import QueryPrompt

# Загружает БД с векторными представлениями фрагментов
# db = DB_FAISS().load()
# print(db.faiss.index_to_docstore_id)
# db.add_text("DFG большой человек, который живет в ZXC. Он суров и мягок.", "temp")
db = DB_FAISS()
db.add_file("worked.txt")

query_prompt = QueryPrompt()
while True:
    result = query_prompt.invoke(input("Ваш вопрос: "), db.retriever())
    print(result)
