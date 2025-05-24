from os import path
import json

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from rag.logger import logger


class DB_FAISS:
    def __init__(
        self,
        embeddings: OllamaEmbeddings = OllamaEmbeddings(
            model="owl/t-lite", base_url="http://localhost:11434"
        ),
        name_source="faiss_index",
    ):
        self.name_source = name_source
        self.embeddings = embeddings
        self.metadata_dict = {}
        self.faiss = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(0),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def load(self) -> "DB_FAISS":
        """
        Загружает БД с векторными представлениями фрагментов
        """
        if path.isdir(self.name_source):
            logger.info("Загружаем БД векторов")
            self.faiss = FAISS.load_local(
                self.name_source,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            with open(self.name_source + r"/source", "r") as f:
                self.metadata_dict = json.load(f)
        else:
            raise FileNotFoundError("Не указан источник для загрузки данных")
        return self

    def retriever(self) -> VectorStoreRetriever:
        return self.faiss.as_retriever()

    async def add_file(self, file_name: str) -> None:
        """
        Добавляем в базу векторное представление из файла
        """
        logger.info("Загрузка файла %s", file_name)
        if path.isfile(file_name):
            documents = TextLoader(file_name).load()
            await self.add_documents(documents)
        else:
            raise FileNotFoundError(f"Файл {file_name} не найден")

    async def add_text(self, text: str, metadata: str) -> None:
        """
        Добавляем в базу векторное представление из текста
        """
        document = Document(page_content=text, metadata={"source": metadata})
        await self.add_documents([document])

    async def add_documents(self, documents: list[Document]) -> None:
        """
        Добавляем в базу векторное представление
        """
        logger.info("Подготовка векторов текста для записи")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        documents_s = text_splitter.split_documents(documents)
        logger.info("Записываем векторные представления фрагментов текста в БД")
        new_faiss = await FAISS.afrom_documents(documents_s, self.embeddings)
        metadata_dict = {}
        for document in new_faiss.docstore.__dict__.get("_dict").values():
            source = document.metadata.get("source")
            if metadata_dict.get(source) is not None:
                metadata_dict[source].append(document.id)
            else:
                metadata_dict[source] = [document.id]
        self.metadata_dict.update(metadata_dict)
        if len(self.faiss.index_to_docstore_id) > 0:
            self.faiss.merge_from(new_faiss)
        else:
            self.faiss = new_faiss
        self.save()

    def save(self) -> None:
        self.faiss.save_local(self.name_source)
        with open(self.name_source + r"/source", "w") as f:
            json.dump(self.metadata_dict, f, ensure_ascii=False, indent=4)

    async def delete(self, source: str) -> None:
        if await self.faiss.adelete(self.metadata_dict.get(source)):
            del self.metadata_dict[source]
            self.save()
