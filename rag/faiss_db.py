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

from .logger import logger


class DB_FAISS:
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
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
        self.chunk_size = 1000
        self.chunk_overlap = 150

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
        if self.length > 0:
            return self.faiss.as_retriever()
        else:
            raise IndexError("В базе данных нет векторов текста")

    async def from_txt_file(self, file_name: str) -> None:
        """
        Добавляем в базу векторное представление из файла
        """
        logger.info("Загрузка файла %s", file_name)
        if path.isfile(file_name):
            documents = TextLoader(file_name).load()
            await self.add_documents(documents)
        else:
            raise FileNotFoundError(f"Файл {file_name} не найден")

    async def from_json_file(self, file_name: str) -> None:
        """
        Добавляем в базу векторное представление из json-файла
        """
        logger.info("Загрузка файла %s", file_name)
        if path.isfile(file_name):
            with open(file_name, "r") as f:
                data = json.load(f)
            documents = [
                Document(
                    page_content=f"{d.get('name')}\n{d.get('specifications')}\nцена {d.get('price')} {d.get("category")}",
                    metadata={
                        "source": d.get("category"),
                        "name": d.get("name"),
                        "specifications": d.get("specifications"),
                        "price": "цена " + d.get("price"),
                        "description": d.get("description"),
                        "url": d.get("url"),
                    },
                )
                for d in data
            ]
            await self.add_documents(documents)
        else:
            raise FileNotFoundError(f"Файл {file_name} не найден")

    async def add_text(self, text: str, metadata: dict) -> None:
        """
        Добавляем в базу векторное представление из текста
        """
        if metadata.get("source"):
            document = Document(page_content=text, metadata=metadata)
            await self.add_documents([document])
        else:
            raise Exception("The metadata must contain the 'source' key")

    async def add_documents(self, documents: list[Document], autosave=True) -> None:
        """
        Добавляем в базу векторное представление
        """
        logger.info("Подготовка векторов текста для записи")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
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
        if self.length > 0:
            self.faiss.merge_from(new_faiss)
            self.metadata_dict.update(metadata_dict)
        else:
            self.faiss = new_faiss
            self.metadata_dict = metadata_dict
        if autosave:
            self.save()

    def save(self) -> None:
        """
        Сохраняет БД с векторными представлениями фрагментов
        """
        self.faiss.save_local(self.name_source)
        with open(self.name_source + r"/source", "w") as f:
            json.dump(self.metadata_dict, f, ensure_ascii=False, indent=4)

    async def delete(self, source: str) -> None:
        """
        Удаляет из базы векторное представление фрагмента, где в metadata источник текста source
        """
        if await self.faiss.adelete(self.metadata_dict.get(source)):
            del self.metadata_dict[source]
            self.save()

    @property
    def length(self):
        length = 0
        for v in self.metadata_dict.values():
            length += len(v)
        return length
