from os import path
import json

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pymorphy3 import MorphAnalyzer

from .logger import logger
from .templates import TemplatePrompt


class DB_FAISS:
    def __init__(
        self,
        model: str,
        base_url: str,
        name_source="faiss_index",
        chunk_size=1000,
        chunk_overlap=150,
    ):
        self.name_source = name_source
        self.embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        self.model = OllamaLLM(model=model, temperature=0.2, base_url=base_url)
        self.metadata_dict = {}
        self.faiss = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(0),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sources = set()
        self.morph_analyzer = MorphAnalyzer()
        self.tamplate = TemplatePrompt.products_prompt

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

    def retriever(self, query: str):
        if self.length > 0:
            stop_words = {"системы", "материалы", "изделия", "продукция", "для"}
            words = self.process_query(query)
            list_query_words = set()
            if words.get("NOUN"):
                list_query_words = list_query_words | set(words.get("NOUN"))
            if words.get("ADJF"):
                list_query_words = list_query_words | set(words.get("ADJF"))
            words_source = {}
            for w in self.metadata_dict.keys():
                w_new = w
                for s in stop_words:
                    w_new = w_new.replace(s, "").strip()
                words_source.update(
                    {self.morph_analyzer.parse(w_new)[0].normal_form: w}
                )
            source = []
            for k, v in words_source.items():
                if k in list_query_words:
                    source.append(v)
            if source:
                logger.info("Поиск в источниках: %s", source)
                return self.faiss.as_retriever(
                    search_kwargs={"k": 30, "filter": {"source": {"$in": source}}},
                )
            else:
                logger.info("Поиск по всем источникам")
                return self.faiss.as_retriever(
                    search_kwargs={"k": 50},
                )
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
                    # page_content=f"{d.get("category")} {d.get('name')} {d.get('specifications')} цена {d.get('price')}",
                    page_content=f"{d.get("category")} {d.get('name')}",
                    metadata={
                        "source": d.get("category"),
                        "name": d.get("name"),
                        "specifications": d.get("specifications"),
                        "price": "цена " + d.get("price"),
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

    def __format_context(self, documents: list[Document]):
        """ """
        sources = set()
        result_content = []
        logger.info("Форматирование контекста")
        for document in documents:
            data = document.metadata
            result_content = f"{data.get("category")} {data.get('name')} {data.get('specifications')} цена {data.get('price')}"
            # result_content.append(document.page_content)
            sources.add(document.metadata.get("source"))
        self.sources = sources
        return "\n".join(result_content)

    def __format_product_list(self, documents: list[Document]):
        """ """
        url = "https://example.ru"
        product_list = []
        logger.info("Форматирование списка товаров")
        for document in documents[:5]:
            data = document.metadata
            data_str = f"\t- {data.get("name")} ({data.get("specifications")}) {data.get("price")} {url + data.get("url")}"
            product_list.append(data_str)
        return "\n".join(product_list)

    def process_query(self, query):
        """Фильтрация слов по частям речи"""
        filtered_words = {}
        for word in query.split():
            parsed_word = self.morph_analyzer.parse(word)[0]
            normalized_word = self.morph_analyzer.parse(word)[0].normal_form
            if parsed_word.tag.POS == "NOUN":
                filtered_words["NOUN"] = (
                    [normalized_word]
                    if not filtered_words.get("NOUN")
                    else filtered_words.get("NOUN") + [normalized_word]
                )
            elif parsed_word.tag.POS == "ADJF":
                filtered_words["ADJF"] = (
                    [normalized_word]
                    if not filtered_words.get("ADJF")
                    else filtered_words.get("ADJF") + [normalized_word]
                )
        return filtered_words

    @property
    def prompt(self):
        logger.info("Настраиваем шаблон запроса в понятном для модели формате")
        result = ChatPromptTemplate.from_messages(self.tamplate)
        return result

    async def invoke(self, query: str):
        """ """
        recommendation = self.__classify_query(query)
        retriever = self.retriever(query)
        product_list = ""
        if recommendation:
            product_list = (retriever | self.__format_product_list).invoke(query)
        chain = (
            RunnableParallel(
                context=retriever | self.__format_context,
                question=lambda data: data,
            )
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        result = await chain.ainvoke(query)
        return (
            result + "\n\nРекомендуем следующие товары:\n" + product_list
            if len(product_list)
            else result
        )

    def __classify_query(self, query: str):
        recommendation = False
        is_noun = (
            {"адрес", "телефон", "работа", "сайт", "режим", "ссылка"}
            & set(self.process_query(query).get("NOUN"))
            if self.process_query(query).get("NOUN")
            else False
        )
        is_verb = (
            {"работает", "открывается"} & set(self.process_query(query).get("VERB"))
            if self.process_query(query).get("VERB")
            else False
        )
        is_infn = (
            {"работать", "находится", "позвонить", "зайти"}
            & set(self.process_query(query).get("INFN"))
            if self.process_query(query).get("INFN")
            else False
        )
        if is_noun or is_verb or is_infn:
            self.tamplate = TemplatePrompt.common_info_prompt
        else:
            self.tamplate = TemplatePrompt.products_prompt
            recommendation = True
        return recommendation
