import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaLLM
from pymorphy3 import MorphAnalyzer

from .logger import logger
from .templates import TemplatePrompt


class Query2LLM:
    def __init__(self, model: OllamaLLM):
        self.model = model
        self.tamplate = TemplatePrompt.products_prompt
        self.sources = set()
        self.morph_analyzer = MorphAnalyzer()

    def format_context(self, documents: list[Document]):
        """ """
        sources = set()
        result_content = []
        logger.info("Форматирование текста")
        for document in documents[:11]:
            result_content.append(document.page_content)
            sources.add(document.metadata.get("source"))
        self.sources = sources
        return "\n".join(result_content)

    def format_product_list(self, documents: list[Document]):
        """ """
        url = "https://example.ru"
        product_list = []
        logger.info("Форматирование текста")
        for document in documents[:6]:
            data = document.metadata
            data_str = f"\t- {data.get("name")} ({data.get("specifications")}) {data.get("price")} {url + data.get("url")}"
            product_list.append(data_str)
        return "\n".join(product_list)

    @property
    def prompt(self):
        logger.info("Настраиваем шаблон запроса в понятном для модели формате")
        result = ChatPromptTemplate.from_messages(self.tamplate)
        return result

    def process_query(self, query):
        """Фильтрация слов по частям речи"""
        filtered_words_noun = []
        filtered_words_verb = []
        for word in query.split():
            parsed_word = self.morph_analyzer.parse(word)[0]
            normalized_word = self.morph_analyzer.parse(word)[0].normal_form
            if parsed_word.tag.POS in {"NOUN"}:
                filtered_words_noun.append(normalized_word)
            elif parsed_word.tag.POS in {"VERB", "INFN"}:
                filtered_words_verb.append(normalized_word)
        self.classify_query(filtered_words_verb, filtered_words_noun)
        filtered_words_noun = " ".join(filtered_words_noun)
        return filtered_words_noun

    async def invoke(self, query: str, retriever: VectorStoreRetriever):
        """ """
        filtered_words_noun = self.process_query(query)
        product_list = (retriever | self.format_product_list).invoke(
            filtered_words_noun
        )
        chain = (
            RunnableParallel(
                context=retriever | self.format_context,
                question=lambda data: data,
            )
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        result = await chain.ainvoke(filtered_words_noun)
        result_rpod_list = "\n\nПользуются спросом следующие товары:\n" + product_list
        return result + result_rpod_list

    def classify_query(self, query_verb: list, query_noun: list):
        # if pattern.match(query):
        #     if category == "fact":
        #         # self.tamplate = TemplatePrompt.common_info_prompt
        #         self.tamplate = TemplatePrompt.products_prompt
        #     elif category == "product":
        #         self.tamplate = TemplatePrompt.products_prompt
        #     else:
        #         self.tamplate = TemplatePrompt.other_prompt
        #     return category
        return None
