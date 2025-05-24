from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaLLM

from .logger import logger


class Query2LLM:
    def __init__(self, model="owl/t-lite"):
        self.model = OllamaLLM(
            model=model, temperature=0.2, base_url="http://localhost:11434"
        )
        self.tamplate = """
        Используйте приведенные ниже фрагменты из извлеченного контекста, чтобы ответить на вопрос.
        Если вы не знаете ответа, просто скажите, что не знаете. Отвечайте как можно короче.
        Context: {context} \nQuestion: {question}"""
        self.sources = set()

    def format_documents(self, documents: list[Document]):
        """ """
        sources = set()
        result_content = []
        logger.info("Форматирование текста")
        for document in documents[:3]:
            result_content.append(document.page_content)
            sources.add(document.metadata.get("source"))
        self.sources = sources
        return "\n".join(result_content)

    @property
    def prompt(self):
        logger.info("Настраиваем шаблон запроса в понятном для модели формате")
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.tamplate,
                )
            ]
        )

    async def invoke(self, query: str, retriever: VectorStoreRetriever):
        chain = (
            RunnableParallel(
                context=retriever | self.format_documents,
                question=lambda data: data,
            )
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        result = await chain.ainvoke(query)
        if self.sources and "не знаю" not in result.lower():
            result += f"\n Источники: {", ".join(self.sources).strip(",")}"
        return result
