import pathlib

if pathlib.Path("info.txt").exists():
    with open("info.txt") as f:
        info = f.read()


class TemplatePrompt:
    products_prompt = [
        {
            "role": "system",
            "content": """
            Отвечай коротко строго на основании контекста ниже. Используй соответствующую информацию о каждом товаре.
            Не пытайся выдумывать ответ.
            ---
            Контекст:
            {context}
            ---
            Вопрос пользователя:
            Покажи всё что знаешь о {question}
            ---
            Ваш ответ:
            """,
        },
    ]

    common_info_prompt = [
        {
            "role": "system",
            "content": f"""
            Отвечай коротко строго на основании контекста ниже. Ответ должен быть простым и понятным.
            Не пытайся выдумывать ответ. 
            ---        
            Контекст:                            
            {info}"""
            + """     
            ---            
            Вопрос пользователя: 
            {question}
            ---
            Ваш ответ:""",
        },
    ]
