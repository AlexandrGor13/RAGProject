import json

from bs4 import BeautifulSoup
import requests

url = "https://batarin-ost.ru/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
category_li = soup.find_all(
    lambda tag: tag.name == "li"
    and tag.get("class") in [["has"], ["nochildren_bg"], ["has active"]]
)

category_folders = [
    cat_li.find(lambda tag: tag.name == "a").get("href") for cat_li in category_li
]

category_folders.append("/magazin/folder/87373702")

products_path = []
for folder in category_folders:
    response = requests.get(url + folder)
    soup = BeautifulSoup(response.content, "html.parser")
    product_div = soup.find_all(
        lambda tag: tag.name == "div" and tag.get("class") == ["product-name"]
    )
    product_path = [
        pr_a.find(lambda tag: tag.name == "a").get("href") for pr_a in product_div
    ]
    products_path += product_path

products = []
products_info = set()
for idx, product in enumerate(products_path):
    response = requests.get(url + product)
    soup = BeautifulSoup(response.content, "html.parser")
    product_div = soup.find(
        lambda tag: tag.name == "div" and tag.get("class") == ["site-main__contain"]
    )
    product_site_path = product_div.find(
        lambda tag: tag.name == "div" and tag.get("class") == ["site-path"]
    )
    product_category = product_site_path.find_all(lambda tag: tag.name == "a")
    product_category = product_category[-1].text.strip() if product_category else ""
    product_name = product_div.find(lambda tag: tag.name == "h1").text.strip()
    product_price = product_div.find(
        lambda tag: tag.name == "div" and tag.get("class") == ["price-current"]
    ).text.strip()
    product_description = product_div.find(
        lambda tag: tag.name == "div"
        and tag.get("class") == ["desc-area", "active-area"]
    )
    product_description = (
        product_description.text.strip() if product_description else ""
    )
    category_item = category.get(product_category.lower())
    current_product = {
        "name": product_name,
        "price": product_price,
        "url": product,
        "category": category_item,
        "description": product_description,
    }
    products.append(current_product)
    print(
        current_product["name"], current_product["price"], current_product["category"]
    )


with open("products.json", "w") as f:
    json.dump(products, f, ensure_ascii=False, indent=4)
