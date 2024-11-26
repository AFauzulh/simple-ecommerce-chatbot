import os
import json
import requests

import numpy as np

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from langchain_core.tools import tool

import pprint

def retrieve_from_apis(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return data

def refine_query(query):
    keyword_responses = {
        "category": " To get the categories list using the get_categories_list() function, if the user requested a list of items in the category you can use get_product_by_category() function.",
        "categories": " To get the categories list using the get_categories_list() function",
        "hot": " You can get the highest rated (hottest) product in the specific category using get_hot_product_by_categories() function. You should give the top 5 cheapest product if there is more than 5 result",
        "cheap": " You can get the cheapest product in a specific category using get_cheapest_product_in_the_category() function first. If the result is empty then you should use get_cheapest_product() function instead to get the cheapest product by product name. You should give the top 5 cheapest product if there is more than 5 result",
        "deal": " To get the best deal or sale product in a specific category, you should get the discounted product by using get_biggest_discount_by_category() function first. If the result is empty then you should use get_biggest_discount_by_product_name() function to get the product with the biggest discount by product name. If the result is also empty, use the get_biggest_discount() function to get the best deal of the all item. You should give the top 5 cheapest product if there is more than 5 result",
        "sale": " To get the best deal or sale product in a specific category, you should get the discounted product by using get_biggest_discount_by_category() function first. If the result is empty then you should use get_biggest_discount_by_product_name() function to get the product with the biggest discount by product name. If the result is also empty, use the get_biggest_discount() function to get the best deal of the all item. You should give the top 5 cheapest product if there is more than 5 result",
        "compare": " This command aims to compare between two products. To do that, you can use search_product_by_name() function multiple times to get the information for each product, then compare them.",
        }
    
    for keyword, additional_query in keyword_responses.items():
        if keyword in query:
            query = query + additional_query
            break
                
    return query

def agent_executor_invoker(query, agent_executor):
    query = refine_query(query)
    return agent_executor.invoke({"input": query})

def get_agent_tools():
    return [
        get_categories_list, get_product_by_category,
        get_hot_product_by_categories,
        get_product_category_by_maxprice,
        get_cheapest_product_in_the_category, get_cheapest_product, 
        get_biggest_discount_by_category, get_biggest_discount_by_product_name, get_biggest_discount,
        search_product_by_name
    ]


@tool
def get_categories_list():
    """
    Get list of category
    """
    
    url = 'https://dummyjson.com/products/category-list'
    return retrieve_from_apis(url)

@tool
def get_product_by_category(category_name, limit):
    """
    this function is to get the random products by category name with the defined limit 
    """
    
    url = f"https://dummyjson.com/products/category/{category_name}?limit={limit}&select=title,rating,reviews,price,discountPercentage"
    return retrieve_from_apis(url)
    
@tool
def get_hot_product_by_categories(category_name):
    """
    this function is to get the hottest or the highest rated product in the specific category by sorting the highest rating of product
    """
    
    url = f"https://dummyjson.com/products/category/{category_name}?select=title,reviews,price,discountPercentage"
    result = retrieve_from_apis(url)
    
    refined_products = []
    for product in result["products"]:
        refined_product = {}
        ratings = []
        
        for review in product["reviews"]:
            ratings.append(review["rating"])
        
        avg_rating = np.average(ratings)
        refined_product = {
            "title": product["title"],
            "average_rating": float("{:.1f}".format(avg_rating)),
            "price": product["price"],
            "discount": product["discountPercentage"],
            "price_after_discount": product["price"] * (1 - product["discountPercentage"] / 100)
        }
        
        refined_products.append(refined_product)
    print(refined_products)
    return sorted(refined_products, key=lambda x: x["average_rating"], reverse=True)

@tool
def get_cheapest_product_in_the_category(category_name):
    """
    get the cheapest or lowest product price in the specific category, use this if user give the category name
    """
    
    url = f"https://dummyjson.com/products/category/{category_name}?limit=10&select=title,price,discountPercentage"
    results = retrieve_from_apis(url)["products"]
    for product in results:
        product["discounted_price"] = product["price"] * (1 - product["discountPercentage"] / 100)
    
    
    return sorted(results, key=lambda x: x["discounted_price"], reverse=True)

@tool 
def get_cheapest_product(product_name):
    """
    get the cheapest or lowest product price after discount by product name, use this if only user give the name of the product or if the name of the product is not exist in product categories list
    """
    
    url = f"https://dummyjson.com/products/search?q={product_name}&limit=10&select=title,price,discountPercentage"
    results = retrieve_from_apis(url)["products"]
    for product in results:
        product["discounted_price"] = product["price"] * (1 - product["discountPercentage"] / 100)
    
    
    return sorted(results, key=lambda x: x["discounted_price"], reverse=True)

@tool
def get_biggest_discount_by_category(category_name):
    """
    get products in a specific category with the largest or biggest discount percentages, or you can say this is the best deal for the product in a specific category on this ecommerce
    """
    
    url = f"https://dummyjson.com/products/category/{category_name}?sortBy=discountPercentage&order=desc&select=title,price,discountPercentage"
    result = retrieve_from_apis(url)["products"]
    
    return result

@tool
def get_biggest_discount_by_product_name(product_name):
    """
    get products with the largest or biggest discount percentages using a product name, or you can say this is the best deal for the product on this ecommerce using product name
    """
    
    url = f"https://dummyjson.com/products/search?q={product_name}&sortBy=discountPercentage&order=desc&select=title,price,discountPercentage"
    result = retrieve_from_apis(url)["products"]
    
    return result

@tool
def get_biggest_discount():
    """
    get products with the largest or biggest discount percentages, or you can say this is the best deal for the product on this ecommerce
    """
    
    url = "https://dummyjson.com/products?sortBy=discountPercentage&order=desc&select=title,price,discountPercentage"
    result = retrieve_from_apis(url)["products"]
    
    return result

@tool
def search_product_by_name(product_name):
    """
    get product information by product name
    """
    
    url = f"https://dummyjson.com/products/search?q={product_name}"
    return retrieve_from_apis(url)


@tool
def get_product_category_by_maxprice(category_name, max_price):
    """
    get the product in a specific category with the defined maximum price
    """
    
    url = f"https://dummyjson.com/products/category/{category_name}?select=title,description,price,discountPercentage,rating,images"
    result = retrieve_from_apis(url)["products"]
    
    filtered_result = []
    for product in result:
        if product["price"] <= float(max_price):
            filtered_result.append(product)
    
    return filtered_result