# ==============================
# SMART SHOPPING ASSISTANT
# Retail & E-Commerce using GenAI
# ==============================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

import streamlit as st
import requests
import json
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛒 Smart Shopping Recommendation Assistant")

st.write("""
This AI Assistant recommends products based on

✔ Budget

✔ Requirements

✔ Product Features

✔ Comparison

✔ AI Buying Suggestion

✔ Alternatives

✔ Latest Product Search
""")

st.divider()

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("API Configuration")

GOOGLE_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

TAVILY_API_KEY = st.sidebar.text_input(
    "Tavily API Key",
    type="password"
)

all_api = [
    GOOGLE_API_KEY,
    TAVILY_API_KEY
]

if not all(all_api):

    st.warning("Please Enter All API Keys")

    st.stop()

st.sidebar.success("API Loaded Successfully")

# ==============================
# GEMINI MODEL
# ==============================

model = ChatGoogleGenerativeAI(

    model="gemini-2.5-flash",

    google_api_key=GOOGLE_API_KEY

)

# ==============================
# SHOPPING DETAILS
# ==============================

st.sidebar.title("Shopping Preferences")

category = st.sidebar.selectbox(

    "Product Category",

    [

        "electronics",

        "jewelery",

        "men's clothing",

        "women's clothing"

    ]

)

budget = st.sidebar.number_input(

    "Budget",

    min_value=100,

    max_value=1000000,

    value=50000

)

brand = st.sidebar.text_input(

    "Preferred Brand",

    placeholder="Optional"

)

rating = st.sidebar.slider(

    "Minimum Rating",

    1,

    5,

    4

)

st.markdown("## Product Requirements")

requirements = st.text_area(

    "Describe your requirements",

    height=180,

    placeholder="""
Example:

Gaming Laptop

RTX 4060

16GB RAM

1TB SSD

Battery Backup

Video Editing

Budget 70000
"""

)

st.divider()

# ==============================
# USER QUERY
# ==============================

user_query = f"""

Budget : {budget}

Category : {category}

Brand : {brand}

Minimum Rating : {rating}

Requirements :

{requirements}

"""
# ===========================================
# TOOL 1 : SEARCH LATEST PRODUCTS USING TAVILY
# ===========================================

def search_products(query):
    """
    Search latest products, reviews and buying guides.
    """

    client = TavilyClient(api_key=TAVILY_API_KEY)

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    return response


# ===========================================
# TOOL 2 : GET PRODUCTS
# ===========================================

def get_products(category):
    """
    Fetch products from FakeStore API.
    """

    url = f"https://fakestoreapi.com/products/category/{category}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()

    return []


# ===========================================
# TOOL 3 : BUDGET FILTER
# ===========================================

def budget_filter(products, budget):
    """
    Filter products according to user budget.
    """

    result = []

    for product in products:

        price = float(product.get("price", 0))

        if price <= budget:
            result.append(product)

    return result


# ===========================================
# TOOL 4 : PRODUCT COMPARISON
# ===========================================

def compare_products(products):
    """
    Compare products using Gemini.
    """

    prompt = f"""

You are an AI Shopping Assistant.

Compare the following products.

Return output in table format.

Mention

1 Price

2 Features

3 Pros

4 Cons

5 Rating

6 Best Choice

Products

{products}

"""

    response = model.invoke(prompt)

    return response.content


# ===========================================
# TOOL 5 : PRODUCT RECOMMENDATION
# ===========================================

def recommend_products(products, requirements):
    """
    Recommend best products based on user requirements.
    """

    prompt = f"""

You are an AI Shopping Recommendation Assistant.

Requirements

{requirements}

Products

{products}

Recommend Top 5 Products.

For every product provide

Product Name

Price

Why Recommended

Pros

Cons

Buying Score out of 10

"""

    response = model.invoke(prompt)

    return response.content


# ===========================================
# TOOL 6 : ALTERNATIVE PRODUCTS
# ===========================================

def alternative_products(products):
    """
    Suggest alternative products.
    """

    prompt = f"""

Suggest affordable alternatives.

Products

{products}

"""

    response = model.invoke(prompt)

    return response.content


# ===========================================
# CREATE LEADER AGENT
# ===========================================

leader_agent = create_agent(

    model=model,

    tools=[

        search_products,

        get_products

    ]

)

print("Leader Agent Created Successfully")

# ===========================================
# MAIN SHOPPING AGENT
# ===========================================

def shopping_assistant(agent, query):
    """
    Leader Agent responsible for generating
    final shopping recommendation.
    """

    prompt = f"""
You are Smart Shopping Recommendation Assistant.

Your responsibilities are:

1. Understand customer requirements.

2. Understand budget.

3. Recommend only products within budget.

4. Compare products.

5. Suggest alternatives.

6. Explain why product is best.

7. Mention Pros.

8. Mention Cons.

9. Give Buying Verdict.

10. Show latest product information if required.

Return output only in HTML.

User Query:

{query}

"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    try:
        return response["messages"][-1].content
    except:
        return str(response)


# ===========================================
# LOAD PRODUCTS
# ===========================================

products = get_products(category)

filtered_products = budget_filter(products, budget)

# ===========================================
# BUTTON
# ===========================================

if st.button("🛒 Recommend Products", use_container_width=True):

    with st.spinner("Finding Best Products..."):

        latest = search_products(
            f"{category} best products under {budget}"
        )

        latest_results = latest.get("results", [])

        html = shopping_assistant(
            leader_agent,
            user_query
        )

        st.success("Recommendation Generated Successfully")

        st.markdown("## AI Shopping Recommendation")

        try:
            st.html(
                html,
                width="stretch",
                unsafe_allow_javascript=True
            )

        except:
            st.markdown(html)

        st.divider()

        st.subheader("Products From FakeStore API")

        if len(filtered_products) == 0:

            st.warning("No products found within your budget.")

        else:

            cols = st.columns(2)

            index = 0

            for product in filtered_products:

                with cols[index % 2]:

                    st.image(
                        product["image"],
                        width=180
                    )

                    st.markdown(
                        f"### {product['title']}"
                    )

                    st.write(
                        product["description"][:180]
                    )

                    st.metric(
                        "Price",
                        f"${product['price']}"
                    )

                    st.write(
                        "⭐",
                        product["rating"]["rate"]
                    )

                index += 1

        st.divider()

        st.subheader("Latest Shopping Results")

        if len(latest_results) == 0:

            st.info("No latest search results found.")

        else:

            for item in latest_results:

                title = item.get("title", "No Title")

                url = item.get("url", "")

                content = item.get("content", "")

                with st.expander(title):

                    st.write(content)

                    st.link_button(
                        "Open Website",
                        url
                    )

        st.divider()

        st.subheader("Product Comparison")

        comparison = compare_products(
            filtered_products
        )

        st.markdown(comparison)

        st.divider()

        st.subheader("Alternative Products")

        alt = alternative_products(
            filtered_products
        )

        st.markdown(alt)

        st.divider()

        st.success("Shopping Recommendation Completed")
