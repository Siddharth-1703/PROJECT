# ==============================
# SMART SHOPPING ASSISTANT
# Retail & E-Commerce using GenAI
# ==============================
import re
import warnings
import requests
import streamlit as st
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient
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
# SIDEBAR - API KEYS
# ==============================
st.sidebar.title("API Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TAVILY_API_KEY = st.sidebar.text_input("Tavily API Key", type="password")

if not all([GOOGLE_API_KEY, TAVILY_API_KEY]):
    st.warning("Please Enter All API Keys")
    st.stop()

st.sidebar.success("API Loaded Successfully")
# ==============================
# GEMINI MODEL
# ==============================
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)
# ==============================
# SHOPPING DETAILS
# ==============================
st.sidebar.title("Shopping Preferences")
category = st.sidebar.selectbox(
    "Product Category",
    ["electronics", "jewelery", "men's clothing", "women's clothing"]
)
budget = st.sidebar.number_input(
    "Budget",
    min_value=100,
    max_value=1000000,
    value=50000
)
brand = st.sidebar.text_input("Preferred Brand", placeholder="Optional")
rating = st.sidebar.slider("Minimum Rating", 1, 5, 4)
st.markdown("## Product Requirements")
requirements = st.text_area(
    "Describe your requirements",
    height=180,
    placeholder="""Example:
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
Brand : {brand if brand else "Any"}
Minimum Rating : {rating}

Requirements :
{requirements}
"""
# ===========================================
# TOOL 1 : SEARCH LATEST PRODUCTS USING TAVILY
# ===========================================
def search_products(query: str) -> dict:
    """Search latest products, reviews and buying guides."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        return client.search(query=query, search_depth="advanced", max_results=5)
    except Exception as e:
        st.warning(f"Tavily search failed: {e}")
        return {"results": []}
# ===========================================
# TOOL 2 : GET PRODUCTS
# ===========================================
def get_products(category: str) -> list:
    """Fetch products from FakeStore API."""
    try:
        url = f"https://fakestoreapi.com/products/category/{category}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        st.warning(f"Could not fetch products: {e}")
    return []
# ===========================================
# TOOL 3 : BUDGET / RATING / BRAND FILTER
# ===========================================
def filter_products(products, budget, min_rating=1, brand=""):
    """Filter products by budget, minimum rating and (optional) brand keyword."""
    result = []
    brand_lower = brand.strip().lower()
    
    for product in products:
        price = float(product.get("price", 0))
        product_rating = float(product.get("rating", {}).get("rate", 0))
        title = product.get("title", "").lower()

        if price > budget:
            continue
        if product_rating < min_rating:
            continue
        if brand_lower and brand_lower not in title:
            continue

        result.append(product)

    return result
# ===========================================
# TOOL 4 : PRODUCT COMPARISON
# ===========================================
def compare_products(products):
    """Compare products using Gemini."""
    if not products:
        return "No products available to compare."
    prompt = f"""
You are an AI Shopping Assistant.
Compare the following products.
Return output as a Markdown table with columns:
Price | Features | Pros | Cons | Rating | Best Choice
Products:
{products}
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# TOOL 5 : PRODUCT RECOMMENDATION
# ===========================================
def recommend_products(products, requirements):
    """Recommend best products based on user requirements."""
    if not products:
        return "No products available to recommend."
    prompt = f"""
You are an AI Shopping Recommendation Assistant.

Requirements:
{requirements}
Products:
{products}
Recommend Top 5 Products.
For every product provide (in Markdown):
- Product Name
- Price
- Why Recommended
- Pros
- Cons
- Buying Score out of 10
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# TOOL 6 : ALTERNATIVE PRODUCTS
# ===========================================
def alternative_products(products):
    """Suggest alternative products."""
    if not products:
        return "No products available to suggest alternatives for."
    prompt = f"""
Suggest affordable alternatives (in Markdown) for the following products.
Products:
{products}
"""
    response = model.invoke(prompt)
    return response.content
# ===========================================
# HELPER: clean model output before rendering
# ===========================================
def clean_llm_output(text) -> str:
    """Strip ```html / ``` code fences the model sometimes wraps output in.
    Defensive against non-string input: some LangChain/Gemini responses
    return `content` as None or as a list of content-part dicts instead
    of a plain string.
    """
    if text is None:
        return ""
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        text = "\n".join(p for p in parts if p)
        
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:html|markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    
    text = "\n".join(line.lstrip() for line in text.splitlines())

    return text
# ===========================================
# CREATE LEADER AGENT
# ===========================================
leader_agent = create_agent(
    model=model,
    tools=[search_products, get_products]
)
# ===========================================
# MAIN SHOPPING AGENT
# ===========================================
def shopping_assistant(agent, query):
    """Leader Agent responsible for generating the final shopping recommendation."""
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
Return output only in clean HTML (no markdown, no code fences).
User Query:
{query}
"""

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    try:
        content = response["messages"][-1].content
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        st.warning(f"Unexpected agent response shape ({e}); showing raw output.")
        content = str(response)

    return clean_llm_output(content)
# ===========================================
# LOAD PRODUCTS
# ===========================================
products = get_products(category)
filtered_products = filter_products(products, budget, rating, brand)
# ===========================================
# BUTTON
# ===========================================
if st.button("🛒 Recommend Products", use_container_width=True):
    with st.spinner("Finding Best Products..."):

        latest = search_products(f"{category} best products under {budget}")
        latest_results = latest.get("results", [])

        try:
            html = shopping_assistant(leader_agent, user_query)
        except Exception as e:
            st.error(f"Gemini request failed: {e}")
            st.stop()

        st.success("Recommendation Generated Successfully")
        st.markdown("## AI Shopping Recommendation")

        cleaned_html = clean_llm_output(html)
        components.html(cleaned_html, height=900, scrolling=True)

        st.divider()
        st.subheader("Products From FakeStore API")

        if len(filtered_products) == 0:
            st.warning("No products found matching your budget / rating / brand filters.")
        else:
            cols = st.columns(2)
            for index, product in enumerate(filtered_products):
                with cols[index % 2]:
                    st.image(product["image"], width=180)
                    st.markdown(f"### {product['title']}")
                    st.write(product["description"][:180])
                    st.metric("Price", f"${product['price']}")
                    st.write("⭐", product["rating"]["rate"])

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
                    if url:
                        st.link_button("Open Website", url)

        st.divider()
        st.subheader("Product Comparison")
        comparison = compare_products(filtered_products)
        st.markdown(clean_llm_output(comparison))

        st.divider()
        st.subheader("Alternative Products")
        alt = alternative_products(filtered_products)
        st.markdown(clean_llm_output(alt))

        st.divider()
        st.success("Shopping Recommendation Completed")
