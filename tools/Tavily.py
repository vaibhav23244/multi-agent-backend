from typing import Dict, Union
import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

class TavilySearchQuery(BaseModel):
    query: str = Field(..., description="A natural language query.")

def tavily_search(query: str) -> Union[Dict, str]:
    """Answer the query asked by the user using the Tavily search API."""
    
    # Handle specific user-friendly responses
    greetings = ["hello", "hi", "hey", "greetings"]
    farewell = ["bye", "goodbye", "see you", "take care"]
    
    normalized_query = query.lower().strip()
    if any(greeting in normalized_query for greeting in greetings):
        return "Hello! I was designed by Vaibhav Verma, to answer your queries. How can I assist you today?"
    elif any(farewell in normalized_query for farewell in farewell):
        return "Goodbye! Have a great day!"

    # Proceed with the API call if no predefined responses match
    token = "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"
    url = "https://api.tavily.com/search"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": token
    }
    
    payload = {
        "api_key": token,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 1,
        "include_domains": [],
        "exclude_domains": []
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        search_data = response.json()
        
        answer = search_data.get("answer", "No answer found.")
        
        return answer 
    
    except requests.exceptions.RequestException as err:
        print("Request Error:", err)
        return f"There was an error performing the search. Error: {str(err)}"

tavily_search_tool = StructuredTool.from_function(
    func=tavily_search,
    name="tavily-search",
    description="Perform a web search using the Tavily API to answer user queries.",
    args_schema=TavilySearchQuery,
    return_direct=True
)