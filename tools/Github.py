from typing import Dict, Union
import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
import re

class GithubRepoQuery(BaseModel):
    query: str = Field(..., description="A natural language query about a GitHub repository.")

def parse_github_query(query: str) -> Dict[str, str]:
    # Updated regex to correctly extract owner and repo
    match = re.search(r'([a-zA-Z0-9-]+)/([a-zA-Z0-9-]+)', query)
    if match:
        return {"owner": match.group(1), "repo": match.group(2)}
    else:
        return {}

def github_repo(query: str) -> Union[Dict, str]:
    """Get brief information about a GitHub repository based on a natural language query."""
    parsed = parse_github_query(query)
    if not parsed:
        return "Could not parse the repository information from the query. Please ensure you mention the repository in the format 'owner/repo'."

    owner, repo = parsed['owner'], parsed['repo']
    token = "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"
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    url = f"https://api.github.com/repos/{owner}/{repo}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repo_data = response.json()
        
        description = repo_data.get("description", "")
        language = repo_data.get("language", "")
        use_cases = []
        
        if "UI" in description or "frontend" in description.lower():
            use_cases.append("Can be used as a reference for building user interfaces")
        if "TypeScript" in language:
            use_cases.append("Useful for learning TypeScript best practices in web development")
        if "LangChain" in description:
            use_cases.append("Can be studied to understand LangChain implementation in web applications")
        if "AI" in description or "generative" in description.lower():
            use_cases.append("Provides insights into integrating AI capabilities in web applications")
        
        return {
            "owner": owner,
            "repo": repo,
            "description": description,
            "stars": repo_data.get("stargazers_count", 0),
            "language": language,
            "suggested_use_cases": use_cases or "No relevant use cases were found."
        }
    except requests.exceptions.RequestException as err:
        print(err)
        return f"There was an error fetching the repository. Please check the owner and repo names. Error: {str(err)}"

github_repo_tool = StructuredTool.from_function(
    func=github_repo,
    name="github-repo",
    description="Get brief information about a GitHub repository based on a natural language query.",
    args_schema=GithubRepoQuery,
    return_direct=True
)