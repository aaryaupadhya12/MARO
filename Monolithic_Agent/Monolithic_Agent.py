import wikipedia
import arxiv
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.agents.llm_agent import LlmAgent
import os 
from dotenv import load_dotenv
from googlesearch import search
from google.adk.models.lite_llm import LiteLlm


def google_search(query: str) -> str:
    """
    Searches Google for recent web articles and returns a list of URLs.
    
    Args:
        query (str): The search term to look up on Google.
    """
    try:
        results = []
        for url in search(query, num_results=5, lang="en"):
            results.append(url)
        if not results:
            return f"No results found for '{query}'."
        return "\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def wikipedia_tool(query: str) -> str:
    """
    Searches Wikipedia for a given query and returns a summary of the top result.

    Args:
        query (str): The search term to look up on Wikipedia.
    """
    try:
        # The summary method automatically finds the best-matching page
        # and returns a summary of it.
        summary = wikipedia.summary(query)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle cases where a query is ambiguous (e.g., "Java")
        return f"The query '{query}' is ambiguous. Please be more specific. Options: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        # Handle cases where the page does not exist
        return f"Sorry, I could not find a Wikipedia page for '{query}'."
    except Exception as e:
        return f"An unexpected error occurred while searching Wikipedia: {e}"


def arxiv_tool(query: str) -> str:
    """
    Searches the arXiv repository for academic papers matching a query.

    Args:
        query (str): The topic to search for academic papers on.
    """
    try:
        # Create a client to interact with the arXiv API
        client = arxiv.Client()

        # Define the search parameters
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        # Use the client to execute the search and get the results
        for result in client.results(search):
            results.append(f"Title: {result.title}\nSummary: {result.summary}\nURL: {result.entry_id}")
            
        if not results:
            return f"No academic papers found on arXiv for the query '{query}'."
            
        return "\n---\n".join(results)
        
    except Exception as e:
        return f"An unexpected error occurred while searching arXiv: {e}"
        
def report_writer_tool(content: str, filename: str) -> str:
    """
    Writes the given content to a local file. Appends if the file already exists.

    Args:
        content (str): The text content to write to the file.
        filename (str): The name of the file to save the content in (e.g., 'report.txt').
    """
    try:
        # Use 'a' for append mode. This will create the file if it doesn't exist,
        # or add to the end of it if it does.
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
        return f"Successfully appended content to {filename}."
    except Exception as e:
        return f"An error occurred while writing to file: {e}"
        
instruction = """
You are a helpful and diligent research assistant. Your goal is to produce a 
detailed research report on a given topic.

You must follow these steps in order:
1. CALL wikipedia_tool with the topic. If not found, try broader terms like 
   "Variational Autoencoder" or "Introspective VAE".
2. CALL arxiv_tool with the topic to get academic papers. Also search with 
   broader terms if needed (e.g. "soft introspective VAE medical imaging").
3. CALL google_search with the topic.
4. Synthesize ALL gathered information into a detailed report covering:
   - Background and motivation
   - Technical approach
   - Key findings from papers with their URLs
   - Applications and results
   - Limitations and future work
5. CALL report_writer_tool to save to 'report.txt'.

If Wikipedia returns nothing, compensate by extracting MORE detail from 
the arXiv papers. Never write a shallow report just because one tool failed.
"""

from google.adk.models.lite_llm import LiteLlm
from googlesearch import search

root_agent = LlmAgent(
    name='research_assistant',
    model=LiteLlm(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,        # stick to tool results, less hallucination
        max_tokens=8192,        # prevent cut-off reports
    ),
    description='A research assistant that gathers information from Wikipedia, arXiv, and Google Search to create a summary report on a given topic.',
    instruction=instruction,
    tools=[
        wikipedia_tool,
        arxiv_tool,
        google_search,          # plain function, not GoogleSearchTool
        report_writer_tool
    ],
)
