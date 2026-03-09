from google.adk.agents.llm_agent import LlmAgent
import wikipedia
from google.adk.models.lite_llm import LiteLlm
from googlesearch import search
import arxiv

def wikipedia_tool(query : str) -> str:
    """ Searches wikipedia for a given query and returns the summary of the top results.

    Args:
        Query : The search terms that we want to look up in wikipedia
    """

    try:
        summary = wikipedia.summary(query)
        return summary 
    except Exception as e:
        return f"The error could be {e}"

instruction_wikipedia = """
You are a specialized agent and your only taks is to accept a research query and use the wikipedia_tool to retrive relevant infotmation.
"""


wikipedia_agent = LlmAgent(
    name = 'wikipedia_researcher',
    model =LiteLlm(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,        # stick to tool results, less hallucination
        max_tokens=2048,        # prevent cut-off reports
    ),
    description = 'An expert at finidng and summarizing information from wikipedia',
    instruction= instruction_wikipedia,
    tools = [wikipedia_tool],
    output_key = 'wikipedia_result' # stores the output of that session state 

)

def arxiv_tool(query : str) -> str:
    """ Searches Arxiv archives to get the most relevant papers present 

    Args:
        Query: The terms that yeilds the present papers
    """

    try:
        client = arxiv.Client()

        arxiv_search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []

        for entry in client.results(arxiv_search):
            results.append(f"Title: {entry.title}\nSummary: {entry.summary}\nURL: {entry.entry_id}")

        if not results:
            return f"No academic papers found for '{query}'"

        return "\n....\n".join(results)
    except Exception as e:
        return f"The exception that occurs is {e}"



instruction_arxiv = """
You are a specialized agent where the only job is to search arXiv for academic papers on a given topic. Use the arxiv_tool to find and summarize relevant papers.
"""
arxiv_agent = LlmAgent(
    name = 'arxiv_researcher',
    model =LiteLlm(
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,        # stick to tool results, less hallucination
        max_tokens=2048,        # prevent cut-off reports
    ),
    description = 'An expert at finidng and summarizing information from arxiv',
    instruction= instruction_arxiv,
    tools = [arxiv_tool],
    output_key = 'arxiv_result'
)

from googlesearch import search

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

google_search_agent = LlmAgent(
    name = "web_crawler",
    model =LiteLlm(
        model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.2,        # stick to tool results, less hallucination
        max_tokens=1024,        # prevent cut-off reports
    ),
    description = 'An expert at searching the web to find relevant, up-to-date information on a topic using Google search',
    instruction = """ 
    Your only task is to accept the research topic and use the google_search tool to find relevant URLs. Return a brief summary of the results.
    """,
    tools = [google_search],
    output_key = 'web_result'
)


import time 
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

from google.adk.models.lite_llm import LiteLlm

writer_agent = LlmAgent(
    name='report_writer',
    model=LiteLlm(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,        # stick to tool results, less hallucination
        max_tokens=8192,        # prevent cut-off reports
    ),
    description='An expert at writing content to a file.',
    instruction="""
    You are a report writer. You will receive research from three sources stored as:
    - {wikipedia_result}: background information
    - {arxiv_result}: academic papers  
    - {web_result}: recent web articles

    Synthesize ALL THREE into a structured report and save it using report_writer_tool.
    Filename format: <topic>_report.txt
    """,
    tools=[report_writer_tool]
)


Controller_instruction = """
You are a research assistant who orchestrates a team of specialist agents to produce a high-quality research report. 
Your primary role is to delegate tasks, synthesize the results, and ensure the final report is well-structured.

Your specialist team consists of:
- `wikipedia_researcher`: Use this agent to get general background information and a high-level overview.
- `arxiv_researcher`: Use this agent to find relevant academic papers and their summaries.
- `web_crawler`: Use this agent to find up-to-date information and supplementary context from the web.

Your workflow must be as follows:
1.  First, call all three specialist research agents (`wikipedia_researcher`, `arxiv_researcher`, and `web_crawler`) to gather a comprehensive set of information on the topic.
2.  Once all information has been gathered, you must personally synthesize the content from all three sources into a single, coherent summary.
3.  Finally, call the `report_writer` agent with the synthesized content and a filename based on the topic (e.g., black_holes_report.txt).
"""

from google.adk.tools.agent_tool import AgentTool
from google.adk.agents import SequentialAgent


# The aboove needs a better orchistration model , free models can handle orchistration that easily , the above makes it model specific calling 
# The below code runs them sequentually , not the best but does the job 
root_agent = SequentialAgent(
    name='Controller',
    sub_agents=[wikipedia_agent, arxiv_agent, google_search_agent, writer_agent]
)
