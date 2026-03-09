# Multi-Agent Research System

A research pipeline that automatically gathers, synthesizes, and saves a report on any topic using a team of specialized AI agents.

---

## How It Works

When you enter a research topic, four agents run in sequence:

```
User Input
    │
    ▼
wikipedia_researcher  →  fetches background info from Wikipedia
    │
    ▼
arxiv_researcher      →  finds relevant academic papers
    │
    ▼
web_crawler           →  searches Google for recent articles
    │
    ▼
report_writer         →  synthesizes all results and saves to a .txt file
```

Each agent passes its output to the next via session state. The final report is saved locally as `<topic>_report.txt`.

---

## Project Structure

```
Multi_Agent_Systems/
├── agent.py          # All agent and tool definitions
├── requirements.txt  # Dependencies
└── README.md
```

---

## Setup

**1. Install dependencies**
```bash
pip install google-adk litellm wikipedia arxiv googlesearch-python
```

**2. Set API keys**
```powershell
# PowerShell
$env:GROQ_API_KEY = "your_groq_key"
$env:GOOGLE_API_KEY = "your_gemini_key"
```

```bash
# Mac/Linux
export GROQ_API_KEY="your_groq_key"
export GOOGLE_API_KEY="your_gemini_key"
```

**3. Run**
```bash
adk run agent.py
```

---

## Agents

| Agent | Model | Job |
|---|---|---|
| `wikipedia_researcher` | groq/llama-3.3-70b-versatile | Fetches Wikipedia summary |
| `arxiv_researcher` | groq/llama-4-scout-17b-16e-instruct | Finds academic papers |
| `web_crawler` | groq/llama-4-maverick-17b-128e-instruct | Searches Google for URLs |
| `report_writer` | groq/llama-3.3-70b-versatile | Synthesizes and saves report |

The orchestration controller runs on `gemini/gemini-2.0-flash` which has a separate rate limit pool from Groq.

---

## Usage

Run the agent and type a research topic:

```
[user]: global warming
```

The pipeline runs automatically and saves a file called `global_warming_report.txt` in the current directory.

---

## Output

The saved report follows this structure:

```
Title
Introduction / Background
Key Concepts
Academic Research Insights
Recent Developments
Challenges and Open Problems
Conclusion
```

---

## API Keys

- **Groq** — https://console.groq.com
- **Gemini** — https://aistudio.google.com/app/apikey
