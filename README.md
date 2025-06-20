
## Project Overview

The Model Router is a Python-based tool that solves a key problem for businesses using AI: how to efficiently and effectively leverage the ever-growing ecosystem of Large Language Models (LLMs).

Instead of being locked into a single provider or using a powerful, expensive model for every task, this router analyzes a user's request and dynamically routes it to the most suitable model from OpenAI, Anthropic, or Google, using a specialized, task-specific instruction to ensure the highest quality output.

## How to Run and Test the Demo

### 1. Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt```

**Set Up API Keys:**
Create a `.env` file in the project root. You can copy the template provided:
```bash
cp env_template.txt .env
```
Now, add your API keys for OpenAI, Anthropic, and Google to the `.env` file. The application will warn you if a key is missing, but it will still run if you only want to test models from a provider you have a key for.

### 2. Usage

Run the interactive command-line application:
```bash
python main.py
```

The application starts in a continuous conversation mode. Simply type your prompt and the router will:
1.  Analyze your request to select the best task category.
2.  Choose the most appropriate instruction for that category.
3.  Route to the optimal model for the task.
4.  Stream the response to the console.

You can type `clear` to reset the conversation history or `quit` to exit.

*Note on LLM Usage:* I have leveraged Google's Gemini to accelerate aspects on development for writing boilerplate code and generating documentation.