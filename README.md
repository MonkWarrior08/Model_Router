
## Project Overview

Model Router is a Python-based tool that solves a key problem for businesses using AI: how to efficiently and effectively leverage the ever-growing ecosystem of Large Language Models (LLMs).

Instead of being locked into a single provider or using a powerful, expensive model for every task, this router intelligently analyzes each user request through multiple dimensions:

1. **Task Classification**: Categorizes requests into specialized domains (professional coding, creative writing, conversational AI, or tutoring)
2. **Instruction Selection**: Chooses between default and specialized instructions within each category (e.g., poetry-specific prompts for creative writing, or Socratic method for tutoring)
3. **Complexity Analysis**: Determines the thinking level (low/medium/high) based on prompt complexity and keywords
4. **Dynamic Parameter Optimization**: Automatically configures model-specific parameters including temperature, thinking budgets for Gemini 2.5 models, and reasoning effort for OpenAI's o-series models

The router then routes to the optimal model from OpenAI, Anthropic, or Google with precisely tuned parameters to ensure the highest quality output for each specific task type.

## How to Run and Test the Demo

### 1. Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

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

Upon launching the application, you enter a continuous conversation mode. When you input your prompt, the router performs the following steps:
1.  Task categorization: It identifies the most suitable task category for your request.
2.  Instruction selection: It picks the most fitting instruction tailored to the selected category.
3.  Model routing: It directs your task to the most optimal model available.
4.  Response streaming: It displays the model's response directly in the console.

You can type `clear` to reset the conversation history or `quit` to exit.

*Note on LLM Usage:* I have leveraged Google's Gemini to accelerate aspects on development for writing boilerplate code and generating documentation.
