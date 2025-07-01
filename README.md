## Project Overview

Model Router is a Python application that simplifies how organizations interact with AI by dynamically selecting the most suitable Large Language Model (LLM) for each specific task. This eliminates the common dilemma of choosing between cost efficiency and performance when working with multiple AI providers.

The system employs advanced prompt analysis to route requests through a sophisticated decision-making pipeline:

1. **Intelligent Task Classification**: Automatically categorizes user requests into specialized domains including professional coding, creative writing, conversational AI, and educational tutoring
2. **Context-Aware Instruction Selection**: Dynamically chooses between default and specialized instruction sets (e.g., Socratic method for tutoring, poetry-specific prompts for creative writing)
3. **Sophisticated Complexity Assessment**: Analyzes prompt complexity and keywords to determine optimal thinking levels (low/medium/high) and temperature settings for enhanced reasoning

The router intelligently directs requests to the optimal model from OpenAI, Anthropic, or Google, ensuring maximum quality and efficiency for each unique task type while minimizing costs.

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

You can type `clear` to reset the conversation history or `quit` to exit the application.

*Development Note:* This project leverages Google's Gemini models for development acceleration, including code generation and documentation creation.
