# Model_Router

An intelligent AI router that automatically selects the best model and instruction for your specific task.

## Features

- **Smart Task Classification**: Automatically categorizes your request into the most appropriate task category
- **Instruction Selection**: Chooses the best instruction prompt for your specific task type
- **Model Optimization**: Routes to the best-performing model for each task category
- **Multiple AI Providers**: Supports OpenAI, Anthropic, and Google models

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root with your API keys:

```bash
# Copy the template
cp env_template.txt .env

# Edit the .env file with your actual API keys
```

Add your API keys to the `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://makersuite.google.com/app/apikey

## Usage

Run the router:

```bash
python main.py
```

The system will:
1. Analyze your request and select the best task category
2. Choose the most appropriate instruction for that category
3. Route to the optimal model for your task

## Task Categories

- **Professional Coding**: Web development, data science, system architecture
- **Advanced Reasoning**: Mathematical, scientific, philosophical problems
- **Fast Reasoning**: Quick math, logic puzzles, decision making
- **Creative Writing**: Poetry, storytelling, copywriting, scriptwriting
- **General QA**: Educational, research, practical advice
- **Fast Simple Task**: Summarization, translation, formatting
- **Conversational AI**: Casual chat, therapeutic, entertainment