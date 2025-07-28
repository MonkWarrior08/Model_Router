# AI Model Router

An intelligent routing system that automatically selects the optimal AI model and configuration for any given task using LLM-powered analysis.

<img width="936" height="930" alt="Screenshot 2025-07-28 at 8 36 08 PM" src="https://github.com/user-attachments/assets/353c4031-3fa9-4270-86f0-3cb3e88bf1e9" />
<img width="1210" height="984" alt="Screenshot 2025-07-28 at 8 36 22 PM" src="https://github.com/user-attachments/assets/a42ee751-cf20-40cd-b9c9-4a9e3968a1fd" />

## 🎯 What It Does

The AI Model Router analyzes your prompts and automatically:
- **Selects the best AI provider** (OpenAI, Anthropic, Google)
- **Chooses the optimal model** for your specific task type
- **Configures ideal parameters** (temperature, thinking levels, reasoning effort)
- **Applies specialized instructions** tailored to your task category

Instead of manually deciding between GPT-4, Claude, or Gemini for each task, the router uses AI to make these decisions intelligently.

## ✨ Key Features

### 🧠 **Intelligent Task Analysis**
- Uses Gemini Flash 2.0 to analyze and categorize your prompts
- Automatically detects task types: coding, creative writing, tutoring, conversation
- Selects optimal model configurations based on task requirements

### 🔄 **Multi-Provider Support**
- **OpenAI**: GPT models with reasoning effort controls
- **Anthropic**: Claude models with temperature tuning
- **Google**: Gemini models with thinking budgets and temperature controls

### 🎛️ **Dynamic Parameter Optimization**
- **Temperature Control**: Automatically adjusts creativity vs precision
- **Thinking Levels**: Configures reasoning depth for complex tasks
- **Specialized Instructions**: Applies task-specific system prompts

### 💻 **Dual Interface**
- **CLI Tool**: Interactive terminal interface for developers
- **Web App**: Modern streaming chat interface with real-time model selection display

### ⚙️ **Configurable Architecture**
- JSON-based configuration system
- Hot-reloadable settings
- Built-in backup system for configuration changes
- Easy to add new models and task categories

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for at least one provider (OpenAI, Anthropic, or Google)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Model_Router
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env and add your API keys
   ```

4. **run the web interface**
   ```bash
   python app.py
   # Open http://localhost:5000 in your browser
   ```

## 🏗️ Architecture

### Core Components

1. **Router (`router.py`)**
   - LLM-powered prompt analysis
   - Model and parameter selection logic
   - Configuration validation

2. **CLI Interface (`main.py`)**
   - Interactive terminal chat
   - Conversation history management
   - Multi-provider execution

3. **Web Interface (`app.py`)**
   - Flask-based streaming API
   - Real-time model selection display
   - Configuration management UI

4. **Configuration (`model_prompt.json`)**
   - Task category definitions
   - Model mappings and parameters
   - Specialized instruction templates

### Task Categories (Default)

- **Conversational AI**: General chat with Gemini Flash Lite
- **Creative Writing**: Storytelling and poetry with Gemini Flash
- **Professional Coding**: Development tasks with Gemini Pro + thinking budget
- **Tutoring**: Educational content with OpenAI o4-mini + reasoning effort


## 🔧 API Keys Setup

Get your API keys from:

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://makersuite.google.com/app/apikey

Add them to your `.env` file:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## 🙋‍♂️ Support

For questions, suggestions, or issues, please open a GitHub issue or reach out to the maintainers.

