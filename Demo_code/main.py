import os
import click
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from router import choose_model

# Load environment variables from .env file
load_dotenv()

# Check if API keys are loaded
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if not openai_key or openai_key == "your_openai_api_key_here":
    print("⚠️ Warning: OPENAI_API_KEY not found or not set properly in .env file")
if not anthropic_key or anthropic_key == "your_anthropic_api_key_here":
    print("⚠️ Warning: ANTHROPIC_API_KEY not found or not set properly in .env file")
if not google_key or google_key == "your_google_api_key_here":
    print("⚠️ Warning: GOOGLE_API_KEY not found or not set properly in .env file")

# --- API Client Initializations ---
openai_client = openai.OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
genai.configure(api_key=google_key)

# --- Model Execution Functions ---

def execute_openai(prompt: str, conversation_history: list, model_name: str = "gpt-4o", temperature: float = 0.3):
    if not openai_key or openai_key == "your_openai_api_key_here":
        return "❌ Error: OpenAI API key not configured. Please add your OPENAI_API_KEY to the .env file."
    
    print(f"\n--- Executing with OpenAI ({model_name}) at temperature {temperature} ---\n")
    
    try:
        # Build messages with conversation history
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end='', flush=True)
                full_response += content
        print()
        
        return full_response
    except Exception as e:
        error_msg = f"❌ OpenAI Error: {str(e)}"
        print(error_msg)
        return error_msg


def execute_claude(prompt: str, conversation_history: list, model_name: str = "claude-3-opus-20240229", temperature: float = 0.3):
    if not anthropic_key or anthropic_key == "your_anthropic_api_key_here":
        return "❌ Error: Anthropic API key not configured. Please add your ANTHROPIC_API_KEY to the .env file."
    
    print(f"\n--- Executing with Anthropic ({model_name}) at temperature {temperature} ---\n")
    
    try:
        # Build messages with conversation history
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        full_response = ""
        with anthropic_client.messages.stream(
            model=model_name,
            max_tokens=2048,
            temperature=temperature,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print()
        
        return full_response
    except Exception as e:
        error_msg = f"❌ Anthropic Error: {str(e)}"
        print(error_msg)
        return error_msg


def execute_gemini(prompt: str, conversation_history: list, model_name: str = "gemini-2.5-pro", temperature: float = 0.3):
    if not google_key or google_key == "your_google_api_key_here":
        return "❌ Error: Google API key not configured. Please add your GOOGLE_API_KEY to the .env file."
    
    print(f"\n--- Executing with Google ({model_name}) at temperature {temperature} ---\n")
    
    try:
        # Build conversation history for Gemini
        chat = genai.GenerativeModel(model_name).start_chat(history=[])
        
        # Add conversation history
        for msg in conversation_history:
            if msg["role"] == "user":
                chat.send_message(msg["content"])
            else:
                # For assistant messages, we need to simulate the response
                # This is a simplified approach - in practice you might want to store actual responses
                pass
        
        # Send the current prompt
        response = chat.send_message(
            prompt, 
            stream=True,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        
        full_response = ""
        for chunk in response:
            print(chunk.text, end="", flush=True)
            full_response += chunk.text
        print()
        
        return full_response
    except Exception as e:
        error_msg = f"❌ Google Error: {str(e)}"
        print(error_msg)
        return error_msg


# --- Main CLI Command ---

@click.command()
def main():
    """AI router that selects the best model for your task."""
    
    print("🤖 AI Model Router")
    print("'quit' to end the conversation.")
    print("'clear' to clear conversation history.\n")
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        # Get user input interactively
        prompt = input("User: ")
        
        # Check for exit commands
        if prompt.lower() in ['quit']:
            print("Ending conversation...")
            break
        
        # Check for clear command
        if prompt.lower() == 'clear':
            conversation_history = []
            print("History cleared")
            continue
        
        # Skip empty inputs
        if not prompt.strip():
            continue
        
        # 1. Let the router AI choose the best model provider and get its instruction
        router_result = choose_model(prompt)
        model_info = router_result["model_info"]
        instruction = router_result["instruction"]
       
        
        # Combine the instruction with the user's prompt
        full_prompt = f"{instruction}\n\nUser: {prompt}"
        
        # 2. A simple 'factory' to call the correct execution function with the full prompt
        ai_response = ""
        if model_info["provider"] == "openai":
            ai_response = execute_openai(full_prompt, conversation_history, model_info["model"], model_info["temperature"])
        elif model_info["provider"] == "claude":
            ai_response = execute_claude(full_prompt, conversation_history, model_info["model"], model_info["temperature"])
        elif model_info["provider"] == "gemini":
            ai_response = execute_gemini(full_prompt, conversation_history, model_info["model"], model_info["temperature"])
        else:
            print(f"Error: Unknown model provider '{model_info['provider']}'.")
        
        # Add the current exchange to conversation history
        conversation_history.append({"role": "user", "content": prompt})
        if ai_response:
            conversation_history.append({"role": "assistant", "content": ai_response})
        
        print()  # Add a blank line for better readability


if __name__ == "__main__":
    main()