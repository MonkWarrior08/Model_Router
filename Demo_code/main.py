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


# --- API Client Initializations ---
openai_client = openai.OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
genai.configure(api_key=google_key)

# --- Model Execution Functions ---

def execute_openai(prompt: str, conversation_history: list, model_name: str, temperature: float = None):
    
    temp_msg = f"at temperature {temperature}" if temperature is not None else "at default temperature"
    print(f"\n--- Executing with OpenAI ({model_name}) {temp_msg} ---\n")
    
    try:
        # Build messages with conversation history
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        api_params = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            api_params["temperature"] = temperature

        response = openai_client.chat.completions.create(**api_params)
        
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


def execute_claude(prompt: str, conversation_history: list, model_name: str, temperature: float = None):
    
    temp_msg = f"at temperature {temperature}" if temperature is not None else "at default temperature"
    print(f"\n--- Executing with Anthropic ({model_name}) {temp_msg} ---\n")
    
    try:
        # Build messages with conversation history
        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        api_params = {
            "model": model_name,
            "max_tokens": 2048,
            "messages": messages,
        }
        if temperature is not None:
            api_params["temperature"] = temperature

        full_response = ""
        with anthropic_client.messages.stream(**api_params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print()
        
        return full_response
    except Exception as e:
        error_msg = f"❌ Anthropic Error: {str(e)}"
        print(error_msg)
        return error_msg


def execute_gemini(prompt: str, conversation_history: list, model_name: str, temperature: float = None, thinking_budget: int = None):
    
    temp_msg = f"at temperature {temperature}" if temperature is not None else "at default temperature"
    thinking_msg = f"with thinking_budget: {thinking_budget}" if thinking_budget is not None else ""
    print(f"\n--- Executing with Google ({model_name}) {temp_msg} {thinking_msg}---\n")
    
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
        
        # Set up generation config
        gen_config_params = {}
        if temperature is not None:
            gen_config_params["temperature"] = temperature
        
        if thinking_budget is not None:
            gen_config_params["thinking_config"] = genai.types.ThinkingConfig(thinking_budget=thinking_budget)

        gen_config = genai.types.GenerationConfig(**gen_config_params)

        # Send the current prompt
        response = chat.send_message(
            prompt, 
            stream=True,
            generation_config=gen_config
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
    
    print("AI Model Router")
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
        
        # Prepare parameters that are common to all execution functions
        params = {
            "prompt": full_prompt,
            "conversation_history": conversation_history,
            "model_name": model_info["model"],
        }
        # Add temperature if it's specified for the selected model config
        if "temperature" in model_info:
            params["temperature"] = model_info["temperature"]

        if model_info["provider"] == "openai":
            ai_response = execute_openai(**params)
        elif model_info["provider"] == "claude":
            ai_response = execute_claude(**params)
        elif model_info["provider"] == "gemini":
            # Add Gemini-specific parameters
            if "thinking_budget" in model_info:
                params["thinking_budget"] = model_info.get("thinking_budget")
            ai_response = execute_gemini(**params)
        else:
            print(f"Error: Unknown model provider '{model_info['provider']}'.")
        
        # Add the current exchange to conversation history
        conversation_history.append({"role": "user", "content": prompt})
        if ai_response:
            conversation_history.append({"role": "assistant", "content": ai_response})
        
        print()  # Add a blank line for better readability


if __name__ == "__main__":
    main()