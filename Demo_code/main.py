import os
import click
import openai
import anthropic
from google import genai
from google.genai import types
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
genai_client = genai.Client(api_key=google_key)

# --- Model Execution Functions ---

def execute_openai(prompt: str, conversation_history: list, model_name: str, temperature: float = None, reasoning_effort: str = None):
    
    temp_msg = f"at temperature {temperature}" if temperature is not None else "at default temperature"
    reasoning_msg = f"with reasoning_effort: {reasoning_effort}" if reasoning_effort is not None else ""
    print(f"\n--- Executing with OpenAI ({model_name}) {temp_msg} {reasoning_msg} ---\n")
    
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
        
        # Add reasoning parameters for o-series models (o4-mini, o3, etc.)
        if reasoning_effort is not None and ("o4" in model_name or "o3" in model_name or "o1" in model_name):
            api_params["reasoning_effort"] = reasoning_effort

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
        contents = []
        for msg in conversation_history:
            contents.append(f"{msg['role']}: {msg['content']}")
        contents.append(f"user: {prompt}")
        
        # Combine all conversation into a single prompt
        full_conversation = "\n".join(contents)
        
        # Set up generation config
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        
        # Set up thinking config if thinking_budget is specified
        thinking_config = None
        if thinking_budget is not None:
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
        
        # Create generation config
        config = types.GenerateContentConfig(
            **config_params,
            thinking_config=thinking_config
        ) if config_params or thinking_config else None

        # Generate content using the new API
        response = genai_client.models.generate_content(
            model=model_name,
            contents=full_conversation,
            config=config
        )
        
        print(response.text, end="", flush=True)
        print()
        
        return response.text
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
        
        # Add reasoning parameters for OpenAI o-series models
        if "reasoning_effort" in model_info:
            params["reasoning_effort"] = model_info["reasoning_effort"]

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