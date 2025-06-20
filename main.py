import os
import click
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from router import choose_model

# Load environment variables from .env file
load_dotenv()

# --- API Client Initializations ---
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Model Execution Functions ---

def execute_openai(prompt: str, conversation_history: list, model_name: str = "gpt-4o", temperature: float = 0.3):
    print(f"\n--- Executing with OpenAI ({model_name}) at temperature {temperature} ---\n")
    
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


def execute_claude(prompt: str, conversation_history: list, model_name: str = "claude-3-opus-20240229", temperature: float = 0.3):
    print(f"\n--- Executing with Anthropic ({model_name}) at temperature {temperature} ---\n")
    
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


def execute_gemini(prompt: str, conversation_history: list, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.3):
    print(f"\n--- Executing with Google ({model_name}) at temperature {temperature} ---\n")
    
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


# --- Main CLI Command ---

@click.command()
def main():
    """An intelligent AI router that selects the best model for your task."""
    
    print("🤖 AI Model Router - Continuous Conversation Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'clear' to clear conversation history.\n")
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        # Get user input interactively
        prompt = input("User: ")
        
        # Check for exit commands
        if prompt.lower() in ['quit', 'exit', 'bye', 'q']:
            print("👋 Goodbye! Thanks for chatting!")
            break
        
        # Check for clear command
        if prompt.lower() == 'clear':
            conversation_history = []
            print("🧹 Conversation history cleared!")
            continue
        
        # Skip empty inputs
        if not prompt.strip():
            continue
        
        # 1. Let the router AI choose the best model provider and get its instruction
        router_result = choose_model(prompt)
        model_info = router_result["model_info"]
        instruction = router_result["instruction"]
        category = router_result["category"]
        instruction_subtype = router_result["instruction_subtype"]
        
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