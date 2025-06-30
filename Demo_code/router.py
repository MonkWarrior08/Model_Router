import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# We use a cheap and fast model for the routing itself.
# Gemini Flash 2.0 Flash Lite is a good choice for fast classification tasks.
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load our benchmark data
with open('model_prompt.json', 'r') as f:
    PROMPT_CONFIG = json.load(f)

TASK_CATEGORIES = list(PROMPT_CONFIG.keys())

def analyze_prompt_with_llm(prompt: str) -> dict:
    """
    Uses LLM to analyze the user prompt and determine:
    1. Task category
    2. Instruction type (default or specialized)
    3. Thinking level (low, medium, high)
    """
    
    # Build available options for each category
    category_options = {}
    for category, config in PROMPT_CONFIG.items():
        category_options[category] = {
            "instructions": list(config["instructions"].keys()),
            "thinking_levels": ["low", "medium", "high"]
        }
    
    system_prompt = f"""
    You are an expert task analyzer. Analyze the user's prompt and make three decisions:

    1. CATEGORY: Which task category best fits this prompt?
    2. INSTRUCTION_TYPE: Which instruction approach would work best?
    3. THINKING_LEVEL: How much complexity/creativity does this task require?

    Available categories and their instruction types:
    
    professional_coding:
    - default: Expert software engineer for general coding
    - web_development: Full-stack web developer for web applications
    
    creative_writing:
    - default: Master storyteller and creative writer
    - poem: Specialized poet for verses and poetry
    
    conversational_ai:
    - default: Engaging conversational AI with personality
    - therapeutic: Supportive partner for emotional intelligence
    
    tutor:
    - default: Educational expert explaining complex concepts
    - socratic: Tutor using Socratic method with guiding questions

    Thinking levels (complexity/creativity needed):
    - low: Simple, factual, straightforward tasks
    - medium: Moderate complexity, balanced approach
    - high: Complex, creative, multi-step, or innovative tasks

    Respond in this EXACT JSON format:
    {{
        "category": "category_name",
        "instruction_type": "instruction_name", 
        "thinking_level": "level_name",
        "reasoning": "Brief explanation of your choices"
    }}

    User prompt: {prompt}
    """

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.0-flash-lite-preview-02-05',
            contents=system_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=200
            )
        )

        # Parse the JSON response
        result_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if result_text.startswith('```json'):
            result_text = result_text.replace('```json', '').replace('```', '').strip()
        elif result_text.startswith('```'):
            result_text = result_text.replace('```', '').strip()
            
        result = json.loads(result_text)
        
        # Validate the results
        category = result.get("category")
        instruction_type = result.get("instruction_type")
        thinking_level = result.get("thinking_level")
        
        # Validate category
        if category not in PROMPT_CONFIG:
            print(f"⚠️ Router: Invalid category '{category}'. Defaulting to 'conversational_ai'.")
            category = "conversational_ai"
            result["category"] = category
        
        # Validate instruction type for the category
        available_instructions = list(PROMPT_CONFIG[category]["instructions"].keys())
        if instruction_type not in available_instructions:
            print(f"⚠️ Router: Invalid instruction type '{instruction_type}' for category '{category}'. Defaulting to 'default'.")
            instruction_type = "default"
            result["instruction_type"] = instruction_type
            
        # Validate thinking level
        if thinking_level not in ["low", "medium", "high"]:
            print(f"⚠️ Router: Invalid thinking level '{thinking_level}'. Defaulting to 'medium'.")
            thinking_level = "medium"
            result["thinking_level"] = thinking_level
        
        result["valid"] = True
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Router: Failed to parse LLM response as JSON: {e}")
        print(f"Raw response: {response.text}")
        return {
            "category": "conversational_ai",
            "instruction_type": "default", 
            "thinking_level": "medium",
            "reasoning": "Fallback due to parsing error",
            "valid": False
        }
    except Exception as e:
        print(f"⚠️ Router: Error analyzing prompt with LLM: {e}")
        return {
            "category": "conversational_ai",
            "instruction_type": "default",
            "thinking_level": "medium", 
            "reasoning": "Fallback due to error",
            "valid": False
        }

def choose_model(prompt: str) -> dict:
    """
    Uses LLM to analyze the user request and selects the model configuration,
    instruction, temperature, and thinking budget.
    """
    print("🧠 Router: Analyzing prompt with LLM...")

    # 1. Analyze the prompt with LLM to get category, instruction type, and thinking level
    analysis = analyze_prompt_with_llm(prompt)
    category = analysis["category"]
    instruction_type = analysis["instruction_type"]
    thinking_level = analysis["thinking_level"]
    
    category_config = PROMPT_CONFIG[category]
    
    # 2. Get the model configuration
    model_info = category_config["model"].copy()

    # 3. Get the selected instruction
    selected_instruction = category_config["instructions"][instruction_type]
    
    # 4. Select the temperature based on the thinking level, if available
    if "temperatures" in category_config:
        selected_temperature = category_config["temperatures"][thinking_level]
        
        # Clamp temperature values to valid ranges for each provider
        if model_info["provider"] == "claude":
            # Claude accepts temperature 0-1
            selected_temperature = min(1.0, max(0.0, selected_temperature))
        elif model_info["provider"] == "openai":
            # OpenAI accepts temperature 0-2
            selected_temperature = min(2.0, max(0.0, selected_temperature))
        elif model_info["provider"] == "gemini":
            # Gemini accepts temperature 0-2
            selected_temperature = min(2.0, max(0.0, selected_temperature))
            
        model_info["temperature"] = selected_temperature
    
    # 5. Select and apply thinking budget if available (only for Gemini 2.5 models)
    if "thinking_levels" in category_config and "2.5" in model_info["model"]:
        selected_level_value = category_config["thinking_levels"][thinking_level]
        if selected_level_value is not None:
            # Apply model-specific thinking budget constraints for Gemini 2.5 models
            if model_info["model"] == "gemini-2.5-pro":
                # 2.5 Pro: 128 to 32768, cannot disable thinking
                selected_level_value = max(128, min(32768, selected_level_value))
            elif model_info["model"] == "gemini-2.5-flash":
                # 2.5 Flash: 0 to 24576
                selected_level_value = max(0, min(24576, selected_level_value))
            
            model_info["thinking_budget"] = selected_level_value
    
    # 6. Select and apply reasoning parameters for OpenAI o-series models
    if "reasoning_levels" in category_config and model_info["provider"] == "openai":
        reasoning_effort = category_config["reasoning_levels"][thinking_level]
        if reasoning_effort is not None:
            model_info["reasoning_effort"] = reasoning_effort
            
    # --- Print summary ---
    print(f"✅ Router: Selected category: '{category}'")
    print(f"📝 Router: Selected instruction: '{instruction_type}'")

    if "temperature" in model_info:
        print(f"🌡️ Router: Applying temperature: {model_info['temperature']}")
    
    if "thinking_budget" in model_info and model_info["thinking_budget"] is not None:
        print(f"💡 Router: Applying thinking budget: {model_info['thinking_budget']}")
    elif "reasoning_effort" in model_info:
        print(f"🧠 Router: Applying reasoning effort: {model_info['reasoning_effort']}")
    elif "thinking_levels" in category_config or "reasoning_levels" in category_config:
        print(f"🎯 Router: Selected thinking level: '{thinking_level}'")

    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "instruction_subtype": instruction_type,
        "thinking_level": thinking_level,
        "analysis": analysis
    }