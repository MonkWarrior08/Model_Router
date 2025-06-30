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
    3. Specific temperature and thinking level values based on what's available
    """
    
    # Build available options for each category dynamically
    category_details = {}
    for category, config in PROMPT_CONFIG.items():
        category_info = {
            "instructions": {}
        }
        
        # Add instruction descriptions
        for inst_key, inst_value in config["instructions"].items():
            category_info["instructions"][inst_key] = inst_value
        
        # Add temperature options if available
        if "temperatures" in config:
            category_info["temperatures"] = config["temperatures"]
        
        # Add thinking level options if available
        if "thinking_levels" in config:
            category_info["thinking_levels"] = config["thinking_levels"]
            
        category_details[category] = category_info
    
    system_prompt = f"""
    You are an expert task analyzer. Analyze the user's prompt and determine the best configuration.

    Available categories and their options:
    {json.dumps(category_details, indent=2)}

    Based on the user's prompt, choose:
    1. CATEGORY: Which task category best fits this prompt?
    2. INSTRUCTION_TYPE: Which instruction approach would work best?
    3. TEMPERATURE: If the category has temperature options, choose the most appropriate value (or null if not available)
    4. THINKING_LEVEL: If the category has thinking_levels, choose the most appropriate value (or null if not available)

    Guidelines:
    - For simple, factual tasks: use lower temperature and thinking levels
    - For creative, complex tasks: use higher temperature and thinking levels
    - Only choose temperature/thinking_level if they exist for the category
    - Use the exact values from the available options

    Respond in this EXACT JSON format:
    {{
        "category": "category_name",
        "instruction_type": "instruction_key",
        "temperature": temperature_value_or_null,
        "thinking_level": thinking_level_value_or_null,
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
        temperature = result.get("temperature")
        thinking_level = result.get("thinking_level")
        
        # Validate category
        if category not in PROMPT_CONFIG:
            print(f"⚠️ Router: Invalid category '{category}'. Defaulting to 'conversational_ai'.")
            category = "conversational_ai"
            result["category"] = category
        
        category_config = PROMPT_CONFIG[category]
        
        # Validate instruction type for the category
        available_instructions = list(category_config["instructions"].keys())
        if instruction_type not in available_instructions:
            print(f"⚠️ Router: Invalid instruction type '{instruction_type}' for category '{category}'. Defaulting to 'default'.")
            instruction_type = "default"
            result["instruction_type"] = instruction_type
        
        # Validate temperature if category has temperature options
        if "temperatures" in category_config and temperature is not None:
            available_temps = list(category_config["temperatures"].values())
            if temperature not in available_temps:
                print(f"⚠️ Router: Invalid temperature '{temperature}' for category '{category}'. Using default.")
                result["temperature"] = None
        elif "temperatures" not in category_config:
            # Category doesn't support temperature, set to None
            result["temperature"] = None
            
        # Validate thinking level if category has thinking level options
        if "thinking_levels" in category_config and thinking_level is not None:
            available_thinking = list(category_config["thinking_levels"].values())
            if thinking_level not in available_thinking:
                print(f"⚠️ Router: Invalid thinking level '{thinking_level}' for category '{category}'. Using default.")
                result["thinking_level"] = None
        elif "thinking_levels" not in category_config:
            # Category doesn't support thinking levels, set to None
            result["thinking_level"] = None
        
        result["valid"] = True
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Router: Failed to parse LLM response as JSON: {e}")
        print(f"Raw response: {response.text}")
        return {
            "category": "conversational_ai",
            "instruction_type": "default", 
            "temperature": None,
            "thinking_level": None,
            "reasoning": "Fallback due to parsing error",
            "valid": False
        }
    except Exception as e:
        print(f"⚠️ Router: Error analyzing prompt with LLM: {e}")
        return {
            "category": "conversational_ai",
            "instruction_type": "default",
            "temperature": None,
            "thinking_level": None, 
            "reasoning": "Fallback due to error",
            "valid": False
        }

def choose_model(prompt: str) -> dict:
    """
    Uses LLM to analyze the user request and selects the model configuration,
    instruction, temperature, and thinking budget.
    """
    print("🧠 Router: Analyzing prompt with LLM...")

    # 1. Analyze the prompt with LLM to get category, instruction type, temperature, and thinking level
    analysis = analyze_prompt_with_llm(prompt)
    category = analysis["category"]
    instruction_type = analysis["instruction_type"]
    selected_temperature = analysis["temperature"]
    selected_thinking_level = analysis["thinking_level"]
    
    category_config = PROMPT_CONFIG[category]
    
    # 2. Get the model configuration
    model_info = category_config["model"].copy()

    # 3. Get the selected instruction
    selected_instruction = category_config["instructions"][instruction_type]
    
    # 4. Apply the selected temperature if available
    if selected_temperature is not None:
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
    
    # 5. Apply thinking budget for Gemini 2.5 models if selected
    if selected_thinking_level is not None and "2.5" in model_info["model"]:
        # Apply model-specific thinking budget constraints for Gemini 2.5 models
        if model_info["model"] == "gemini-2.5-pro":
            # 2.5 Pro: 128 to 32768, cannot disable thinking
            selected_thinking_level = max(128, min(32768, selected_thinking_level))
        elif model_info["model"] == "gemini-2.5-flash":
            # 2.5 Flash: 0 to 24576
            selected_thinking_level = max(0, min(24576, selected_thinking_level))
        
        model_info["thinking_budget"] = selected_thinking_level
    
    # 6. Apply reasoning parameters for OpenAI o-series models if selected
    if selected_thinking_level is not None and model_info["provider"] == "openai":
        if selected_thinking_level in ["low", "medium", "high"]:
            model_info["reasoning_effort"] = selected_thinking_level
            
    # --- Print summary ---
    print(f"✅ Router: Selected category: '{category}'")
    print(f"📝 Router: Selected instruction: '{instruction_type}'")

    if "temperature" in model_info:
        print(f"🌡️ Router: Applying temperature: {model_info['temperature']}")
    
    if "thinking_budget" in model_info and model_info["thinking_budget"] is not None:
        print(f"💡 Router: Applying thinking budget: {model_info['thinking_budget']}")
    elif "reasoning_effort" in model_info:
        print(f"🧠 Router: Applying reasoning effort: {model_info['reasoning_effort']}")
    elif selected_thinking_level is not None:
        print(f"🎯 Router: Selected thinking level: '{selected_thinking_level}'")

    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "instruction_subtype": instruction_type,
        "thinking_level": selected_thinking_level,
        "analysis": analysis
    }