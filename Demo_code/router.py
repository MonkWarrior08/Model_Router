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

    Guidelines for THINKING_LEVEL/REASONING_EFFORT selection:
    
    Choose "low" for:
    - Simple factual questions with clear answers
    - Basic definitions or explanations
    - Straightforward calculations
    - Simple code fixes or syntax questions
    
    Choose "medium" for:
    - Multi-step problems requiring moderate analysis
    - Concept explanations with examples
    - Code debugging with moderate complexity
    - Questions requiring some reasoning but not deep analysis
    
    Choose "high" for:
    - Complex multi-step problems requiring deep analysis
    - Abstract or philosophical questions
    - Complex code architecture or design decisions
    - Problems requiring creative or innovative solutions
    - Questions with multiple valid approaches that need careful consideration
    - Mathematical proofs or complex logical reasoning

    Temperature Guidelines:
    - For simple, factual tasks: use lower temperature
    - For creative, open-ended tasks: use higher temperature
    - Only choose temperature/thinking_level if they exist for the category
    - Use the exact values from the available options

    Respond in this EXACT JSON format:
    {{
        "category": "category_name",
        "instruction_type": "instruction_key",
        "temperature": temperature_value_or_null,
        "thinking_level": thinking_level_value_or_null
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
        
        # Basic validation - use defaults if invalid
        category = result.get("category", "conversational_ai")
        if category not in PROMPT_CONFIG:
            category = "conversational_ai"
            result["category"] = category
        
        category_config = PROMPT_CONFIG[category]
        
        # Validate instruction type
        instruction_type = result.get("instruction_type", "default")
        if instruction_type not in category_config["instructions"]:
            instruction_type = "default"
            result["instruction_type"] = instruction_type
        
        # Clean up temperature and thinking level
        if "temperatures" not in category_config:
            result["temperature"] = None
        if "thinking_levels" not in category_config:
            result["thinking_level"] = None
        
        return result
        
    except:
        # Simple fallback on any error
        return {
            "category": "conversational_ai",
            "instruction_type": "default", 
            "temperature": None,
            "thinking_level": None
        }

def choose_model(prompt: str) -> dict:
    """
    Uses LLM to analyze the user request and selects the model configuration,
    instruction, temperature, and thinking budget.
    """
    
    # 1. Analyze the prompt with LLM
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
            selected_temperature = min(1.0, max(0.0, selected_temperature))
        elif model_info["provider"] == "openai":
            selected_temperature = min(2.0, max(0.0, selected_temperature))
        elif model_info["provider"] == "gemini":
            selected_temperature = min(2.0, max(0.0, selected_temperature))
            
        model_info["temperature"] = selected_temperature
    
    # 5. Apply thinking budget for Gemini 2.5 models if selected
    if selected_thinking_level is not None and "2.5" in model_info["model"]:
        # Only apply if thinking level is an integer (Gemini style)
        if isinstance(selected_thinking_level, int):
            if model_info["model"] == "gemini-2.5-pro":
                selected_thinking_level = max(128, min(32768, selected_thinking_level))
            elif model_info["model"] == "gemini-2.5-flash":
                selected_thinking_level = max(0, min(24576, selected_thinking_level))
            
            model_info["thinking_budget"] = selected_thinking_level
    
    # 6. Apply reasoning parameters for OpenAI o-series models if selected
    if selected_thinking_level is not None and model_info["provider"] == "openai":
        # Only apply if thinking level is a string (OpenAI style)
        if isinstance(selected_thinking_level, str) and selected_thinking_level in ["low", "medium", "high"]:
            model_info["reasoning_effort"] = selected_thinking_level

    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "thinking_level": selected_thinking_level,
        "analysis": analysis
    }