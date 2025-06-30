import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# We use a cheap and fast model for the routing itself.
# Gemini Flash 2.0 Flash Lite is a good choice for fast classification tasks.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load our benchmark data
with open('model_prompt.json', 'r') as f:
    PROMPT_CONFIG = json.load(f)

TASK_CATEGORIES = list(PROMPT_CONFIG.keys())

def categorize_task(prompt: str) -> dict:
    """
    Categorizes the user request into a task category.
    """
    system_prompt = f"""
    You are an expert task classifier. Your job is to analyze a user's prompt and
    classify it into one of the following task categories.
    Respond with ONLY the category name and nothing else.

    Available categories are: {', '.join(TASK_CATEGORIES)}

    Examples:
    - "Write a poem about love" → creative_writing
    - "Build a web application" → professional_coding
    - "Can you explain this to me?" → tutor
    - "Let's have a casual chat" → conversational_ai
    """


        # Create the model instance
    model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        
        # Combine system and user prompt for Gemini
    full_prompt = f"{system_prompt}\n\nUser prompt: {prompt}"
        
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=50
        )
    )

    category = response.text.strip()
        
        # Validate the category
    if category in PROMPT_CONFIG:
        return {"category": category, "valid": True}
    else:
        print(f"⚠️ Router: Could not classify category '{category}'. Defaulting to 'conversational_ai'.")
        return {"category": "conversational_ai", "valid": False}
            

def select_instruction_for_category(category: str, prompt: str) -> str:
    """
    Selects the appropriate instruction for the given category based on prompt content.
    Choose between 'default' and one specialized prompt per category.
    """
    instruction_set = PROMPT_CONFIG[category]["instructions"]
    
    prompt_lower = prompt.lower()
    
    # It checks for keywords associated with specialized instructions.
    for instruction_type in instruction_set:
        if instruction_type == "default":
            continue
            
        keywords = []
        if category == "creative_writing" and instruction_type == "poem":
            keywords = ["poem", "poetry", "verse", "sonnet", "haiku", "rhyme", "stanza", "lyric"]
        elif category == "professional_coding" and instruction_type == "web_development":
            keywords = ["web", "html", "css", "javascript", "frontend", "website", "react", "node", "responsive", "ui", "ux"]
        elif category == "tutor" and instruction_type == "socratic":
            keywords = ["socratic", "guide me", "ask me questions", "help me understand"]
        elif category == "conversational_ai" and instruction_type == "therapeutic":
            keywords = ["support", "emotional", "therapeutic", "comfort", "feel", "help me", "advice", "struggling", "difficult"]

        if any(word in prompt_lower for word in keywords):
            return instruction_type

    return "default"

def select_thinking_level(category: str, prompt: str) -> str:
    """
    Selects the thinking level (e.g., low, medium, high) based on prompt complexity.
    This thinking level is used to select temperature and thinking budget.
    """
    prompt_lower = prompt.lower()
    word_count = len(prompt_lower.split())

    # Keywords for higher creativity/complexity
    high_keywords = [
        "creative", "imagine", "novel idea", "out of the box", "brainstorm", "explore", 
        "different perspective", "complex", "multi-step", "advanced", "detailed", 
        "in-depth", "architecture", "system design", "recursive", "algorithm", 
        "proof", "theorem", "comprehensive"
    ]
    
    # Keywords for lower, more factual/simple tasks
    low_keywords = [
        "fact", "strictly", "exact", "precise", "code", "step-by-step", "calculate",
        "summarize", "list", "what is"
    ]

    if any(keyword in prompt_lower for keyword in high_keywords) or word_count > 150:
        return "high"
    
    if any(keyword in prompt_lower for keyword in low_keywords) or word_count < 20:
        return "low"

    # Default to medium
    return "medium"

def choose_model(prompt: str) -> dict:
    """
    First categorizes the user request, then selects the instruction, temperature,
    and thinking budget to build the final model configuration.
    """
    print("🧠 Router: Analyzing prompt...")

    # 1. Categorize the task
    categorization = categorize_task(prompt)
    category = categorization["category"]
    category_config = PROMPT_CONFIG[category]
    
    # 2. Get the model configuration (it's now a single object)
    model_info = category_config["model"].copy() # Use .copy() to avoid modifying the original config

    # 3. Select the instruction
    instruction_type = select_instruction_for_category(category, prompt)
    selected_instruction = category_config["instructions"][instruction_type]
    
    # 4. Select the operational thinking level
    thinking_level = select_thinking_level(category, prompt)
    
    # 5. Select the temperature based on the thinking level, if available
    if "temperatures" in category_config:
        selected_temperature = category_config["temperatures"][thinking_level]
        model_info["temperature"] = selected_temperature
    
    # 6. Select and apply thinking budget if available
    if "thinking_levels" in category_config:
        selected_level_value = category_config["thinking_levels"][thinking_level]
        if selected_level_value is not None:
            model_info["thinking_budget"] = selected_level_value
            
    # --- Print summary ---
    print(f"✅ Router: Selected category: '{category}'")
    print(f"📝 Router: Selected instruction: '{instruction_type}'")

    # Only show thinking level if it's for a "thinking" category
    if "thinking_levels" in category_config:
        print(f"⚙️ Router: Selected thinking level: '{thinking_level}'")

    if "temperature" in model_info:
        print(f"🌡️ Router: Applying temperature: {model_info['temperature']}")
    
    if "thinking_budget" in model_info and model_info["thinking_budget"] is not None:
        print(f"💡 Router: Applying thinking budget: {model_info['thinking_budget']}")

    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "instruction_subtype": instruction_type
    }