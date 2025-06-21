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
    MODEL_PROMPTS = json.load(f)

TASK_CATEGORIES = list(MODEL_PROMPTS.keys())

def categorize_task_and_instruction(prompt: str) -> dict:
    """
    Categorizes the user request into a task category and selects the appropriate instruction.
    """
    system_prompt = f"""
    You are an expert task classifier. Your job is to analyze a user's prompt and
    classify it into one of the following task categories.
    Respond with ONLY the category name and nothing else.

    Available categories are: {', '.join(TASK_CATEGORIES)}

    Examples:
    - "Write a poem about love" → creative_writing
    - "Build a web application" → professional_coding
    - "Solve this math equation" → advanced_reasoning
    - "Quick calculation" → fast_reasoning
    - "Explain quantum physics" → general_qa
    - "Summarize this article" → fast_simple_task
    - "Let's have a casual chat" → conversational_ai
    """

    try:
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
        if category in MODEL_PROMPTS:
            return {"category": category, "valid": True}
        else:
            print(f"⚠️ Router: Could not classify category '{category}'. Defaulting to 'general_qa'.")
            return {"category": "general_qa", "valid": False}
            
    except Exception as e:
        print(f"🚨 Router Error: {e}. Defaulting to 'general_qa'.")
        return {"category": "general_qa", "valid": False}

def select_instruction_for_category(category: str, prompt: str) -> str:
    """
    Selects the appropriate instruction for the given category based on prompt content.
    Choose between 'default' and one specialized prompt per category.
    """
    category_info = MODEL_PROMPTS[category]
    
    prompt_lower = prompt.lower()
    
    # Select instruction based on category and prompt content
    if category == "creative_writing":
        # Check for storytelling-specific keywords
        if any(word in prompt_lower for word in ["story", "narrative", "tale", "fiction", "novel", "plot", "character", "world", "dialogue"]):
            return "storytelling"
        else:
            return "default"
    
    elif category == "professional_coding":
        # Check for web development-specific keywords
        if any(word in prompt_lower for word in ["web", "html", "css", "javascript", "frontend", "website", "react", "node", "responsive", "ui", "ux"]):
            return "web_development"
        else:
            return "default"
    
    elif category == "advanced_reasoning":
        # Check for mathematical reasoning with LaTeX formatting
        if any(word in prompt_lower for word in ["math", "mathematical", "equation", "calculation", "proof", "solve", "formula", "theorem", "algebra", "calculus", "geometry"]):
            return "mathematical"
        else:
            return "default"
    
    elif category == "fast_reasoning":
        # Check for decision analysis keywords
        if any(word in prompt_lower for word in ["decision", "choose", "option", "recommend", "analysis", "compare", "evaluate", "select", "best", "trade-off"]):
            return "decision_analysis"
        else:
            return "default"
    
    elif category == "general_qa":
        # Check for educational/tutorial keywords
        if any(word in prompt_lower for word in ["explain", "teach", "learn", "education", "tutorial", "how to", "guide", "understand", "concept"]):
            return "educational"
        else:
            return "default"
    
    elif category == "fast_simple_task":
        # Check for summarization keywords
        if any(word in prompt_lower for word in ["summarize", "summary", "condense", "brief", "overview", "key points", "main points"]):
            return "summarization"
        else:
            return "default"
    
    elif category == "conversational_ai":
        # Check for therapeutic/support keywords
        if any(word in prompt_lower for word in ["support", "emotional", "therapeutic", "comfort", "feel", "help me", "advice", "struggling", "difficult"]):
            return "therapeutic"
        else:
            return "default"
    
    else:
        return "default"

def choose_model(prompt: str) -> dict:
    """
    First categorizes the user request into a task category and selects an instruction,
    then selects the best model for that category.
    """
    print("🧠 Router: Analyzing prompt to select category and instruction...")

    # 1. Categorize the task
    categorization = categorize_task_and_instruction(prompt)
    category = categorization["category"]
    
    # 2. Select the appropriate instruction for this category
    instruction_type = select_instruction_for_category(category, prompt)
    
    # 3. Get the model info and instruction
    model_info = MODEL_PROMPTS[category]
    selected_instruction = model_info["instructions"][instruction_type]
    
    print(f"✅ Router: Selected category: '{category}'")
    print(f"📝 Router: Selected instruction: '{instruction_type}'")
    
    return {
        "model_info": model_info,
        "instruction": selected_instruction,
        "category": category,
        "instruction_subtype": instruction_type
    }