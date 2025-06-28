"""
Root agent for ADK web development and viewing
"""

from google.adk.agents import Agent
from config import Config
from agents.ml_tools import train_ml_model, predict_with_model, list_available_models, list_available_datasets, detect_target_column_from_query
from agents.prompt import ML_AGENT_PROMPT

# Root agent variable for ADK web
root_agent = Agent(
    name="ml_agentic_system",
    model=Config.GEMINI_MODEL,
    description=(
        "An intelligent ML agent that can train classification models, make predictions, and provide comprehensive ML assistance. "
        "Specializes in classification tasks with multiple algorithms including Random Forest, SVM, and Logistic Regression."
    ),
    instruction=ML_AGENT_PROMPT,
    tools=[
        train_ml_model,
        predict_with_model,
        list_available_models,
        list_available_datasets,
        detect_target_column_from_query
    ]
) 
