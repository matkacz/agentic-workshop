"""
Root agent for ADK web development and viewing
"""

# TODO 1: Import the necessary modules
# HINT: You need Agent from google.adk.agents
# HINT: You need Config from config
# HINT: You need all 5 tools from agents.ml_tools
# HINT: You need ML_AGENT_PROMPT from agents.prompt

# TODO: Add your imports here
# from google.adk.agents import ???
# from config import ???
# from agents.ml_tools import ???, ???, ???, ???, ???
# from agents.prompt import ???

# TODO 2: Create the root_agent variable
# HINT: Use Agent() constructor with these exact parameters:
#   - name: "ml_agentic_system"
#   - model: Config.GEMINI_MODEL  
#   - description: (see hint below)
#   - instruction: ML_AGENT_PROMPT
#   - tools: [list all 5 imported tools]

# DESCRIPTION HINT: 
# "An intelligent ML agent that can train classification models, make predictions, and provide comprehensive ML assistance. "
# "Specializes in classification tasks with multiple algorithms including Random Forest, SVM, and Logistic Regression."

# TODO: Replace None with the Agent() constructor call
root_agent = None

# WORKSHOP INSTRUCTOR NOTES:
# =========================
# 1. Guide attendees through each import step by step
# 2. Explain what each tool does as they add it
# 3. Discuss the Agent parameters and their purpose
# 4. Test the agent after creation to ensure it works
# 5. Expected completion time: 15-20 minutes 