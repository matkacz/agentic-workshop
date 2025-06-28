"""
Comprehensive instructions for the ML Agent
"""

# TODO: Create ML_AGENT_PROMPT variable with the complete prompt
# HINT: This should be a multi-line string that starts with:
# **Your Role:** You are an expert ML Classification Assistant...

# INSTRUCTOR GUIDANCE: Build this prompt section by section during the workshop
# Each section teaches different concepts about prompt engineering and ML

ML_AGENT_PROMPT = """
# TODO SECTION 1: Define the agent's role and core capabilities
# HINT: Start with "**Your Role:** You are an expert ML Classification Assistant powered by Google ADK..."
# HINT: List the 5 core capabilities: Train Models, Make Predictions, Model Management, Dataset Discovery, General ML Advice



# TODO SECTION 2: Decision Making Process
# HINT: Explain how the agent analyzes user intent
# HINT: Cover Training Requests, Prediction Requests, Model Management, Dataset Discovery, General Questions



# TODO SECTION 3: Action Routing (Most Important Section!)
# HINT: For Training Requests - MUST call train_ml_model with required parameters
# HINT: For Prediction Requests - MUST call predict_with_model
# HINT: For Model Management - MUST call list_available_models
# HINT: For Dataset Discovery - MUST call list_available_datasets
# HINT: For General Questions - DO NOT call any tools



# TODO SECTION 4: Result Presentation
# HINT: How to format training results with metrics
# HINT: How to format prediction results with label decoding
# HINT: How to present model listings



# TODO SECTION 5: Error Handling and Educational Approach
# HINT: How to handle errors gracefully
# HINT: Always explain what you're doing and why



# TODO SECTION 6: Available Models and Examples
# HINT: List available models: random_forest, svm, logistic_regression
# HINT: Provide example interactions



"""

# WORKSHOP INSTRUCTOR NOTES:
# =========================
# 1. Build this prompt section by section (20-25 minutes total)
# 2. Explain prompt engineering concepts as you go
# 3. Show how each section affects agent behavior
# 4. Test the agent after each major section is added
# 5. Emphasize the importance of clear instructions for AI agents
# 
# SECTION BREAKDOWN:
# - Section 1 (5 min): Role definition and capabilities
# - Section 2 (5 min): Intent analysis and routing logic  
# - Section 3 (10 min): Detailed tool usage instructions (MOST CRITICAL)
# - Section 4 (3 min): Result formatting
# - Section 5 (2 min): Error handling and educational approach
# - Section 6 (3 min): Examples and available models
#
# TEACHING POINTS:
# - Why specific instructions matter for AI agents
# - How tool calling works in ADK
# - The importance of parameter validation
# - Classification-only focus and why
# - Real-world prompt engineering best practices 