"""
Comprehensive instructions for the ML Agent - Intermediate Workshop Template
"""

# TODO: Create ML_AGENT_PROMPT variable containing the complete agent instructions
# The prompt should be a multi-line string with the following structure:

ML_AGENT_PROMPT = """
# TODO: Section 1 - Role Definition (8-10 lines)
# Define the agent as an expert ML Classification Assistant powered by Google ADK
# List the 5 core capabilities: train models, make predictions, model management, dataset discovery, smart target detection, general ML advice
# Specify focus on classification only (Random Forest, SVM, Logistic Regression)

# TODO: Section 2 - Decision Making Process (15-20 lines)
# Create intent analysis system with keywords for:
# - Training requests: "train", "build", "create model", "learn from data", "classify", "classifier"
# - Prediction requests: "predict", "classify", "forecast", "inference", "estimate"  
# - Model management: "list models", "available models", "show trained models"
# - Dataset discovery: "what data", "available datasets", "show datasets", "data files"
# - General questions: ML concepts, best practices, algorithm selection

# TODO: Section 3 - Tool Routing Logic (40-50 lines)
# For each intent type, specify:
# A. Training Requests:
#    - MUST call train_ml_model with task_type='classification' and appropriate model_type
#    - Handle dataset path detection (personality_dataset.csv for personality tasks)
#    - Smart target column detection rules
#    - Default to 'random_forest' if model type not specified
# 
# B. Prediction Requests:
#    - MUST call predict_with_model with task_type='classification'
#    - Handle various input formats (JSON, CSV paths, test data)
#    - Include label decoding instructions for classification results
#
# C. Model Management:
#    - MUST call list_available_models
#    - Present results in organized format
#
# D. Dataset Discovery:
#    - MUST call list_available_datasets
#    - Show available datasets with structure
#
# E. General Questions:
#    - DO NOT call tools, provide educational guidance

# TODO: Section 4 - Result Presentation (25-30 lines)
# Specify formatting rules for:
# - Training results: Show metrics, model name, performance explanation
# - Prediction results: Handle user-specific formatting requests (first N rows, etc.)
# - CRITICAL: Always decode classification predictions using label_mapping
# - Model listings: Show names, dates, metrics
# - Error handling: Clear explanations and suggestions

# TODO: Section 5 - System Constraints (10-15 lines)
# - Only classification supported (no regression)
# - Available models: random_forest, svm, logistic_regression  
# - Data formats: CSV, Excel, JSON
# - Always use task_type="classification"
# - Educational and encouraging approach

# TODO: Section 6 - Example Interactions (15-20 lines)
# Provide 8-10 example queries and their expected tool calls:
# - "Train a classification model" → detect_target_column_from_query + train_ml_model
# - "Classify this data: {...}" → predict_with_model
# - "What models do I have?" → list_available_models
# - "Make predictions using data/test.csv" → predict_with_model
# - General ML questions → No tools, educational response
# - Include critical reminders about required parameters

"""

# IMPLEMENTATION HINTS:
# - The prompt should be comprehensive (140+ lines) to match production system
# - Focus on classification-only functionality throughout
# - Include specific parameter requirements for each tool
# - Emphasize label decoding for classification predictions
# - Provide clear error handling guidance
# - Make it educational and user-friendly
# - Ensure ADK web compatibility 