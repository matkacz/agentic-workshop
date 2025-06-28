# 🎓 Instructor Solutions - Intermediate Workshop

This document contains the complete solutions for the intermediate ML Classification Agent workshop.

## 📁 File: `agents/agent.py`

```python
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
```

## 📁 File: `agents/ml_tools.py`

**Note:** The `predict_with_model` function is already provided in the template and should NOT be modified by attendees.

**Functions to implement:**
- `train_ml_model` - Core training functionality
- `list_available_models` - Model discovery
- `list_available_datasets` - Dataset discovery  
- `detect_target_column_from_query` - Target column detection

```python
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
from google.adk.tools import ToolContext
import json
import traceback
import os

from config import Config
from ml_models import ModelManager

# Initialize global components
model_manager = ModelManager(Config.MODELS_DIR)

def train_ml_model(task_type: str, model_type: str, dataset_path: str, target_column: str) -> Dict[str, Any]:
    """Train classification model with specified algorithm and return metrics"""
    try:
        print(f"--- ML Tool: train_ml_model called ---")
        print(f"--- Parameters: task_type={task_type}, model_type={model_type}, dataset_path={dataset_path}, target_column={target_column} ---")
        
        # Validate task type
        if task_type != 'classification':
            return {
                'status': 'error',
                'message': f'Only classification is supported. Got: {task_type}'
            }
        
        # Load data
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"--- Loaded data: {X.shape} features, {len(y)} samples ---")
        
        # Handle categorical features (simple encoding)
        from sklearn.preprocessing import LabelEncoder
        categorical_encoders = {}
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                print(f"--- Encoding categorical column: {col} ---")
                
                # Handle missing values consistently with prediction function
                X[col] = X[col].fillna('nan')
                
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                categorical_encoders[col] = le
        
        # Handle categorical target for classification
        target_encoder = None
        if y.dtype == 'object':
            print(f"--- Encoding categorical target: {target_column} ---")
            
            # Handle missing values in target consistently
            y = y.fillna('nan')
            
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        # Handle missing values in numeric columns
        X = X.fillna(0)
        print(f"--- Filled missing values in numeric columns with 0 ---")
        
        # Simple preprocessing (scaling)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"--- Applied feature scaling ---")
        
        # Train classification model based on model_type
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42)  # Enable probability for predict_proba
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            return {
                'status': 'error',
                'message': f'Unsupported classification model type: {model_type}. Available: random_forest, svm, logistic_regression'
            }
        
        model.fit(X_scaled, y)
        print(f"--- Model training completed: {model.__class__.__name__} ---")
        
        # Basic evaluation
        y_pred = model.predict(X_scaled)
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y, y_pred)
        metric_name = 'accuracy'
        
        print(f"--- Model performance: {metric_name} = {score:.3f} ---")
        
        # Create appropriate model wrapper
        from ml_models.classification_models import RandomForestClassificationModel, SVMClassificationModel, LogisticRegressionModel
        
        if model_type == 'random_forest':
            model_wrapper = RandomForestClassificationModel()
        elif model_type == 'svm':
            model_wrapper = SVMClassificationModel()
        elif model_type == 'logistic_regression':
            model_wrapper = LogisticRegressionModel()
        
        # Set the trained model and metadata
        model_wrapper.model = model
        model_wrapper.is_trained = True
        model_wrapper.preprocessing_info = {
            'feature_scaler': scaler,
            'feature_names': X.columns.tolist(),
            'categorical_encoders': categorical_encoders
        }
        
        if target_encoder:
            model_wrapper.preprocessing_info['target_encoder'] = target_encoder
        
        model_wrapper.model_metadata.update({
            'training_samples': len(y),
            'feature_names': X.columns.tolist(),
            'sklearn_version': '1.3.0',  # Approximate version
            'model_algorithm': model.__class__.__name__
        })
        
        # Save the model
        save_result = model_manager.save_model(model_wrapper)
        model_name = model_wrapper.model_name
        print(f"--- Model saved: {model_name} ---")
        
        return {
            'status': 'success',
            'model_name': model_name,
            'model_type': model_type,
            'task_type': task_type,
            'score': score,
            'metric_name': metric_name,
            'training_samples': len(y),
            'feature_count': X.shape[1],
            'algorithm': model.__class__.__name__,
            'message': f'Successfully trained {model_type} classification model ({model.__class__.__name__}) with {metric_name}: {score:.3f}'
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"--- ERROR: {error_msg} ---")
        return {
            'status': 'error',
            'message': error_msg
        }

def predict_with_model(task_type: str, input_data: str) -> Dict[str, Any]:
    """Make classification predictions and return results"""
    try:
        print(f"--- ML Tool: predict_with_model called ---")
        print(f"--- Parameters: task_type={task_type}, input_data={input_data} ---")
        
        # Validate task type
        if task_type != 'classification':
            return {
                'status': 'error',
                'message': f'Only classification is supported. Got: {task_type}'
            }
        
        # Parse input (JSON or CSV path)
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
            # Remove target column if it exists
            if 'target' in df.columns:
                df = df.drop(columns=['target'])
            X_input_df = df  # Preserve DataFrame for categorical encoding
            X = df.values
            print(f"--- Loaded CSV data: {X.shape} ---")
        else:
            import json
            data = json.loads(input_data)
            X = np.array([list(data.values())])
            X_input_df = None
            print(f"--- Loaded JSON data: {X.shape} ---")
        
        # Find and load a trained classification model
        trained_models = model_manager.list_trained_models()
        if not trained_models:
            return {
                'status': 'error',
                'message': 'No trained models found. Please train a classification model first.'
            }
        
        # Find a classification model
        matching_models = [m for m in trained_models if m['task_type'] == 'classification']
        if not matching_models:
            return {
                'status': 'error',
                'message': f'No classification models found. Available models: {[m["task_type"] for m in trained_models]}'
            }
        
        # Use the most recent model
        model_info = matching_models[0]
        model_name = model_info['model_name']
        print(f"--- Using model: {model_name} ---")
        
        # Load the model
        model = model_manager.load_model(model_name, 'classification')
        if not model:
            return {
                'status': 'error',
                'message': f'Failed to load model: {model_name}'
            }
        
        # Apply preprocessing if available
        if hasattr(model, 'preprocessing_info') and model.preprocessing_info:
            preprocessing_info = model.preprocessing_info
            
            # Handle categorical encoding for CSV input data
            if 'categorical_encoders' in preprocessing_info and input_data.endswith('.csv') and X_input_df is not None:
                print(f"--- Applying categorical encoding to CSV data ---")
                df_input = X_input_df.copy()
                
                # Apply categorical encoders
                categorical_encoders = preprocessing_info['categorical_encoders']
                for col, encoder in categorical_encoders.items():
                    if col in df_input.columns:
                        print(f"--- Encoding column: {col} ---")
                        try:
                            # Handle missing values by filling with a default value first
                            df_input[col] = df_input[col].fillna('nan')
                            
                            # Convert to string to ensure consistent data type
                            col_values = df_input[col].astype(str)
                            
                            # Apply encoder with fallback for unseen categories
                            encoded_values = []
                            for value in col_values:
                                if value in encoder.classes_:
                                    encoded_values.append(encoder.transform([value])[0])
                                else:
                                    # Use 'nan' as fallback for unseen categories (it's in encoder.classes_)
                                    print(f"--- Warning: Unseen category '{value}' in column {col}, using 'nan' fallback ---")
                                    encoded_values.append(encoder.transform(['nan'])[0])
                            
                            df_input[col] = encoded_values
                            
                        except Exception as e:
                            print(f"--- Error encoding column {col}: {e} ---")
                            # If encoding fails completely, fill with zeros
                            df_input[col] = 0
                
                # Handle any remaining missing values in numeric columns
                df_input = df_input.fillna(0)
                
                # Convert back to numpy array
                X = df_input.values
                print(f"--- Applied categorical encoding ---")
            
            # Scale features
            if 'feature_scaler' in preprocessing_info:
                X = preprocessing_info['feature_scaler'].transform(X)
                print(f"--- Applied feature scaling ---")
        
        # Make predictions
        predictions = model.predict(X)
        print(f"--- Made {len(predictions)} predictions ---")
        
        # Get probabilities for classification
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            print(f"--- Got prediction probabilities ---")
        
        # Return results
        results = {
            'status': 'success',
            'predictions': predictions.tolist(),
            'model_name': model_name,
            'task_type': 'classification',
            'total_predictions': len(predictions),
            'message': f'Made {len(predictions)} predictions using {model_name}'
        }
        
        if probabilities is not None:
            results['probabilities'] = probabilities.tolist()
        
        # Add label mapping for classification if available
        if hasattr(model, 'preprocessing_info') and model.preprocessing_info:
            target_encoder = model.preprocessing_info.get('target_encoder')
            if target_encoder:
                # Provide label mapping so LLM can decode predictions
                label_mapping = {i: label for i, label in enumerate(target_encoder.classes_)}
                results['label_mapping'] = label_mapping
                results['encoded_predictions'] = predictions.tolist()  # Keep original for reference
                results['message'] += f'. Use label_mapping to decode predictions: {label_mapping}'
                print(f"--- Added label mapping: {label_mapping} ---")
        
        return results
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"--- ERROR: {error_msg} ---")
        return {
            'status': 'error',
            'message': error_msg
        }

def list_available_models(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    List all available and trained classification models in the system.
        
    Returns:
        Dictionary containing available model types and trained models
    """
    try:
        print(f"--- ML Tool: list_available_models called ---")
        
        # Get available model types (classification only)
        available_types = model_manager.get_available_models()
        
        # Get trained models
        trained_models = model_manager.list_trained_models()
        
        # Filter for classification models only
        classification_models = [model for model in trained_models if model['task_type'] == 'classification']
        
        print(f"--- Found {len(classification_models)} trained classification models ---")
        
        return {
            'status': 'success',
            'available_model_types': available_types,
            'trained_models': classification_models,
            'total_trained_models': len(classification_models)
        }
        
    except Exception as e:
        error_msg = f"Error listing models: {str(e)}"
        print(f"--- ERROR: {error_msg} ---")
        return {
            'status': 'error',
            'message': error_msg
        }

def list_available_datasets() -> Dict[str, Any]:
    """List available CSV datasets for classification"""
    data_dir = Path("data")
    if not data_dir.exists():
        return {'status': 'success', 'datasets': [], 'message': 'No data folder found'}
    
    csv_files = list(data_dir.glob("*.csv"))
    datasets = [{'name': f.name, 'path': str(f)} for f in csv_files]
    
    return {
        'status': 'success',
        'datasets': datasets,
        'message': f'Found {len(datasets)} CSV files for classification'
    }

def detect_target_column_from_query(user_query: str, dataset_path: str) -> Dict[str, Any]:
    """Detect target column for classification from user query and dataset"""
    try:
        df = pd.read_csv(dataset_path, nrows=5)
        return {
            'status': 'success',
            'columns': df.columns.tolist(),
            'message': f'Available columns for classification: {df.columns.tolist()}'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
```

## 📁 File: `agents/prompt.py`

```python
"""
Comprehensive instructions for the ML Agent
"""

ML_AGENT_PROMPT = """
**Your Role:** You are an expert ML Classification Assistant powered by Google ADK that helps users with classification machine learning tasks.

**Core Capabilities:**
1. **Train Classification Models:** Can train classification models using various algorithms (Random Forest, SVM, Logistic Regression)
2. **Make Predictions:** Can load existing models or train new ones for classification inference
3. **Model Management:** Can list, compare, and manage trained classification models
4. **Dataset Discovery:** Can find and analyze available datasets in the data folder
5. **Smart Target Detection:** Can automatically detect target columns from user queries
6. **General ML Advice:** Can answer questions about classification and machine learning concepts

**Decision Making Process:**

**1. Analyze User Intent:**
- **Training Request:** Keywords like "train", "build", "create model", "learn from data", "classify", "classifier"
- **Prediction Request:** Keywords like "predict", "classify", "forecast", "inference", "estimate"
- **Model Management:** Keywords like "list models", "available models", "show trained models"
- **Dataset Discovery:** Keywords like "what data", "available datasets", "show datasets", "data files"
- **General Question:** ML concepts, best practices, algorithm selection, etc.

**2. Route Actions Based on Intent:**

**A. For Training Requests:**
- **MUST call `train_ml_model` tool with ALL required parameters**
- **REQUIRED PARAMETERS (must always provide):**
  - `task_type`: ALWAYS 'classification' (only classification is supported)
  - `model_type`: MUST choose from available options below
- **MUST select appropriate model type (REQUIRED):**
  - **Classification**: 'random_forest', 'svm', 'logistic_regression'
  - **Default choice**: Use 'random_forest' if user doesn't specify
- **Dataset Handling Priority:**
  1. If user mentions specific dataset path, use it exactly
  2. For personality/extrovert-introvert tasks, use "data/personality_dataset.csv"
  3. For other tasks, suggest looking in ./data/ folder for relevant files
  4. Only use synthetic data if no real dataset is available or requested
- **Smart Target Column Detection:**
  1. **DO NOT** call `detect_target_column_from_query` for synthetic data (dataset_path=None)
  2. **ONLY** call `detect_target_column_from_query` when dataset_path is provided (real datasets)
  3. For synthetic data: Always use target_column="target" (synthetic datasets always have "target" column)
  4. For real datasets: Use detected target column with high confidence
  5. For medium/low confidence, mention the detection and ask for confirmation
- Extract any hyperparameters mentioned by user
- **CRITICAL:** Never call train_ml_model without both task_type='classification' AND model_type parameters

**B. For Prediction Requests:**
- **MUST call `predict_with_model` tool**
- **ALWAYS use task_type='classification'** (only classification is supported)
- Extract input data (can be JSON, CSV file path, test data, or description)
- **Input data formats supported:**
  - JSON string: `{"feature1": value1, "feature2": value2}` or array of objects
  - CSV file path: `"data/inference_data.csv"` (will load and process the file)
  - "test" keyword: generates synthetic test data
- If specific model mentioned, use model_name parameter

**C. For Model Management:**
- **MUST call `list_available_models` tool**
- Present results in organized, readable format
- Highlight model performance metrics when available

**D. For Dataset Discovery:**
- **MUST call `list_available_datasets` tool**
- Show available datasets with their structure and size
- Suggest appropriate datasets for the user's classification task
- Explain how to use the dataset_path parameter

**E. For General Questions:**
- **DO NOT call any tools**
- Provide educational, accurate ML guidance focused on classification
- Explain concepts clearly with examples
- Suggest practical approaches

**3. Presenting Results:**

**For Training Results:**
- Clearly state the model type and that it's a classification task
- Present key metrics prominently:
  - **Classification**: Accuracy, Precision, Recall, F1-score
- Explain what the metrics mean in practical terms
- If visualizations are available, mention them and their insights
- Provide the model name for future reference

**For Prediction Results:**
- **IMPORTANT**: Format results based on user's specific request
- If user asks for "first 10 rows", "top 5 predictions", "show results for first 20", etc., extract and display exactly what they requested
- **For classification predictions with label_mapping**: ALWAYS decode the predictions using the provided mapping
  - Raw predictions are encoded values (0, 1, 2, etc.)
  - Use label_mapping to convert to meaningful labels (e.g., {0: 'Introvert', 1: 'Extrovert'})
  - Display both the decoded labels AND the confidence scores if available
  - Example: "Prediction: Extrovert (confidence: 0.85)" instead of just "1"
- For classification, include confidence scores if available and requested
- Present predictions in a clear, readable format (table, list, etc.)
- Explain what the predictions mean in context
- If a new model was trained, briefly mention its performance
- **Examples of user-specific formatting:**
  - "Show first 10 predictions" → Display predictions[0:10] with decoded labels
  - "Display results for first 5 rows" → Show first 5 predictions with row numbers and decoded labels
  - "What are the top predictions?" → Show all predictions or ask for specific count, with labels
  - "Show me some sample results" → Display 5-10 representative predictions with meaningful labels

**For Model Listings:**
- Show classification model names, creation dates, and key metrics
- Help user choose appropriate model for their classification needs

**4. Error Handling:**
- If a tool returns an error, explain the issue clearly
- Suggest specific fixes (e.g., correct data format, valid parameters)
- Offer alternatives when possible
- If user asks for regression, explain that only classification is supported

**5. Educational Approach:**
- Always explain what you're doing and why
- Provide context about algorithm choices for classification
- Suggest improvements or next steps
- Be encouraging and supportive for beginners

**6. Visualization and Results:**
- When training results include visualizations, describe what they show
- Explain how to interpret plots like confusion matrices, feature importance, etc.
- Guide users on how to use the insights for model improvement

**Available Models:**
- **Classification Only:** random_forest, svm, logistic_regression

**Data Formats Supported:**
- CSV, Excel, JSON files for training
- JSON objects/arrays for prediction input
- CSV file paths for prediction input (e.g., "data/inference_data.csv")
- Automatic synthetic data generation for demos

**Example Interactions:**
- "Train a classification model to predict customer churn" → detect_target_column_from_query, then train_ml_model(task_type="classification", model_type="random_forest", ...)
- "Classify this data: {...}" → predict_with_model(task_type="classification", input_data=...)
- "Make predictions using data/inference_personality.csv" → predict_with_model(task_type="classification", input_data="data/inference_personality.csv")
- "Classify the data in data/test_data.csv" → predict_with_model(task_type="classification", input_data="data/test_data.csv")
- "What models do I have available?" → list_available_models
- "What datasets are available?" → list_available_datasets
- "Train a personality classifier" → detect_target_column_from_query("personality classifier", "data/personality_dataset.csv"), then train_ml_model(task_type="classification", model_type="random_forest", dataset_path="data/personality_dataset.csv", target_column=detected_column)
- "Build a spam classifier" → detect_target_column_from_query("spam classifier", dataset_path), then train_ml_model(task_type="classification", model_type="random_forest", ...)
- "Train a classification model on synthetic data" → train_ml_model(task_type="classification", model_type="random_forest", dataset_path=None, target_column="target")
- "Explain the difference between precision and recall" → General explanation (no tools)

**CRITICAL REMINDERS:** 
- Always use task_type="classification" when calling train_ml_model and predict_with_model
- Always provide both task_type and model_type when calling train_ml_model
- If user asks for regression, politely explain that only classification is supported
- Never call tools with missing required parameters

Remember: Always be helpful, educational, and encouraging. Make classification ML accessible while being technically accurate.
"""
```

## 🎯 Key Implementation Notes

### Agent Architecture
- **Variable Name**: Must use `root_agent` for ADK web compatibility
- **Model**: Uses `Config.GEMINI_MODEL` (Gemini 2.0 Flash)
- **Tools**: All 5 tools must be imported and included in the tools list
- **Description**: Focus on classification capabilities

### Prompt Engineering
- **Length**: Comprehensive prompt (~4,000+ characters)
- **Structure**: 6 main sections covering role, decision making, tool routing, result presentation, constraints, and examples
- **Classification Focus**: Consistently emphasizes classification-only functionality
- **Tool Parameters**: Explicit parameter requirements for each tool call
- **Label Decoding**: Critical instructions for classification prediction formatting

### Common Student Mistakes to Watch For
1. **Missing imports**: Forgetting to import all required modules
2. **Wrong variable name**: Using something other than `root_agent`
3. **Incomplete tools list**: Missing one or more of the 5 required tools
4. **Short prompt**: Not including comprehensive instructions
5. **Missing task_type**: Forgetting to specify task_type='classification'
6. **Tool parameter errors**: Not including required parameters in prompt instructions

### Workshop Timing Guidelines
- **Phase 1 (25 min)**: Agent setup should be straightforward for intermediate level
- **Phase 2 (35 min)**: Prompt engineering is the main challenge - provide guidance on structure
- **Phase 3 (15 min)**: Testing should validate complete functionality

### Assessment Criteria
- ✅ Agent loads and initializes correctly
- ✅ All 5 tools are properly connected
- ✅ Prompt is comprehensive and covers all required sections
- ✅ Classification-only focus is maintained
- ✅ Tool parameter specifications are complete
- ✅ System integrates with CLI and web interfaces
- ✅ Matches production system functionality exactly

---

**Instructor Notes:**
- Intermediate students should need minimal guidance on basic Python/imports
- Focus assistance on prompt engineering structure and completeness
- Emphasize the importance of matching the production system exactly
- Use testing commands to validate implementations progressively
- Be available for questions on ADK-specific concepts and tool integration 