"""
ML Tools Implementation - Intermediate Workshop Template
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from google.adk.tools import ToolContext

from config import Config
from ml_models import ModelManager

# TODO: Import necessary modules
# HINT: You'll need Config from config
# HINT: You'll need ModelManager from ml_models

# TODO: Initialize global components
# HINT: Create model_manager instance using Config.MODELS_DIR
# HINT: model_manager = ModelManager(Config.MODELS_DIR)

def train_ml_model(task_type: str, model_type: str, dataset_path: str, target_column: str) -> Dict[str, Any]:
    """
    Train classification model with specified algorithm and return metrics
    
    Args:
        task_type: Type of ML task (must be 'classification')
        model_type: Algorithm type ('random_forest', 'svm', 'logistic_regression')
        dataset_path: Path to training dataset
        target_column: Name of target column
    
    Returns:
        Dictionary with training results and metrics
    """
    try:
        print(f"--- ML Tool: train_ml_model called ---")
        print(f"--- Parameters: task_type={task_type}, model_type={model_type}, dataset_path={dataset_path}, target_column={target_column} ---")
        
        # TODO: Validate task_type (only 'classification' allowed)
        # HINT: Check if task_type != 'classification' and return error if not
        
        # TODO: Load and prepare data
        # HINT: Use pd.read_csv() to load dataset_path
        # HINT: Separate features (X) and target (y) using target_column
        # HINT: Print data shape for debugging
        
        # TODO: Handle categorical features
        # HINT: Use LabelEncoder for object/string columns
        # HINT: Fill missing values with 'nan' before encoding
        # HINT: Store encoders in categorical_encoders dict for later use
        
        # TODO: Handle categorical target for classification
        # HINT: If target is object type, use LabelEncoder
        # HINT: Store target_encoder for prediction decoding
        
        # TODO: Handle missing values in numeric columns
        # HINT: Use fillna(0) for numeric columns
        
        # TODO: Apply feature scaling
        # HINT: Use StandardScaler to scale features
        # HINT: Store scaler for prediction preprocessing
        
        # TODO: Create and train model based on model_type
        # HINT: Use RandomForestClassifier for 'random_forest'
        # HINT: Use SVC(probability=True) for 'svm'
        # HINT: Use LogisticRegression for 'logistic_regression'
        # HINT: All should have random_state=42
        
        # TODO: Train the model
        # HINT: Use model.fit(X_scaled, y)
        
        # TODO: Evaluate model performance
        # HINT: Use model.predict() for predictions
        # HINT: Use accuracy_score from sklearn.metrics
        # HINT: Calculate accuracy on training data
        
        # TODO: Create appropriate model wrapper
        # HINT: Import classification models from ml_models.classification_models
        # HINT: Create instance based on model_type (RandomForestClassificationModel, etc.)
        
        # TODO: Set model wrapper properties
        # HINT: Set model_wrapper.model = trained_model
        # HINT: Set model_wrapper.is_trained = True
        # HINT: Set preprocessing_info with scaler, encoders, feature_names
        # HINT: Update model_metadata with training info
        
        # TODO: Save the model using model_manager
        # HINT: Call model_manager.save_model(model_wrapper)
        
        # TODO: Return success response
        # HINT: Include status, model_name, metrics, training info
        # HINT: Format message with algorithm name and performance
        
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
    List all available and trained classification models in the system
    
    Returns:
        Dictionary containing available model types and trained models
    """
    try:
        print(f"--- ML Tool: list_available_models called ---")
        
        # TODO: Get available model types
        # HINT: Use model_manager.get_available_models()
        
        # TODO: Get trained models
        # HINT: Use model_manager.list_trained_models()
        
        # TODO: Filter for classification models only
        # HINT: Filter trained models where model['task_type'] == 'classification'
        
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
    """
    List available CSV datasets for classification
    
    Returns:
        Dictionary with available datasets
    """
    # TODO: Implement dataset discovery
    # HINT: Check Path("data") directory
    # HINT: Use glob("*.csv") to find CSV files
    # HINT: Return list with name and path for each dataset
    # HINT: Handle case where data directory doesn't exist
    
    return {
        'status': 'success',
        'datasets': datasets,
        'message': f'Found {len(datasets)} CSV files for classification'
    }

def detect_target_column_from_query(user_query: str, dataset_path: str) -> Dict[str, Any]:
    """
    Detect target column for classification from user query and dataset
    
    Args:
        user_query: User's query text
        dataset_path: Path to dataset file
    
    Returns:
        Dictionary with available columns
    """
    try:
        # TODO: Load dataset sample and return available columns
        # HINT: Use pd.read_csv(dataset_path, nrows=5) to load just a few rows
        # HINT: Return df.columns.tolist() to show available columns
        
        return {
            'status': 'success',
            'columns': df.columns.tolist(),
            'message': f'Available columns for classification: {df.columns.tolist()}'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# IMPLEMENTATION NOTES:
# =====================
# 1. This template provides function signatures and detailed TODOs
# 2. The predict_with_model function is already implemented - DO NOT MODIFY
# 3. Attendees need to implement: train_ml_model, list_available_models, list_available_datasets, detect_target_column_from_query
# 4. Focus on classification-only functionality
# 5. Use consistent error handling and logging
# 6. Expected implementation time: 30 minutes
# 7. Test each function as you implement it
# 8. Refer to the production ml_tools.py for reference if needed 