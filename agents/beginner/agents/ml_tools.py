from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from google.adk.tools import ToolContext

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