import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, List
import json

class DataUtils:
    """Utility class for data handling and preprocessing"""
    
    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """Load dataset from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, task_type: str, target_column: str) -> Dict[str, Any]:
        """Validate dataset for ML task"""
        validation_info = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'target_column': target_column
            }
        }
        
        # Check if target column exists
        if target_column not in df.columns:
            validation_info['is_valid'] = False
            validation_info['issues'].append(f"Target column '{target_column}' not found in dataset")
            return validation_info
        
        # Check for missing values
        if df.isnull().any().any():
            validation_info['issues'].append("Dataset contains missing values")
            validation_info['suggestions'].append("Consider handling missing values through imputation or removal")
        
        # Check minimum samples
        if len(df) < 10:
            validation_info['is_valid'] = False
            validation_info['issues'].append("Dataset too small (minimum 10 samples required)")
        
        # Task-specific validation
        if task_type == 'classification':
            unique_values = df[target_column].nunique()
            if unique_values < 2:
                validation_info['is_valid'] = False
                validation_info['issues'].append("Classification requires at least 2 unique target values")
            elif unique_values > 100:
                validation_info['issues'].append("High number of classes detected, consider if this is intended")
        else:
            validation_info['is_valid'] = False
            validation_info['issues'].append(f"Unsupported task type: {task_type}. Only 'classification' is supported.")
        
        return validation_info
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, target_column: str, task_type: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Preprocess data for ML training"""
        preprocessing_info = {}
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        encoders = {}
        
        if categorical_columns:
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
            preprocessing_info['categorical_encoders'] = encoders
        
        # Handle target encoding for classification
        target_encoder = None
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            preprocessing_info['target_encoder'] = target_encoder
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        preprocessing_info['feature_scaler'] = scaler
        preprocessing_info['feature_names'] = X.columns.tolist()
        
        # Convert y to numpy array if it's still a pandas Series
        if hasattr(y, 'values'):
            y = y.values
        
        return X_scaled, y, preprocessing_info
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42, task_type: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets"""
        # Only stratify for classification tasks with discrete classes
        stratify = None
        if task_type == 'classification':
            unique_classes = np.unique(y)
            # Only stratify if we have multiple classes and each class has at least 2 samples
            if len(unique_classes) > 1:
                class_counts = np.bincount(y.astype(int))
                if np.all(class_counts >= 2):
                    stratify = y
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    
    @staticmethod
    def create_synthetic_dataset(task_type: str, n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
        """Create synthetic dataset for testing"""
        if task_type != 'classification':
            raise ValueError(f"Unsupported task type: {task_type}. Only 'classification' is supported.")
        
        np.random.seed(42)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        # Create classification target
        coefficients = np.random.randn(n_features)
        linear_combination = X @ coefficients
        probabilities = 1 / (1 + np.exp(-linear_combination))
        y = (probabilities > 0.5).astype(int)
        
        # Add some categorical features
        df = pd.DataFrame(X, columns=feature_names)
        df['category_A'] = np.random.choice(['type1', 'type2', 'type3'], n_samples)
        df['category_B'] = np.random.choice(['red', 'blue', 'green'], n_samples)
        df['target'] = y
        
        return df 