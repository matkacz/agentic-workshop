from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import joblib
from pathlib import Path
import json
from datetime import datetime

class BaseMLModel(ABC):
    """Abstract base class for all ML models in the system - primarily for metadata storage and persistence"""
    
    def __init__(self, model_name: str, task_type: str):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.model_name = model_name
        self.task_type = task_type
        self.model = None
        self.is_trained = False
        self.preprocessing_info = {}
        self.training_history = {}
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'task_type': task_type,
            'sklearn_version': None,  # Will be set during training
            'training_samples': None,
            'feature_names': None,
            'target_classes': None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.validate_input(X):
            raise ValueError("Invalid input data format")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available
        
        Returns:
            Feature importance array or None if not available
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Default implementation for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        # For linear models, use coefficient magnitudes
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            if coefs.ndim > 1:
                return np.mean(np.abs(coefs), axis=0)
            else:
                return np.abs(coefs)
        
        return None
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for classification models
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions or None if not applicable
        """
        if hasattr(self.model, 'predict_proba') and self.task_type == 'classification':
            return self.model.predict_proba(X)
        return None
    
    def save_model(self, save_path: Path) -> Dict[str, str]:
        """
        Save the trained model to disk
        
        Args:
            save_path: Directory path to save the model
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific directory
        model_dir = save_path / f"{self.model_name}_{self.task_type}"
        model_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save the actual model
        model_path = model_dir / "model.joblib"
        joblib.dump(self.model, model_path)
        saved_files['model'] = str(model_path)
        
        # Save preprocessing info
        if self.preprocessing_info:
            preprocessing_path = model_dir / "preprocessing.joblib"
            joblib.dump(self.preprocessing_info, preprocessing_path)
            saved_files['preprocessing'] = str(preprocessing_path)
        
        # Save training history
        if self.training_history:
            history_path = model_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            saved_files['history'] = str(history_path)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        return saved_files
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            # Load the actual model
            model_file = model_path / "model.joblib"
            if model_file.exists():
                self.model = joblib.load(model_file)
                self.is_trained = True
            else:
                return False
            
            # Load preprocessing info
            preprocessing_file = model_path / "preprocessing.joblib"
            if preprocessing_file.exists():
                self.preprocessing_info = joblib.load(preprocessing_file)
            
            # Load training history
            history_file = model_path / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.training_history = json.load(f)
            
            # Load metadata
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                    # Update current model info
                    self.model_name = self.model_metadata.get('model_name', self.model_name)
                    self.task_type = self.model_metadata.get('task_type', self.task_type)
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model metadata and status
        """
        info = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'is_trained': self.is_trained,
            'has_preprocessing': bool(self.preprocessing_info),
            'has_history': bool(self.training_history),
            'metadata': self.model_metadata
        }
        
        if self.is_trained and hasattr(self.model, '__class__'):
            info['model_type'] = self.model.__class__.__name__
        
        return info
    
    def validate_input(self, X: np.ndarray) -> bool:
        """
        Validate input data format
        
        Args:
            X: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(X, np.ndarray):
            return False
        
        if len(X.shape) != 2:
            return False
        
        if self.is_trained and 'feature_names' in self.model_metadata:
            expected_features = len(self.model_metadata['feature_names'])
            if X.shape[1] != expected_features:
                return False
        
        return True 