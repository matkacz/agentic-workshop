import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union
import json
from datetime import datetime

from .base_model import BaseMLModel
from .classification_models import (
    RandomForestClassificationModel,
    SVMClassificationModel, 
    LogisticRegressionModel
)

class ModelManager:
    """Manages ML models - creation, training, loading, and inference"""
    
    def __init__(self, models_dir: Path):
        """
        Initialize ModelManager
        
        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of available model classes (classification only)
        self.model_registry = {
            'classification': {
                'random_forest': RandomForestClassificationModel,
                'svm': SVMClassificationModel,
                'logistic_regression': LogisticRegressionModel
            }
        }
        
        # Cache for loaded models
        self._loaded_models: Dict[str, BaseMLModel] = {}
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available model types by task"""
        return {
            task_type: list(models.keys()) 
            for task_type, models in self.model_registry.items()
        }
    
    def create_model(self, task_type: str, model_type: str, model_name: Optional[str] = None) -> BaseMLModel:
        """
        Create a new model instance
        
        Args:
            task_type: Type of ML task ('classification' only)
            model_type: Type of model algorithm
            model_name: Custom name for the model
            
        Returns:
            Initialized model instance
        """
        if task_type not in self.model_registry:
            raise ValueError(f"Unsupported task type: {task_type}. Only 'classification' is supported.")
        
        if model_type not in self.model_registry[task_type]:
            available = list(self.model_registry[task_type].keys())
            raise ValueError(f"Unsupported model type '{model_type}' for {task_type}. Available: {available}")
        
        model_class = self.model_registry[task_type][model_type]
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{task_type}_{timestamp}"
        
        return model_class(model_name)
    
    def save_model(self, model: BaseMLModel) -> Dict[str, str]:
        """
        Save a trained model to disk
        
        Args:
            model: Trained model to save
            
        Returns:
            Dictionary with paths to saved files
        """
        return model.save_model(self.models_dir)
    
    def load_model(self, model_name: str, task_type: str) -> Optional[BaseMLModel]:
        """
        Load a previously trained model
        
        Args:
            model_name: Name of the model to load
            task_type: Task type of the model ('classification' only)
            
        Returns:
            Loaded model or None if not found
        """
        cache_key = f"{model_name}_{task_type}"
        
        # Check if already loaded in cache
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        # Look for model directory
        model_dir = self.models_dir / f"{model_name}_{task_type}"
        
        if not model_dir.exists():
            return None
        
        # Load metadata to determine model type
        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Determine model type from metadata or filename
            model_type = self._infer_model_type(model_name, metadata)
            
            if not model_type:
                return None
            
            # Create model instance
            model = self.create_model(task_type, model_type, model_name)
            
            # Load the trained model
            if model.load_model(model_dir):
                self._loaded_models[cache_key] = model
                return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
        
        return None
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """
        List all trained models in the models directory
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        model_info = {
                            'model_name': metadata.get('model_name', model_dir.name),
                            'task_type': metadata.get('task_type', 'unknown'),
                            'created_at': metadata.get('created_at', 'unknown'),
                            'training_samples': metadata.get('training_samples', 0),
                            'model_path': str(model_dir)
                        }
                        models.append(model_info)
                        
                    except Exception as e:
                        print(f"Error reading metadata for {model_dir.name}: {e}")
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def find_matching_models(self, task_type: str, feature_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find models that match the given criteria
        
        Args:
            task_type: Task type to match ('classification' only)
            feature_count: Number of features to match (optional)
            
        Returns:
            List of matching model information
        """
        all_models = self.list_trained_models()
        matching_models = []
        
        for model_info in all_models:
            if model_info['task_type'] == task_type:
                # Load full metadata to check feature count
                try:
                    metadata_file = Path(model_info['model_path']) / "metadata.json"
                    with open(metadata_file, 'r') as f:
                        full_metadata = json.load(f)
                    
                    model_features = full_metadata.get('n_features', 0)
                    
                    if feature_count is None or model_features == feature_count:
                        model_info['n_features'] = model_features
                        matching_models.append(model_info)
                        
                except Exception as e:
                    print(f"Error checking features for {model_info['model_name']}: {e}")
        
        return matching_models
    
    def delete_model(self, model_name: str, task_type: str) -> bool:
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model to delete
            task_type: Task type of the model ('classification' only)
            
        Returns:
            True if successfully deleted, False otherwise
        """
        model_dir = self.models_dir / f"{model_name}_{task_type}"
        cache_key = f"{model_name}_{task_type}"
        
        try:
            # Remove from cache if loaded
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
            
            # Delete directory and all contents
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                return True
                
        except Exception as e:
            print(f"Error deleting model {model_name}: {e}")
        
        return False
    
    def _infer_model_type(self, model_name: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Infer model type from model name or metadata
        
        Args:
            model_name: Name of the model
            metadata: Model metadata
            
        Returns:
            Inferred model type or None
        """
        # Check if model type is stored in metadata
        if 'model_type' in metadata:
            return metadata['model_type'].lower()
        
        # Infer from model name
        model_name_lower = model_name.lower()
        
        if 'random_forest' in model_name_lower or 'rf' in model_name_lower:
            return 'random_forest'
        elif 'svm' in model_name_lower:
            return 'svm'
        elif 'logistic' in model_name_lower:
            return 'logistic_regression'
        
        return None
    
    def get_model_summary(self, model_name: str, task_type: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed summary of a specific model
        
        Args:
            model_name: Name of the model
            task_type: Task type of the model ('classification' only)
            
        Returns:
            Model summary dictionary or None if not found
        """
        model = self.load_model(model_name, task_type)
        if model:
            return model.get_model_info()
        return None 