import numpy as np
from typing import Dict, Any, Optional
from .base_model import BaseMLModel

class RandomForestClassificationModel(BaseMLModel):
    """Random Forest Classification Model - metadata container"""
    
    def __init__(self, model_name: str = "random_forest_classifier"):
        super().__init__(model_name, "classification")

class SVMClassificationModel(BaseMLModel):
    """Support Vector Machine Classification Model - metadata container"""
    
    def __init__(self, model_name: str = "svm_classifier"):
        super().__init__(model_name, "classification")

class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression Classification Model - metadata container"""
    
    def __init__(self, model_name: str = "logistic_regression"):
        super().__init__(model_name, "classification") 