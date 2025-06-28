import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO

class VisualizationUtils:
    """Utility class for creating ML visualizations"""
    
    @staticmethod
    def setup_matplotlib_style():
        """Setup consistent matplotlib styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
        """Plot training and validation loss/metrics over epochs"""
        VisualizationUtils.setup_matplotlib_style()
        
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return "No training history to plot"
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(history[metric], label=f'Training {metric}', linewidth=2)
            axes[i].set_title(f'{metric.capitalize()} Over Epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> str:
        """Create confusion matrix visualization"""
        VisualizationUtils.setup_matplotlib_style()
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray,
                              max_features: int = 20, save_path: Optional[str] = None) -> str:
        """Plot feature importance"""
        VisualizationUtils.setup_matplotlib_style()
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:max_features]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        plt.figure(figsize=(10, max(6, len(sorted_features) * 0.3)))
        bars = plt.barh(range(len(sorted_features)), sorted_scores)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    @staticmethod
    def plot_data_distribution(df: pd.DataFrame, target_column: str,
                             max_features: int = 6, save_path: Optional[str] = None) -> str:
        """Plot data distribution for features and target"""
        VisualizationUtils.setup_matplotlib_style()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        # Limit number of features to plot
        features_to_plot = numeric_columns[:max_features]
        n_features = len(features_to_plot)
        
        if n_features == 0:
            return "No numeric features to plot"
        
        fig, axes = plt.subplots(2, max(3, (n_features + 1) // 2), figsize=(15, 8))
        axes = axes.flatten()
        
        # Plot feature distributions
        for i, feature in enumerate(features_to_plot):
            if i < len(axes) - 1:
                axes[i].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Plot target distribution
        if len(axes) > n_features:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                axes[n_features].hist(df[target_column], bins=30, alpha=0.7, edgecolor='black')
            else:
                df[target_column].value_counts().plot(kind='bar', ax=axes[n_features])
            axes[n_features].set_title(f'Distribution of {target_column} (Target)')
            axes[n_features].set_xlabel(target_column)
            axes[n_features].set_ylabel('Frequency')
            axes[n_features].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    @staticmethod
    def create_model_comparison_chart(models_results: Dict[str, Dict[str, float]],
                                    save_path: Optional[str] = None) -> str:
        """Create comparison chart for multiple models"""
        VisualizationUtils.setup_matplotlib_style()
        
        if not models_results:
            return "No model results to compare"
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(models_results).T
        metrics = df.columns.tolist()
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(df.index, df[metric])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}" 