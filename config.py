import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the ML Agentic System"""
    
    # Google AI API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Google Cloud Configuration (optional)
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Directory Configuration
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
    TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", BASE_DIR / "data"))
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
    
    # ADK Configuration
    ADK_APP_NAME = os.getenv("ADK_APP_NAME", "ml_agentic_system")
    ADK_USER_ID = os.getenv("ADK_USER_ID", "default_user")
    
    # Model Configuration
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(exist_ok=True)
    TRAINING_DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required. Get it from https://aistudio.google.com/app/apikey")
        return True 