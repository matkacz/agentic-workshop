import streamlit as st
import asyncio
import time
import os
from pathlib import Path
import base64
from io import BytesIO

# Google ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Local imports
from config import Config
from agents import root_agent

# Page configuration
st.set_page_config(
    page_title="ML Agentic System - Google ADK",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    .success-message {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    .code-block {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ADK components
@st.cache_resource
def initialize_adk():
    """Initialize ADK Runner and Session Service"""
    try:
        # Validate configuration
        Config.validate()
        
        # Create session service
        session_service = InMemorySessionService()
        
        # Create runner
        runner = Runner(
            agent=root_agent,
            app_name=Config.ADK_APP_NAME,
            session_service=session_service
        )
        
        # Generate session ID
        session_id = f"streamlit_session_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Create initial session
        session_service.create_session(
            app_name=Config.ADK_APP_NAME,
            user_id=Config.ADK_USER_ID,
            session_id=session_id,
            state={}
        )
        
        return runner, session_id, True, "ADK initialized successfully!"
        
    except Exception as e:
        return None, None, False, f"ADK initialization failed: {str(e)}"

# Async runner wrapper
async def run_agent_async(runner: Runner, session_id: str, user_message: str) -> str:
    """Run agent asynchronously and return final response"""
    try:
        content = genai_types.Content(
            role='user',
            parts=[genai_types.Part(text=user_message)]
        )
        
        final_response = ""
        async for event in runner.run_async(
            user_id=Config.ADK_USER_ID,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response = event.content.parts[0].text
                break
        
        return final_response or "Agent completed but produced no response."
        
    except Exception as e:
        return f"Error running agent: {str(e)}"

def run_agent_sync(runner: Runner, session_id: str, user_message: str) -> str:
    """Synchronous wrapper for async agent execution"""
    return asyncio.run(run_agent_async(runner, session_id, user_message))

def display_image_from_base64(base64_string: str, caption: str = ""):
    """Display base64 encoded image in Streamlit"""
    if base64_string.startswith('data:image'):
        # Extract base64 data
        base64_data = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_data)
        st.image(image_data, caption=caption, use_column_width=True)

def format_metrics_display(metrics: dict, title: str):
    """Format and display metrics in a nice layout"""
    st.subheader(title)
    
    # Create columns for metrics
    cols = st.columns(len(metrics))
    
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=f"{value:.4f}"
                )
            else:
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=str(value)
                )

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 ML Agentic System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #666;'>Powered by Google ADK & Gemini 2.0 Flash</p>", unsafe_allow_html=True)
    
    # Initialize ADK
    runner, session_id, init_success, init_message = initialize_adk()
    
    if not init_success:
        st.markdown(f"<div class='error-message'>{init_message}</div>", unsafe_allow_html=True)
        st.stop()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🎛️ System Information")
        
        st.markdown(f"**Status:** {'🟢 Active' if init_success else '🔴 Error'}")
        st.markdown(f"**Model:** {Config.GEMINI_MODEL}")
        st.markdown(f"**Session:** `{session_id[-12:]}`")
        
        st.divider()
        
        st.header("📚 Quick Examples")
        
        example_prompts = [
            "Train a classification model to predict customer churn",
            "Train a regression model to predict house prices", 
            "What models do I have available?",
            "Predict with test data using classification",
            "Explain the difference between precision and recall",
            "Create a Random Forest model for classification"
        ]
        
        for prompt in example_prompts:
            if st.button(f"💡 {prompt[:30]}...", key=f"example_{hash(prompt)}", use_container_width=True):
                st.session_state.example_prompt = prompt
        
        st.divider()
        
        st.header("🛠️ Available Models")
        st.markdown("""
        **Classification:**
        - Random Forest
        - SVM  
        - Logistic Regression
        
        **Regression:**
        - Random Forest Regressor
        - SVR
        - Linear Regression
        """)
        
        st.divider()
        
        st.header("📁 Directory Structure")
        st.markdown(f"""
        - **Models:** `{Config.MODELS_DIR}`
        - **Data:** `{Config.TRAINING_DATA_DIR}`  
        - **Results:** `{Config.RESULTS_DIR}`
        """)
    
    # Main chat interface
    st.header("💬 Chat with ML Agent")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """
👋 **Welcome to the ML Agentic System!**

I'm your AI-powered ML assistant. I can help you with:

🎯 **Training Models:** Create classification or regression models with various algorithms
📊 **Making Predictions:** Use existing models or train new ones for inference  
📋 **Model Management:** List, compare, and manage your trained models
🎓 **ML Education:** Answer questions about machine learning concepts

**What would you like to do today?**
- Train a model on your data
- Make predictions with existing models
- Learn about machine learning concepts
- Explore available models

Try asking something like: *"Train a classification model"* or *"What models do I have?"*
                """
            }
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle example prompt
    if "example_prompt" in st.session_state:
        user_input = st.session_state.example_prompt
        del st.session_state.example_prompt
    else:
        user_input = st.chat_input("Ask me anything about machine learning...")
    
    # Process user input
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking... (this may take a moment for training/prediction tasks)"):
                try:
                    response = run_agent_sync(runner, session_id, user_input)
                    
                    # Display the response
                    st.markdown(response)
                    
                    # Add to message history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Additional features section
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Recent Results")
        
        # Check for recent visualizations in results directory
        results_dir = Path(Config.RESULTS_DIR)
        if results_dir.exists():
            recent_files = sorted(
                [f for f in results_dir.glob("*.png")], 
                key=lambda x: x.stat().st_mtime, 
                reverse=True
            )[:3]  # Show 3 most recent
            
            if recent_files:
                for file_path in recent_files:
                    st.subheader(f"📈 {file_path.stem}")
                    try:
                        st.image(str(file_path), use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not load image: {e}")
            else:
                st.info("No visualizations yet. Train a model to see results!")
        else:
            st.info("Results directory not found.")
    
    with col2:
        st.header("🗂️ Model Storage")
        
        # Check for saved models
        models_dir = Path(Config.MODELS_DIR)
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            
            if model_dirs:
                st.success(f"Found {len(model_dirs)} saved models")
                for model_dir in sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    with st.expander(f"📦 {model_dir.name}"):
                        # Try to load metadata
                        metadata_file = model_dir / "metadata.json"
                        if metadata_file.exists():
                            try:
                                import json
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                st.json(metadata)
                            except Exception as e:
                                st.error(f"Could not load metadata: {e}")
                        else:
                            st.info("No metadata available")
            else:
                st.info("No models saved yet. Train a model to get started!")
        else:
            st.info("Models directory not found.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 <strong>ML Agentic System</strong> | Powered by Google ADK & Gemini 2.0 Flash</p>
        <p>Built with Streamlit • Scikit-learn • Matplotlib</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 