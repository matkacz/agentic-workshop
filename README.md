# 🤖 ML Classification Agent System

An intelligent **classification-focused** machine learning system powered by **Google Agent Development Kit (ADK)** and **Gemini 2.0 Flash** that can autonomously train classification models, make predictions, and provide ML expertise through natural language conversations.

## 🌟 Key Features

### 🎯 **Intelligent Task Routing**
- **Training Requests**: Automatically detects when you want to train a classification model and selects appropriate algorithms
- **Prediction Requests**: Finds existing models or trains new ones for classification inference tasks
- **Model Management**: Lists, compares, and manages your trained classification models
- **General AI Assistant**: Answers classification ML questions and provides educational guidance

### 🤖 **Autonomous Decision Making**
- **Smart Model Selection**: Chooses the best classification algorithm based on your task
- **Auto-Training**: Creates new classification models when none exist for your prediction requests
- **Data Handling**: Processes various data formats (CSV, Excel, JSON) or generates synthetic data
- **Visualization**: Automatically creates plots, confusion matrices, and training visualizations

### 📊 **Comprehensive Classification Support**

#### Classification Models:
- **Random Forest Classifier** - Robust, handles mixed data types, excellent default choice
- **Support Vector Machine (SVM)** - Great for complex decision boundaries and high-dimensional data
- **Logistic Regression** - Fast, interpretable linear classification with probability outputs

#### Rich Visualizations:
- Training metrics and validation curves
- Feature importance plots
- Confusion matrices for classification performance
- Classification reports with precision, recall, F1-score
- Data distribution visualizations

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Google AI API Key (get from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Installation

```bash
# Clone or download the project
cd google-adk-experiment

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
# Required: Google AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Custom directories
MODELS_DIR=./models
TRAINING_DATA_DIR=./data
RESULTS_DIR=./results

# Optional: ADK configuration
ADK_APP_NAME=ml_agentic_system
ADK_USER_ID=default_user
```

### 4. Run the Application

#### Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

#### Command Line Interface
```bash
python cli.py
```

## 🎯 Usage Examples

### 💬 Natural Language Interactions

The system understands natural language and automatically routes your classification requests:

#### Training Classification Models
```
"Train a classification model to predict customer churn"
"Create a spam classifier using SVM"
"Build a Random Forest classifier to predict personality types"
"Train a logistic regression model for binary classification"
```

#### Making Classification Predictions
```
"Classify this data: {'age': 35, 'income': 50000, 'score': 0.8}"
"Predict the category for this customer profile"
"Make predictions using data/inference_personality.csv"
"Classify the data in data/test_data.csv"
"What class would this sample belong to?"
```

#### Model Management
```
"What classification models do I have available?"
"Show me my trained models"
"List all my classifiers"
"Which model has the best accuracy?"
```

#### Educational Queries
```
"Explain the difference between precision and recall"
"When should I use Random Forest vs SVM for classification?"
"How do I interpret a confusion matrix?"
"What is the difference between binary and multi-class classification?"
```

### 🔧 Programmatic Usage

You can also use the system programmatically:

```python
from agents import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Initialize the agent
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service)

# Train a classification model
response = runner.run("Train a Random Forest classifier")

# Make classification predictions  
response = runner.run("Classify this test data")
```

## 📁 Project Structure

```
google-adk-experiment/
├── 📱 app.py                    # Streamlit web interface
├── 🖥️  cli.py                   # Command line interface  
├── ⚙️  config.py                # Configuration management
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # This file
├── 
├── 🤖 agents/
│   ├── agent.py                 # Root ADK agent definition
│   ├── prompt.py                # Agent instructions/prompt
│   ├── ml_tools.py              # Classification operation tools
│   └── __init__.py
├── 
├── 🧠 ml_models/
│   ├── base_model.py            # Abstract base model class
│   ├── classification_models.py # Classification algorithms
│   ├── model_manager.py         # Model persistence & loading
│   └── __init__.py
├── 
├── 🛠️ utils/
│   ├── data_utils.py            # Data processing utilities
│   ├── visualization_utils.py   # Plotting and visualization
│   └── __init__.py
├── 
├── 💾 models/                   # Saved trained classification models
├── 📊 data/                     # Training datasets
└── 📈 results/                  # Generated visualizations
```

## 🔧 System Architecture

### 🏗️ **Google ADK Integration**

The system leverages Google ADK's powerful agent framework:

- **Agent**: Core classification ML assistant with comprehensive instructions (defined in `agents/agent.py`)
- **Tools**: Classification ML operations (train, predict, list models)
- **Runner**: Orchestrates agent execution and tool calls
- **Session Management**: Maintains conversation state and context

### 🔄 **Intelligent Workflow**

1. **User Input** → Natural language query about classification
2. **Intent Analysis** → Gemini 2.0 Flash determines classification intent
3. **Tool Selection** → ADK routes to appropriate classification ML tools
4. **Execution** → Train classification models, make predictions, or provide info
5. **Response** → Formatted results with classification metrics and visualizations
6. **State Management** → Remembers classification models and context

### 🧩 **Modular Design**

- **Pluggable Models**: Easy to add new classification algorithms
- **Flexible Data**: Supports multiple input formats for classification
- **Rich Visualizations**: Automatic confusion matrices and classification plots
- **Extensible Tools**: Simple to add new classification capabilities

## 📊 Supported Data Formats

### Input Data
- **CSV files** - Comma-separated values (for training and prediction)
- **Excel files** - .xlsx, .xls formats
- **JSON files** - Structured data
- **JSON strings** - For real-time predictions
- **Synthetic data** - Auto-generated for demos

### Model Outputs
- **Saved models** - Persistent .joblib files
- **Metadata** - JSON model information
- **Visualizations** - PNG plots and charts
- **Metrics** - Comprehensive performance data

## 🎨 Visualization Gallery

The system automatically generates beautiful visualizations:

### 📈 **Training Visualizations**
- Training and validation loss curves
- Cross-validation score distributions
- Learning curves over epochs

### 🎯 **Model Performance**
- Confusion matrices for classification
- ROC curves and precision-recall curves
- Classification reports with precision, recall, F1-score
- Feature importance rankings

### 📊 **Data Analysis**
- Data distribution histograms
- Correlation matrices
- Missing value patterns
- Class distribution analysis

### 📋 **Model Comparison**
- Side-by-side metric comparisons
- Performance benchmarking charts
- Algorithm comparison tables

## 🔒 Best Practices

### 🛡️ **Security**
- Store API keys in `.env` files (never in code)
- Use environment variables for sensitive configuration
- Validate all user inputs before processing

### 📈 **Performance**
- Models are cached for fast repeated access
- Visualizations are generated asynchronously
- Large datasets are processed in chunks

### 🧪 **Development**
- Modular architecture for easy extensions
- Comprehensive error handling and logging
- Type hints and documentation throughout

## 🛠️ Extending the System

### Adding New Models

1. Create a new model class inheriting from `BaseMLModel`:

```python
from ml_models.base_model import BaseMLModel

class MyCustomModel(BaseMLModel):
    def __init__(self, model_name: str = "my_model"):
        super().__init__(model_name, "classification")
    
    def train(self, X, y, **kwargs):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
```

2. Register in `ModelManager`:

```python
self.model_registry = {
    'classification': {
        'my_model': MyCustomModel,
        # ... existing models
    }
}
```

### Adding New Tools

1. Create a tool function in `agents/ml_tools.py`:

```python
def my_new_tool(tool_context: ToolContext, param: str) -> Dict[str, Any]:
    """
    Description of what the tool does.
    """
    # Implement tool logic
    return {'status': 'success', 'result': 'data'}
```

2. Add to agent tools in `agents/agent.py`:

```python
root_agent = Agent(
    # ... other parameters ...
    tools=[
        train_ml_model,
        predict_with_model,
        list_available_models,
        list_available_datasets,
        detect_target_column_from_query,
        my_new_tool  # Add your new tool
    ]
)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **New Algorithms**: Deep learning models, ensemble methods
- **Data Sources**: Database connectors, API integrations  
- **Visualizations**: Interactive plots, 3D visualizations
- **Deployment**: Cloud deployment, containerization
- **Testing**: Unit tests, integration tests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Google ADK Team** - For the amazing agent development framework
- **Google AI** - For Gemini 2.0 Flash model access
- **Streamlit** - For the beautiful web interface framework
- **Scikit-learn** - For comprehensive ML algorithms
- **Matplotlib/Seaborn** - For visualization capabilities

---

## 🚀 Ready to Start?

1. **Get your Google AI API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Set up the environment** with the steps above
3. **Run the app** with `streamlit run app.py`
4. **Start chatting** with your AI ML assistant!

**Example first query:** *"Train a classification model and show me the results"*

Happy machine learning! 🤖✨ 