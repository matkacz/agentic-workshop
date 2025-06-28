# 🧪 Workshop Testing Guide

## Pre-Workshop System Check

Before starting the workshop, verify the base system is working:

```bash
# 1. Check Python environment
python --version  # Should be 3.9+

# 2. Verify dependencies
pip list | grep -E "(google-adk|pandas|scikit-learn|streamlit)"

# 3. Test API key
python -c "import os; print('API Key:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'MISSING')"

# 4. Check data files
ls -la data/  # Should show personality_dataset.csv and other files
```

## Workshop Testing Steps

### Phase 1: After Agent Creation

```python
# Test 1: Import and Agent Creation
from agents import root_agent

# Should print agent details
print(f"Agent Name: {root_agent.name}")
print(f"Model: {root_agent.model}")
print(f"Number of Tools: {len(root_agent.tools)}")
print(f"Tool Names: {[tool.__name__ for tool in root_agent.tools]}")

# Expected Output:
# Agent Name: ml_agentic_system
# Model: gemini-2.0-flash-exp
# Number of Tools: 5
# Tool Names: ['train_ml_model', 'predict_with_model', 'list_available_models', 'list_available_datasets', 'detect_target_column_from_query']
```

### Phase 2: After Prompt Creation

```python
# Test 2: Basic Agent Response (CLI)
python cli.py

# Try these commands in the CLI:
```

**Test Commands for CLI:**

1. **General Question (No tools):**
   ```
   > What is the difference between Random Forest and SVM?
   ```
   Expected: Educational explanation without tool calls

2. **Dataset Discovery:**
   ```
   > What datasets are available?
   ```
   Expected: Calls `list_available_datasets` tool

3. **Model Management:**
   ```
   > What models do I have?
   ```
   Expected: Calls `list_available_models` tool

### Phase 3: Full System Testing

#### Test 3: Training a Model

```
> Train a classification model using data/personality_dataset.csv
```

**Expected Behavior:**
1. Calls `detect_target_column_from_query` tool
2. Calls `train_ml_model` tool with:
   - `task_type="classification"`
   - `model_type="random_forest"` (default)
   - `dataset_path="data/personality_dataset.csv"`
   - `target_column="personality"` (detected)

**Expected Output:**
```
✅ Successfully trained random_forest classification model (RandomForestClassifier) with accuracy: 0.XXX
📊 Model saved as: random_forest_classifier_YYYYMMDD_HHMMSS
🎯 Training completed with 1000 samples and 20 features
```

#### Test 4: Making Predictions

```
> Predict personality using data/inference_personality.csv
```

**Expected Behavior:**
1. Calls `predict_with_model` tool with:
   - `task_type="classification"`
   - `input_data="data/inference_personality.csv"`

**Expected Output:**
```
🔮 Made 50 predictions using random_forest_classifier_YYYYMMDD_HHMMSS
📋 Predictions: [Extrovert, Introvert, Extrovert, ...]
📊 Confidence scores available
```

#### Test 5: JSON Prediction

```
> Classify this data: {"age": 25, "income": 50000, "extroversion": 0.8}
```

**Expected Behavior:**
1. Calls `predict_with_model` tool with JSON input

**Expected Output:**
```
🎯 Prediction: Extrovert
🔢 Confidence: 0.85
```

## Streamlit Testing (Optional)

```bash
# Start Streamlit app
streamlit run app.py

# Test in browser at http://localhost:8501
```

**Browser Test Commands:**
1. "Train a classification model"
2. "What models are available?"
3. "Predict with test data"

## Troubleshooting Common Issues

### Issue 1: Import Errors
```python
# Debug imports
try:
    from agents import root_agent
    print("✅ Agent import successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Check file structure and __init__.py files
```

### Issue 2: Tool Not Found
```python
# Check tool availability
from agents.ml_tools import train_ml_model
print(f"✅ Tool available: {train_ml_model.__name__}")
```

### Issue 3: API Key Issues
```python
# Test API connection
from config import Config
try:
    Config.validate()
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Config error: {e}")
```

### Issue 4: Agent Not Responding
```python
# Test minimal agent
from google.adk.agents import Agent
from config import Config

test_agent = Agent(
    name="test",
    model=Config.GEMINI_MODEL,
    description="Test agent",
    instruction="You are a helpful assistant. Always respond with 'Hello World!'"
)

print("✅ Minimal agent created successfully")
```

## Success Verification Checklist

### ✅ Phase 1 Complete
- [ ] Agent imports without errors
- [ ] Agent has correct name and model
- [ ] All 5 tools are attached
- [ ] No syntax errors in agent.py

### ✅ Phase 2 Complete
- [ ] ML_AGENT_PROMPT is defined
- [ ] Prompt contains all required sections
- [ ] No syntax errors in prompt.py
- [ ] Agent responds to general questions

### ✅ Phase 3 Complete
- [ ] Agent can train classification models
- [ ] Agent can list available models
- [ ] Agent can make predictions
- [ ] Agent handles errors gracefully
- [ ] All test commands work as expected

## Performance Benchmarks

**Expected Response Times:**
- General questions: 2-5 seconds
- Dataset listing: 1-3 seconds
- Model training: 10-30 seconds
- Predictions: 3-8 seconds

**Expected Accuracy:**
- Personality dataset: 65-85% accuracy
- Synthetic data: 85-95% accuracy

## Workshop Completion Verification

**Final Test Sequence:**
```python
# 1. Import verification
from agents import root_agent

# 2. Quick training test
# (Use CLI or Streamlit)
"Train a Random Forest classifier"

# 3. Prediction test
"Predict with test data"

# 4. Model management test
"Show me my trained models"

# 5. Educational test
"Explain Random Forest algorithm"
```

**Success Criteria:**
- All commands execute without errors
- Training produces reasonable accuracy (>60%)
- Predictions return decoded labels
- Educational responses are informative
- System matches production behavior

---

**Note for Instructors:** Keep this testing guide handy during the workshop to quickly verify each phase and troubleshoot issues. The success of the workshop depends on each phase building correctly on the previous one. 