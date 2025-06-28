# 🎓 ML Agent Workshop - Beginner Level

## 🎯 Workshop Overview

**Duration:** 60 minutes  
**Level:** Beginner (No ML or ADK experience required)  
**Goal:** Build a complete ML Classification Agent from scratch using Google ADK

### What Attendees Will Build
- A fully functional ML agent that can train classification models
- Complete agent setup with proper ADK integration
- Comprehensive agent instructions using prompt engineering
- Working system identical to the production version

### What Attendees Will Learn
- Google ADK concepts (Agent, Tools, Instructions)
- Prompt engineering for AI agents
- ML agent architecture and design
- Classification ML basics
- How to structure agentic AI systems

## 📋 Prerequisites

### For Attendees
- Basic Python knowledge (variables, functions, imports)
- Laptop with Python 3.9+ installed
- Google AI API key (get from [Google AI Studio](https://aistudio.google.com/app/apikey))
- Workshop materials downloaded

### For Instructors
- Complete working version of the system
- Projector/screen for live coding
- Workshop templates prepared
- Test environment ready

## 🛠️ Workshop Setup (10 minutes)

### Pre-Workshop Checklist
1. **Distribute workshop materials**
   ```
   workshop_templates/beginner/
   ├── agents/
   │   ├── agent.py          # Template with TODOs
   │   └── prompt.py         # Template with TODOs
   ├── README_WORKSHOP.md    # This file
   └── test_commands.md      # Testing commands
   ```

2. **Environment Setup**
   ```bash
   # Attendees should have these ready:
   pip install -r requirements.txt
   export GOOGLE_API_KEY=your_key_here
   ```

3. **Verify Base System**
   - Ensure all other files (ml_tools.py, config.py, etc.) are complete
   - Only agent.py and prompt.py should be templates

## 🚀 Workshop Timeline

### Phase 1: Agent Definition (20 minutes)

#### **Step 1: Understanding ADK Concepts (5 minutes)**
**Instructor explains:**
- What is Google ADK and why use it?
- Core concepts: Agent, Tools, Instructions, Sessions
- How agents make decisions and call tools
- The role of instructions in guiding agent behavior

#### **Step 2: Building agent.py (15 minutes)**

**File:** `workshop_templates/beginner/agents/agent.py`

**Guided Implementation:**
```python
# TODO 1: Import statements (5 minutes)
from google.adk.agents import Agent
from config import Config
from agents.ml_tools import train_ml_model, predict_with_model, list_available_models, list_available_datasets, detect_target_column_from_query
from agents.prompt import ML_AGENT_PROMPT

# TODO 2: Create root_agent (10 minutes)
root_agent = Agent(
    name="ml_agentic_system",
    model=Config.GEMINI_MODEL,
    description=(
        "An intelligent ML agent that can train classification models, make predictions, and provide comprehensive ML assistance. "
        "Specializes in classification tasks with multiple algorithms including Random Forest, SVM, and Logistic Regression."
    ),
    instruction=ML_AGENT_PROMPT,
    tools=[
        train_ml_model,
        predict_with_model,
        list_available_models,
        list_available_datasets,
        detect_target_column_from_query
    ]
)
```

**Teaching Points:**
- Explain each parameter and its purpose
- Discuss tool selection and why these 5 tools
- Show how the agent will use these tools
- Emphasize the importance of clear descriptions

### Phase 2: Agent Instructions (25 minutes)

#### **Step 3: Prompt Engineering Fundamentals (5 minutes)**
**Instructor explains:**
- What are agent instructions/prompts?
- Why detailed instructions matter for AI agents
- Structure of effective prompts
- Classification-only focus and reasoning

#### **Step 4: Building ML_AGENT_PROMPT (20 minutes)**

**File:** `workshop_templates/beginner/agents/prompt.py`

**Section-by-Section Implementation:**

**Section 1: Role and Capabilities (3 minutes)**
```python
**Your Role:** You are an expert ML Classification Assistant powered by Google ADK that helps users with classification machine learning tasks.

**Core Capabilities:**
1. **Train Classification Models:** Can train classification models using various algorithms (Random Forest, SVM, Logistic Regression)
2. **Make Predictions:** Can load existing models or train new ones for classification inference
3. **Model Management:** Can list, compare, and manage trained classification models
4. **Dataset Discovery:** Can find and analyze available datasets in the data folder
5. **General ML Advice:** Can answer questions about classification and machine learning concepts
```

**Section 2: Decision Making Process (3 minutes)**
```python
**Decision Making Process:**

**1. Analyze User Intent:**
- **Training Request:** Keywords like "train", "build", "create model", "learn from data", "classify", "classifier"
- **Prediction Request:** Keywords like "predict", "classify", "forecast", "inference", "estimate"
- **Model Management:** Keywords like "list models", "available models", "show trained models"
- **Dataset Discovery:** Keywords like "what data", "available datasets", "show datasets", "data files"
- **General Question:** ML concepts, best practices, algorithm selection, etc.
```

**Section 3: Action Routing (8 minutes) - MOST CRITICAL**
```python
**2. Route Actions Based on Intent:**

**A. For Training Requests:**
- **MUST call `train_ml_model` tool with ALL required parameters**
- **REQUIRED PARAMETERS (must always provide):**
  - `task_type`: ALWAYS 'classification' (only classification is supported)
  - `model_type`: MUST choose from available options below
- **MUST select appropriate model type (REQUIRED):**
  - **Classification**: 'random_forest', 'svm', 'logistic_regression'
  - **Default choice**: Use 'random_forest' if user doesn't specify

**B. For Prediction Requests:**
- **MUST call `predict_with_model` tool**
- **ALWAYS use task_type='classification'** (only classification is supported)

**C. For Model Management:**
- **MUST call `list_available_models` tool**

**D. For Dataset Discovery:**
- **MUST call `list_available_datasets` tool**

**E. For General Questions:**
- **DO NOT call any tools**
- Provide educational, accurate ML guidance focused on classification
```

**Section 4: Result Presentation (3 minutes)**
```python
**3. Presenting Results:**

**For Training Results:**
- Clearly state the model type and that it's a classification task
- Present key metrics prominently: Accuracy, Precision, Recall, F1-score
- Explain what the metrics mean in practical terms
- Provide the model name for future reference

**For Prediction Results:**
- **IMPORTANT**: Format results based on user's specific request
- **For classification predictions with label_mapping**: ALWAYS decode the predictions using the provided mapping
- Display both the decoded labels AND the confidence scores if available
```

**Section 5: Error Handling (2 minutes)**
```python
**4. Error Handling:**
- If a tool returns an error, explain the issue clearly
- Suggest specific fixes (e.g., correct data format, valid parameters)
- If user asks for regression, explain that only classification is supported

**5. Educational Approach:**
- Always explain what you're doing and why
- Provide context about algorithm choices for classification
- Be encouraging and supportive for beginners
```

**Section 6: Examples (1 minute)**
```python
**Available Models:**
- **Classification Only:** random_forest, svm, logistic_regression

**Example Interactions:**
- "Train a classification model to predict customer churn" → detect_target_column_from_query, then train_ml_model(task_type="classification", model_type="random_forest", ...)
- "Classify this data: {...}" → predict_with_model(task_type="classification", input_data=...)
- "What models do I have available?" → list_available_models
```

### Phase 3: Testing and Validation (15 minutes)

#### **Step 5: Test the Complete System (10 minutes)**

**Test Commands:**
```python
# Test 1: Agent Creation
from agents import root_agent
print(f"Agent: {root_agent.name}")
print(f"Tools: {len(root_agent.tools)}")

# Test 2: Basic Training
"Train a classification model using data/personality_dataset.csv"

# Test 3: Model Listing
"What models do I have available?"

# Test 4: Prediction
"Predict personality using test data"
```

#### **Step 6: Troubleshooting and Q&A (5 minutes)**

**Common Issues:**
- Import errors → Check file structure
- Missing API key → Verify environment setup
- Tool calling errors → Check prompt instructions
- Agent not responding → Verify agent creation

## 🎯 Learning Outcomes

### By the end of this workshop, attendees will:

1. **Understand ADK Architecture**
   - How agents, tools, and instructions work together
   - The role of sessions and state management
   - Tool calling patterns and best practices

2. **Master Prompt Engineering**
   - Structure effective agent instructions
   - Balance specificity with flexibility
   - Handle different user intents appropriately

3. **Grasp ML Agent Design**
   - Classification-focused system design
   - Tool selection and organization
   - Error handling and user experience

4. **Build Production-Ready Code**
   - Complete working ML agent system
   - Proper imports and dependencies
   - Testable and maintainable code

## 🔧 Instructor Tips

### **Preparation**
- Have the complete working system ready for reference
- Test all commands and examples beforehand
- Prepare backup solutions for common issues
- Have extra API keys available if needed

### **During Workshop**
- Live code alongside attendees
- Explain the "why" behind each decision
- Encourage questions and experimentation
- Show immediate results after each major section

### **Troubleshooting**
- Keep a checklist of common issues and solutions
- Have attendees work in pairs for peer support
- Use breakout rooms for individual help if needed
- Always test imports and basic functionality first

### **Engagement Strategies**
- Ask attendees to predict what will happen before testing
- Have them suggest improvements or modifications
- Encourage sharing of results and insights
- Connect concepts to real-world ML applications

## 📚 Additional Resources

### **For Further Learning**
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Scikit-learn Classification Tutorial](https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)

### **Extension Activities**
- Add new model types to the system
- Create custom datasets for testing
- Implement visualization features
- Build a web interface with Streamlit

## 🏆 Success Criteria

### **Workshop is successful if:**
- ✅ All attendees have a working ML agent
- ✅ Agents can train classification models successfully
- ✅ Agents can make predictions with trained models
- ✅ Attendees understand core ADK concepts
- ✅ Attendees can explain prompt engineering basics
- ✅ System produces same results as production version

### **Bonus Goals:**
- Attendees ask thoughtful questions about extensions
- Attendees suggest improvements or modifications
- Attendees express interest in building their own agents
- Attendees understand the broader potential of agentic AI

---

**Remember:** The goal is not just to build working code, but to understand the principles and concepts that make agentic AI systems effective. Focus on the "why" as much as the "how"! 