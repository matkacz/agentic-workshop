# 🤖 ML Classification Agent Workshop - Intermediate Level

**Duration:** 75 minutes  
**Target Audience:** Developers with some ML/Python experience  
**Goal:** Build a complete ML Classification Agent system using Google ADK

## 🎯 Workshop Overview

In this intermediate workshop, you'll implement a production-ready ML Classification Agent that can:
- Train classification models (Random Forest, SVM, Logistic Regression)
- Make intelligent predictions with automatic model selection
- Provide comprehensive ML assistance and education
- Handle real-world data with proper preprocessing

**Final System:** A complete ML agent compatible with ADK web interface with 5 classification tools and comprehensive prompt engineering.

## 📋 Prerequisites

- Python experience (intermediate level)
- Basic understanding of machine learning concepts
- Familiarity with APIs and tool development
- Google AI API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## 🏗️ Workshop Structure

### Phase 1: Agent Architecture (20 minutes)
**Objective:** Create the root agent with proper ADK integration

**Your Tasks:**
1. **Agent Definition** (12 min)
   - Implement `agents/agent.py` with root_agent variable
   - Import required modules and tools
   - Configure agent with proper parameters for classification focus

2. **Tool Integration** (8 min)
   - Connect all 5 ML tools to the agent
   - Ensure proper tool parameter passing
   - Validate agent initialization

**Key Requirements:**
- Must use `root_agent` variable name for ADK web compatibility
- Include all 5 classification tools: `train_ml_model`, `predict_with_model`, `list_available_models`, `list_available_datasets`, `detect_target_column_from_query`
- Use Gemini 2.0 Flash model from config
- Focus on classification-only functionality

### Phase 2: ML Tools Implementation (30 minutes)
**Objective:** Implement the core ML functionality with guided scaffolding

**Your Tasks:**
1. **Core Training Function** (18 min)
   - Implement `train_ml_model` with classification algorithms
   - Handle data preprocessing, categorical encoding, and scaling
   - Support Random Forest, SVM, and Logistic Regression
   - Proper model persistence and evaluation

2. **Utility Functions** (12 min)
   - Implement `list_available_models` for model discovery
   - Implement `list_available_datasets` for data exploration
   - Implement `detect_target_column_from_query` for smart detection

**Key Requirements:**
- Support Random Forest, SVM, and Logistic Regression
- Handle categorical features and missing values
- Proper model persistence and loading
- Classification-only focus (no regression)

**Note:** The `predict_with_model` function is already implemented and working - you don't need to modify it.

### Phase 3: Intelligent Prompt Engineering (20 minutes)
**Objective:** Create comprehensive agent instructions for smart ML assistance

**Your Tasks:**
1. **Role Definition & Intent Analysis** (10 min)
   - Define agent as expert ML Classification Assistant
   - Build decision logic for training vs prediction vs management requests
   - Create keyword-based routing system

2. **Tool Routing & Response Logic** (10 min)
   - Implement smart parameter handling for each tool
   - Create dataset and target column detection logic
   - Build comprehensive response formatting rules

**Key Requirements:**
- Only support classification (no regression)
- Smart target column detection with confidence levels
- Proper label decoding for classification predictions
- Educational explanations for all operations

### Phase 4: Testing & Validation (5 minutes)
**Objective:** Verify complete system functionality

**Your Tasks:**
1. **Integration Testing** (5 min)
   - Test complete training workflow
   - Test prediction workflow with label decoding
   - Verify ADK web compatibility

## 🎯 Success Criteria

Your implementation is complete when:
- ✅ Agent initializes without errors
- ✅ All 5 tools are properly connected
- ✅ Can train classification models successfully
- ✅ Can make predictions with label decoding
- ✅ Provides intelligent responses to various query types
- ✅ Compatible with ADK web interface
- ✅ Matches production system functionality

## 🚀 Getting Started

1. **Setup Environment:**
   ```bash
   cd workshop_templates/intermediate
   # Ensure you have the main project dependencies installed
   ```

2. **Configuration:**
   - Your `.env` file should already contain `GOOGLE_API_KEY`
   - All other components (ml_tools, config, etc.) are already implemented

3. **Implementation Files:**
   - `agents/agent.py` - Your agent implementation
   - `agents/ml_tools.py` - Your ML tools implementation
   - `agents/prompt.py` - Your prompt engineering

4. **Testing:**
   - Use `test_commands.md` for validation steps
   - Compare with instructor solutions when needed

## 📚 Key Concepts to Implement

### Agent Architecture
- **Root Agent Pattern:** ADK-compatible agent structure
- **Tool Integration:** Proper tool parameter passing
- **Model Configuration:** Gemini 2.0 Flash setup

### Prompt Engineering
- **Intent Classification:** Smart request routing
- **Parameter Extraction:** Automatic parameter detection
- **Response Formatting:** User-friendly output generation
- **Error Handling:** Graceful failure management

### ML Workflow
- **Training Pipeline:** Dataset → Preprocessing → Training → Evaluation
- **Prediction Pipeline:** Input → Model Loading → Preprocessing → Prediction → Decoding
- **Model Management:** Listing, selection, and metadata handling

## 🎓 Learning Outcomes

By completing this workshop, you will:
- ✅ Understand ADK agent architecture and tool integration
- ✅ Master prompt engineering for intelligent ML assistance
- ✅ Implement production-ready ML workflows
- ✅ Handle real-world data preprocessing challenges
- ✅ Create user-friendly AI interfaces

## 🆘 Getting Help

**Instructor Support:**
- Ask questions during designated help periods
- Use provided testing commands for validation
- Refer to instructor solutions if completely stuck

**Self-Help Resources:**
- Check `test_commands.md` for validation steps
- Review existing `ml_tools.py` for tool signatures
- Examine `config.py` for available settings

## 🏁 Next Steps

After completing this workshop:
1. **Test your implementation** thoroughly using provided commands
2. **Experiment** with different model types and datasets
3. **Extend** the system with additional features
4. **Deploy** using the web interface or CLI

---

**Remember:** The goal is to build the exact same production system through guided implementation. Focus on understanding the architecture while following the patterns that lead to the working solution.

**Time Management:**
- Phase 1: 20 minutes (Agent setup)
- Phase 2: 30 minutes (ML tools implementation)
- Phase 3: 20 minutes (Prompt engineering)  
- Phase 4: 5 minutes (Testing)

Good luck building your ML Classification Agent! 🤖✨ 