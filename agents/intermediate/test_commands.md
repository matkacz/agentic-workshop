# 🧪 Testing Commands - Intermediate Workshop

Use these commands to validate your ML Classification Agent implementation.

## 🔧 Basic Validation

### 1. Agent Initialization Test
```bash
cd workshop_templates/intermediate
python -c "from agents.agent import root_agent; print('✅ Agent loaded:', root_agent.name)"
```
**Expected:** Should print agent name without errors

### 2. Tools Integration Test
```bash
python -c "from agents.agent import root_agent; print('✅ Tools count:', len(root_agent.tools)); print('✅ Tool names:', [t.__name__ for t in root_agent.tools])"
```
**Expected:** Should show 5 tools: train_ml_model, predict_with_model, list_available_models, list_available_datasets, detect_target_column_from_query

### 3. Prompt Validation Test
```bash
python -c "from agents.prompt import ML_AGENT_PROMPT; print('✅ Prompt length:', len(ML_AGENT_PROMPT), 'characters'); print('✅ Contains classification:', 'classification' in ML_AGENT_PROMPT.lower())"
```
**Expected:** Should show substantial prompt length (3000+ characters) and contain 'classification'

## 🚀 Functional Testing

### 4. CLI Integration Test
```bash
cd ../..  # Return to main project directory
python cli.py
```
**Test Queries:**
- `"What models do I have available?"`
- `"Train a classification model"`
- `"Explain Random Forest algorithm"`
- Type `quit` to exit

**Expected:** Agent should respond appropriately to each query type

### 5. Streamlit Web Interface Test
```bash
streamlit run app.py
```
**Test in Browser:**
- Navigate to http://localhost:8501
- Try: "List my available models"
- Try: "Train a Random Forest classifier"
- Try: "What is the difference between precision and recall?"

**Expected:** Web interface should load and agent should respond correctly

## 🎯 Advanced Validation

### 6. Training Workflow Test
```bash
python -c "
from agents import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
import asyncio

async def test_training():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service)
    session_id = 'test_session'
    session_service.create_session('ml_agentic_system', 'test_user', session_id, {})
    
    content = genai_types.Content(role='user', parts=[genai_types.Part(text='Train a Random Forest classification model')])
    
    async for event in runner.run_async('test_user', session_id, content):
        if event.is_final_response():
            print('✅ Training test completed')
            print('Response length:', len(event.content.parts[0].text) if event.content and event.content.parts else 0)
            break

asyncio.run(test_training())
"
```
**Expected:** Should complete training workflow without errors

### 7. Prediction Workflow Test
```bash
python -c "
from agents import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
import asyncio

async def test_prediction():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service)
    session_id = 'test_session_2'
    session_service.create_session('ml_agentic_system', 'test_user', session_id, {})
    
    content = genai_types.Content(role='user', parts=[genai_types.Part(text='Make predictions with test data')])
    
    async for event in runner.run_async('test_user', session_id, content):
        if event.is_final_response():
            print('✅ Prediction test completed')
            print('Response contains predictions:', 'prediction' in event.content.parts[0].text.lower() if event.content and event.content.parts else False)
            break

asyncio.run(test_prediction())
"
```
**Expected:** Should handle prediction requests appropriately

## 🔍 Troubleshooting Guide

### Common Issues & Solutions

**Issue:** `ImportError: cannot import name 'root_agent'`
- **Solution:** Check that `agents/agent.py` defines `root_agent` variable correctly
- **Verify:** Ensure all imports are correct and Agent class is properly instantiated

**Issue:** `AttributeError: 'NoneType' object has no attribute 'tools'`
- **Solution:** Verify Agent initialization with all required parameters
- **Check:** Ensure tools list contains all 5 functions

**Issue:** `NameError: name 'ML_AGENT_PROMPT' is not defined`
- **Solution:** Check that `agents/prompt.py` defines `ML_AGENT_PROMPT` variable
- **Verify:** Ensure the prompt string is properly formatted and comprehensive

**Issue:** Agent responds with "I don't understand" or similar
- **Solution:** Check prompt completeness and intent analysis logic
- **Verify:** Ensure all tool routing instructions are included

**Issue:** Tools not being called properly
- **Solution:** Verify tool parameter specifications in prompt
- **Check:** Ensure task_type='classification' is specified for ML tools

## ✅ Success Checklist

Your implementation is complete when all tests pass:

- [ ] Agent loads without import errors
- [ ] All 5 tools are properly connected
- [ ] Prompt is comprehensive (3000+ characters)
- [ ] CLI interface works with various queries
- [ ] Web interface loads and responds correctly
- [ ] Training workflow completes successfully
- [ ] Prediction workflow handles requests appropriately
- [ ] Agent provides educational responses for general questions
- [ ] Classification focus is maintained throughout
- [ ] Error handling works gracefully

## 🎯 Final Validation

Run this comprehensive test to verify everything works:

```bash
python -c "
print('🧪 COMPREHENSIVE VALIDATION TEST')
print('=' * 50)

try:
    from agents.agent import root_agent
    print('✅ 1. Agent import successful')
    
    print(f'✅ 2. Agent name: {root_agent.name}')
    print(f'✅ 3. Tools count: {len(root_agent.tools)}')
    
    from agents.prompt import ML_AGENT_PROMPT
    print(f'✅ 4. Prompt length: {len(ML_AGENT_PROMPT)} characters')
    
    tool_names = [t.__name__ for t in root_agent.tools]
    expected_tools = ['train_ml_model', 'predict_with_model', 'list_available_models', 'list_available_datasets', 'detect_target_column_from_query']
    missing_tools = [t for t in expected_tools if t not in tool_names]
    
    if not missing_tools:
        print('✅ 5. All required tools present')
    else:
        print(f'❌ 5. Missing tools: {missing_tools}')
    
    if 'classification' in ML_AGENT_PROMPT.lower():
        print('✅ 6. Classification focus confirmed')
    else:
        print('❌ 6. Classification focus missing from prompt')
    
    print('=' * 50)
    print('🎉 VALIDATION COMPLETE - CHECK RESULTS ABOVE')
    
except Exception as e:
    print(f'❌ VALIDATION FAILED: {e}')
"
```

**Expected Output:** All checks should show ✅ for successful implementation.

---

**Need Help?** 
- Review the workshop README for implementation guidance
- Check instructor solutions if completely stuck
- Ask questions during designated help periods

**Ready for Production?** Your intermediate implementation should match the production system functionality exactly! 🚀 