#!/usr/bin/env python3
"""
Command Line Interface for ML Agentic System
Powered by Google ADK and Gemini 2.0 Flash
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Google ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Local imports
from config import Config
from agents import root_agent

class MLAgentCLI:
    """Command Line Interface for the ML Agent"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.runner = None
        self.session_id = None
        self.session_service = None
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize the ADK agent and session"""
        try:
            print("🤖 Initializing ML Agentic System...")
            
            # Validate configuration
            Config.validate()
            print("✅ Configuration validated")
            
            # Create session service
            self.session_service = InMemorySessionService()
            print("✅ Session service created")
            
            # Create runner
            self.runner = Runner(
                agent=root_agent,
                app_name=Config.ADK_APP_NAME,
                session_service=self.session_service
            )
            print("✅ ADK Runner created")
            
            # Generate session ID
            self.session_id = f"cli_session_{int(time.time())}_{os.urandom(4).hex()}"
            
            # Create initial session
            self.session_service.create_session(
                app_name=Config.ADK_APP_NAME,
                user_id=Config.ADK_USER_ID,
                session_id=self.session_id,
                state={}
            )
            print(f"✅ Session created: {self.session_id[-12:]}")
            print("🚀 ML Agentic System ready!\n")
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            print("\n💡 Make sure you have:")
            print("  1. Set GOOGLE_API_KEY in your .env file")
            print("  2. Installed all dependencies: pip install -r requirements.txt")
            print("  3. Valid API key from https://aistudio.google.com/app/apikey")
            sys.exit(1)
    
    async def run_agent_async(self, user_message: str) -> str:
        """Run agent asynchronously and return final response"""
        try:
            content = genai_types.Content(
                role='user',
                parts=[genai_types.Part(text=user_message)]
            )
            
            final_response = ""
            async for event in self.runner.run_async(
                user_id=Config.ADK_USER_ID,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                        final_response = event.content.parts[0].text
                    break
            
            return final_response or "Agent completed but produced no response."
            
        except Exception as e:
            return f"❌ Error running agent: {str(e)}"
    
    def run_agent_sync(self, user_message: str) -> str:
        """Synchronous wrapper for async agent execution"""
        return asyncio.run(self.run_agent_async(user_message))
    
    def print_welcome(self):
        """Print welcome message"""
        print("=" * 70)
        print("🤖 ML AGENTIC SYSTEM - COMMAND LINE INTERFACE")
        print("=" * 70)
        print()
        print("Powered by Google ADK & Gemini 2.0 Flash")
        print()
        print("I can help you with:")
        print("🎯 Training ML models (classification & regression)")
        print("📊 Making predictions with trained models")
        print("📋 Managing and listing your models")
        print("🎓 Answering ML questions and providing guidance")
        print()
        print("💡 Examples:")
        print("  • 'Train a classification model'")
        print("  • 'Predict with test data'")
        print("  • 'What models do I have?'")
        print("  • 'Explain Random Forest algorithm'")
        print()
        print("Type 'help' for more commands or 'quit' to exit.")
        print("=" * 70)
        print()
    
    def print_help(self):
        """Print help information"""
        print()
        print("🆘 HELP - Available Commands")
        print("-" * 50)
        print()
        print("📝 TRAINING COMMANDS:")
        print("  • train classification model")
        print("  • train regression model")
        print("  • create a Random Forest classifier")
        print("  • build an SVM model")
        print()
        print("🔮 PREDICTION COMMANDS:")
        print("  • predict with test data")
        print("  • classify this data: {...}")
        print("  • make predictions using [model_name]")
        print()
        print("📊 MODEL MANAGEMENT:")
        print("  • list models")
        print("  • what models do I have?")
        print("  • show trained models")
        print()
        print("🎓 EDUCATIONAL QUERIES:")
        print("  • explain [ML concept]")
        print("  • difference between precision and recall")
        print("  • when to use Random Forest vs SVM?")
        print()
        print("⚙️  SYSTEM COMMANDS:")
        print("  • help     - Show this help")
        print("  • clear    - Clear screen")
        print("  • status   - Show system status")
        print("  • quit     - Exit the program")
        print()
        print("-" * 50)
        print()
    
    def print_status(self):
        """Print system status"""
        print()
        print("📊 SYSTEM STATUS")
        print("-" * 30)
        print(f"Model: {Config.GEMINI_MODEL}")
        print(f"Session: {self.session_id[-12:]}")
        print(f"Models Dir: {Config.MODELS_DIR}")
        print(f"Results Dir: {Config.RESULTS_DIR}")
        
        # Check for saved models
        models_dir = Path(Config.MODELS_DIR)
        if models_dir.exists():
            model_count = len([d for d in models_dir.iterdir() if d.is_dir()])
            print(f"Saved Models: {model_count}")
        else:
            print("Saved Models: 0")
        
        # Check for results
        results_dir = Path(Config.RESULTS_DIR)
        if results_dir.exists():
            results_count = len(list(results_dir.glob("*.png")))
            print(f"Generated Plots: {results_count}")
        else:
            print("Generated Plots: 0")
        
        print("-" * 30)
        print()
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run(self):
        """Main CLI loop"""
        self.print_welcome()
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("🤖 ML Agent > ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle system commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 Thanks for using ML Agentic System!")
                        print("Happy machine learning! 🤖✨")
                        break
                    
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    
                    elif user_input.lower() == 'clear':
                        self.clear_screen()
                        self.print_welcome()
                        continue
                    
                    elif user_input.lower() == 'status':
                        self.print_status()
                        continue
                    
                    # Process with ML agent
                    print("\n🤔 Processing... (this may take a moment for training tasks)")
                    
                    start_time = time.time()
                    response = self.run_agent_sync(user_input)
                    end_time = time.time()
                    
                    print(f"\n🤖 ML Agent ({end_time - start_time:.1f}s):")
                    print("-" * 50)
                    print(response)
                    print("-" * 50)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user. Type 'quit' to exit properly.")
                    continue
                
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    print("Type 'help' for usage information.")
                    continue
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        
        except Exception as e:
            print(f"\n💥 Fatal error: {str(e)}")
            sys.exit(1)

def main():
    """Main entry point"""
    try:
        cli = MLAgentCLI()
        cli.run()
    except Exception as e:
        print(f"💥 Failed to start CLI: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 