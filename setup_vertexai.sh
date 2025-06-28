#!/bin/bash

echo "🚀 VertexAI Code Executor Setup Script"
echo "====================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ gcloud CLI found"

# Get project ID
echo ""
echo "📋 Current gcloud configuration:"
gcloud config list --format="table(section,property,value)"

echo ""
read -p "🔧 Enter your Google Cloud PROJECT_ID (or press Enter to use current): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ No project ID found. Please set one."
        exit 1
    fi
fi

echo "📦 Using project: $PROJECT_ID"

# Set project
echo "🔧 Setting project..."
gcloud config set project $PROJECT_ID

# Enable APIs
echo "🔌 Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable notebooks.googleapis.com

# Set up authentication
echo "🔐 Setting up authentication..."
echo "Choose authentication method:"
echo "1) Application Default Credentials (Recommended)"
echo "2) Service Account"
read -p "Enter choice (1 or 2): " AUTH_CHOICE

if [ "$AUTH_CHOICE" = "1" ]; then
    echo "🔑 Setting up Application Default Credentials..."
    gcloud auth application-default login
elif [ "$AUTH_CHOICE" = "2" ]; then
    echo "🔑 Setting up Service Account..."
    SERVICE_ACCOUNT="adk-workshop"
    
    # Create service account
    gcloud iam service-accounts create $SERVICE_ACCOUNT --display-name="ADK Workshop"
    
    # Add permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user"
    
    # Create key
    gcloud iam service-accounts keys create ~/adk-key.json \
        --iam-account="$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com"
    
    echo "export GOOGLE_APPLICATION_CREDENTIALS=~/adk-key.json" >> ~/.bashrc
    export GOOGLE_APPLICATION_CREDENTIALS=~/adk-key.json
    
    echo "✅ Service account key saved to ~/adk-key.json"
else
    echo "❌ Invalid choice"
    exit 1
fi

# Set environment variables
echo "🌍 Setting environment variables..."
echo "export PROJECT_ID=$PROJECT_ID" >> ~/.bashrc
echo "export LOCATION=us-central1" >> ~/.bashrc
export PROJECT_ID=$PROJECT_ID
export LOCATION=us-central1

# Test setup
echo ""
echo "🧪 Testing setup..."
echo "Current project: $(gcloud config get-value project)"
echo "Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"

echo ""
echo "✅ VertexAI setup complete!"
echo "🔄 Please restart your terminal or run: source ~/.bashrc"
echo "🚀 You can now use the VertexAI Code Executor!" 