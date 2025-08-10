#!/bin/bash
# Credit Card Fraud Detection - Local Setup and Run Script

echo "🚀 Credit Card Fraud Detection - Setup Script"
echo "=============================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Step 1: Create virtual environment
echo ""
echo "📦 Step 1: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo ""
echo "🔄 Step 2: Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Step 3: Install dependencies
echo ""
echo "📦 Step 3: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Step 4: Check for dataset
echo ""
echo "📊 Step 4: Checking for dataset..."
if [ -f "data/creditcard.csv" ]; then
    echo "✅ Dataset found: data/creditcard.csv"
    DATASET_AVAILABLE=true
else
    echo "⚠️  Dataset not found: data/creditcard.csv"
    echo "📥 Please download the dataset from Kaggle:"
    echo "   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo "   Place it as: data/creditcard.csv"
    DATASET_AVAILABLE=false
fi

# Step 5: Run tests
echo ""
echo "🧪 Step 5: Running tests..."
python tests/test_preprocess.py
python tests/test_inference.py
echo "✅ Tests completed"

# Step 6: Train models (if dataset available)
if [ "$DATASET_AVAILABLE" = true ]; then
    echo ""
    echo "🤖 Step 6: Training models..."
    echo "⏳ This may take several minutes..."
    python -m src.train
    
    if [ $? -eq 0 ]; then
        echo "✅ Model training completed successfully"
        MODELS_AVAILABLE=true
    else
        echo "❌ Model training failed"
        MODELS_AVAILABLE=false
    fi
else
    echo ""
    echo "⏭️  Step 6: Skipping model training (dataset not available)"
    MODELS_AVAILABLE=false
fi

# Step 7: Launch Streamlit app
echo ""
echo "🎮 Step 7: Launching Streamlit demo..."
echo ""
echo "🌟 Setup Summary:"
echo "   • Virtual environment: ✅ Ready"
echo "   • Dependencies: ✅ Installed"
echo "   • Dataset: $([ "$DATASET_AVAILABLE" = true ] && echo "✅ Available" || echo "❌ Missing")"
echo "   • Models: $([ "$MODELS_AVAILABLE" = true ] && echo "✅ Trained" || echo "❌ Not trained")"
echo ""

if [ "$MODELS_AVAILABLE" = true ]; then
    echo "🚀 Launching Streamlit app with trained models..."
else
    echo "🔧 Launching Streamlit app in demo mode..."
    echo "   (You can explore the interface, but predictions require trained models)"
fi

echo ""
echo "🌐 Opening browser at: http://localhost:8501"
echo "📝 Press Ctrl+C to stop the app"
echo ""

# Launch Streamlit
streamlit run app/streamlit_app.py

echo ""
echo "👋 Thanks for using Credit Card Fraud Detection!"
echo "🐛 Report issues at: https://github.com/your-repo/issues"
