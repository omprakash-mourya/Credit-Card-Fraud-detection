@echo off
REM Credit Card Fraud Detection - Local Setup and Run Script (Windows)

echo 🚀 Credit Card Fraud Detection - Setup Script
echo ==============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Step 1: Create virtual environment
echo.
echo 📦 Step 1: Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Step 2: Activate virtual environment
echo.
echo 🔄 Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Step 3: Install dependencies
echo.
echo 📦 Step 3: Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✅ Dependencies installed

REM Step 4: Check for dataset
echo.
echo 📊 Step 4: Checking for dataset...
if exist "data\creditcard.csv" (
    echo ✅ Dataset found: data\creditcard.csv
    set DATASET_AVAILABLE=true
) else (
    echo ⚠️  Dataset not found: data\creditcard.csv
    echo 📥 Please download the dataset from Kaggle:
    echo    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo    Place it as: data\creditcard.csv
    set DATASET_AVAILABLE=false
)

REM Step 5: Run tests
echo.
echo 🧪 Step 5: Running tests...
python tests\test_preprocess.py
python tests\test_inference.py
echo ✅ Tests completed

REM Step 6: Train models (if dataset available)
if "%DATASET_AVAILABLE%"=="true" (
    echo.
    echo 🤖 Step 6: Training models...
    echo ⏳ This may take several minutes...
    python -m src.train
    
    if errorlevel 1 (
        echo ❌ Model training failed
        set MODELS_AVAILABLE=false
    ) else (
        echo ✅ Model training completed successfully
        set MODELS_AVAILABLE=true
    )
) else (
    echo.
    echo ⏭️  Step 6: Skipping model training ^(dataset not available^)
    set MODELS_AVAILABLE=false
)

REM Step 7: Launch Streamlit app
echo.
echo 🎮 Step 7: Launching Streamlit demo...
echo.
echo 🌟 Setup Summary:
echo    • Virtual environment: ✅ Ready
echo    • Dependencies: ✅ Installed
if "%DATASET_AVAILABLE%"=="true" (
    echo    • Dataset: ✅ Available
) else (
    echo    • Dataset: ❌ Missing
)
if "%MODELS_AVAILABLE%"=="true" (
    echo    • Models: ✅ Trained
) else (
    echo    • Models: ❌ Not trained
)
echo.

if "%MODELS_AVAILABLE%"=="true" (
    echo 🚀 Launching Streamlit app with trained models...
) else (
    echo 🔧 Launching Streamlit app in demo mode...
    echo    ^(You can explore the interface, but predictions require trained models^)
)

echo.
echo 🌐 Opening browser at: http://localhost:8501
echo 📝 Press Ctrl+C to stop the app
echo.

REM Launch Streamlit
python -m streamlit run app\streamlit_app.py

echo.
echo 👋 Thanks for using Credit Card Fraud Detection!
echo 🐛 Report issues at: https://github.com/your-repo/issues
pause
