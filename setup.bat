@echo off
echo Setting up KittenTTS Studio...

if not exist venv (
    echo Creating virtual environment...
    py -3.12 -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete! Run runner.bat to start.
pause
