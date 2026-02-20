@echo off
echo Starting KittenTTS Studio...

call venv\Scripts\activate.bat

:: Find available port starting from 5000
set PORT=5000
:check_port
netstat -an | find ":%PORT%" >nul
if %ERRORLEVEL% equ 0 (
    set /a PORT+=1
    goto check_port
)

echo Found available port: %PORT%

:: Start backend in background
start "KittenTTS Backend" cmd /c "venv\Scripts\python.exe backend.py --port %PORT%"

:: Wait for server to start
timeout /t 3 /nobreak >nul

:: Open browser
start http://localhost:%PORT%

echo KittenTTS Studio is running at http://localhost:%PORT%
echo Press any key to stop...
pause

:: Cleanup
taskkill /FI "WINDOWTITLE eq KittenTTS Backend*" /F >nul 2>&1
