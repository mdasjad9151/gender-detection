@echo off
echo Starting Gender Detection App...

REM Start Backend in a new window
echo Starting Backend on port 8000...
start "Gender Detection Backend" cmd /k "call run_backend.bat"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

REM Start Frontend in a new window
echo Starting Frontend...
start "Gender Detection Frontend" cmd /k "call run_frontend.bat"

echo.
echo Both services have been launched!
echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:5173
echo.
