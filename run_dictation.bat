@echo off
REM Change to the script directory
D:
cd "D:\WINSPER"

REM Activate the virtual environment
echo Activating virtual environment...
call .\winsper_env\Scripts\activate.bat

REM Check if activation was successful (optional, basic check)
if "%VIRTUAL_ENV%"=="" (
    echo Failed to activate virtual environment.
    pause
    exit /b
)

echo Starting dictation script (realtime_dictation.py)...
echo Press Ctrl+C in this window to stop the script.

REM Run the Python script
python realtime_dictation.py

echo Script finished or was interrupted.

REM Deactivate the virtual environment
REM echo Deactivating virtual environment...
REM call .\winsper_env\Scripts\deactivate.bat

REM Pause to see output before closing
pause 