@echo off
rem Launcher used by the "AlpacaBotMonitor" scheduled task: read-only monitor page on 127.0.0.1:8765.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m bot.monitor >> monitor_console.log 2>&1
