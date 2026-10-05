@echo off
rem Launcher used by the "AlpacaBot" scheduled task. Runs the bot from the repo root with the
rem project venv; anything that escapes the logger (crash tracebacks) lands in bot_console.log.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m bot.main >> bot_console.log 2>&1
