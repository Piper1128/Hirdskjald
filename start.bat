@echo off
title Hirdskjald
cd /d "%~dp0"
echo.
echo   Vaekker Hirdskjald - browseren aabner selv naar skjalden er klar (~15 sek).
echo   Luk skjalden igen med Ctrl+C, eller luk dette vindue.
echo.
python app.py
echo.
echo   Skjalden sover. Tryk en tast for at lukke vinduet.
pause >nul
