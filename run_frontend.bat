@echo off
title Crime Detection NLP - Streamlit Web Application
echo ========================================================
echo   Launching Crime Detection NLP - Streamlit Frontend
echo   URL: http://localhost:8501
echo ========================================================
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run app\streamlit_app.py
pause

