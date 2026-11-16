@echo off
title Crime Detection NLP - Jupyter Notebook
echo ========================================================
echo   Launching Crime Detection NLP - Jupyter Notebook
echo   Opening: Crime_Detection_NLP.ipynb
echo ========================================================
cd /d "%~dp0"
.venv\Scripts\jupyter.exe notebook Crime_Detection_NLP.ipynb
pause

