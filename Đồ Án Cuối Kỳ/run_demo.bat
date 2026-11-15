
@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   KHOI DONG HE THONG DU DOAN BENH TIEU DUONG (DEMO)
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Khong tim thay Python trong thu muc .venv
    echo Hay kiem tra lai moi truong ao cua do an.
    pause
    exit /b 1
)

echo Dang khoi dong Streamlit...
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

echo.
echo Ung dung da dung hoac gap loi.
pause

