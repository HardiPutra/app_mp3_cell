@echo off
echo Sedang membuka Aplikasi Klasifikasi...
echo Mohon tunggu sebentar, browser akan terbuka otomatis.
cd /d %~dp0
streamlit run app.py 
pause