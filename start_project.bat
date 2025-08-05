@echo off
cd /d E:\project\FaceAttendanceSystem

start cmd /k "python app.py"
timeout /t 2 >nul

