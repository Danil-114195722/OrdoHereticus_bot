@echo off

call %~dp0Inquisitor\Scripts\activate

cd %~dp0

python bot.py

pause