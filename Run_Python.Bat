@echo off
setlocal EnableDelayedExpansion

:: Initialization
set "ScriptDirectoryAoclTorchLocal=%~dp0"
set "ScriptDirectoryMantella=%ScriptDirectoryAoclTorchLocal:~0,-1%"
pushd "%ScriptDirectoryAoclTorchLocal%"
set PYTHON=C:\Program Files\Python310\Python.exe

:: Display Is Reset
cls
echo ========================================================================================================================
echo                                                      AoclTorchLocal
echo ------------------------------------------------------------------------------------------------------------------------
echo.
timeout /t 1 >nul

:: Running AoclTorchLocal
echo Running AoclTorchLocal...
timeout /t 1 >nul
python .\main_script.py
if errorlevel 1 (
    echo Error occurred while running AoclTorchLocal.
    goto end_of_file
)
echo AoclTorchLocal Exited.
timeout /t 2 >nul

:end_of_file
:: Exit Program
echo Closing in 5 seconds...
timeout /t 5 >nul
exit /b

rem remove later...
pause