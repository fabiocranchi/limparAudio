@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM limpar_audio.bat - Entry point
REM Ativa o venv automaticamente e chama o script Python
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Verificar se o venv existe
if not exist "%PYTHON%" goto ERR_VENV

REM Verificar se FFmpeg esta disponivel
ffmpeg -version >nul 2>&1
if errorlevel 1 goto ERR_FFMPEG

REM Verificar se foi passado argumento
if "%~1"=="" goto ERR_ARG

REM Chamar o script Python com o venv
"%PYTHON%" "%SCRIPT_DIR%limpar_audio.py" %*
goto END

:ERR_VENV
echo.
echo [ERRO] Ambiente virtual nao encontrado.
echo        Execute setup.bat primeiro para instalar as dependencias.
echo.
echo        ^> setup.bat
echo.
pause
exit /b 1

:ERR_FFMPEG
echo.
echo [ERRO] FFmpeg nao encontrado no PATH.
echo        Execute setup.bat para instalar ou adicione FFmpeg ao PATH.
echo.
pause
exit /b 1

:ERR_ARG
echo.
echo [ERRO] Nenhum arquivo especificado.
echo.
echo        Uso Normal: limpar_audio.bat seu_video.mp4
echo        Uso em Lote: limpar_audio.bat -l (selecione multiplos arquivos)
echo.
pause
exit /b 1

:END
endlocal
