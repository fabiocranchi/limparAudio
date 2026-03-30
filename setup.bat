@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo 
echo            LIMPAR AUDIO - Setup            
echo          Instalacao do ambiente               
echo 
echo.

:: ============================================================
:: 1. Verificar Python
:: ============================================================
echo [1/4] Verificando Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python nao encontrado no PATH.
    echo    Tentando instalar Python 3.12 via winget...
    
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo.
        echo     Nao foi possivel instalar Python automaticamente.
        echo    Instale Python 3.10+ manualmente em https://www.python.org/downloads/
        echo    Lembre-se de marcar "Add Python to PATH" na instalacao.
        echo.
        pause
        exit /b 1
    ) else (
        echo.
        echo     Python 3.12 instalado com sucesso!
        echo      ATENCAO: E necessario aplicar a variavel PATH.
        echo      Por favor, feche este terminal, abra novamente e rode setup.bat.
        echo.
        pause
        exit /b 0
    )
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo     Python %PYTHON_VERSION% encontrado.

:: ============================================================
:: 2. Criar Virtual Environment
:: ============================================================
echo.
echo [2/4] Configurando ambiente virtual...

if exist ".venv\Scripts\activate.bat" (
    echo      Ambiente virtual ja existe. Reutilizando...
) else (
    echo    Criando .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo  Falha ao criar ambiente virtual.
        pause
        exit /b 1
    )
    echo     Ambiente virtual criado.
)

:: Ativar venv
call .venv\Scripts\activate.bat

:: ============================================================
:: 3. Instalar dependencias
:: ============================================================
echo.
echo [3/4] Instalando dependencias (pode levar alguns minutos)...
echo     Instalando PyTorch + Demucs + bibliotecas de audio...
echo.

pip install --upgrade pip >nul 2>&1

echo    Instalando PyTorch CPU-only (otimizado, ~200MB)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu 2>setup_error.log
if errorlevel 1 (
    echo      Falha ao instalar PyTorch. Verificando compatibilidade...
    goto :TORCH_ERROR
)

echo    Instalando demais dependencias...
pip install -r requirements.txt 2>setup_error.log
if errorlevel 1 goto INSTALL_ERROR

echo    Instalando IA Resemble Enhance ^(otimizado^)...
pip install resemble-enhance --no-deps 2>>setup_error.log
pip install huggingface_hub omegaconf 2>>setup_error.log
if errorlevel 1 goto INSTALL_ERROR
echo     Todas as dependencias instaladas com sucesso.
goto INSTALL_SUCCESS

:INSTALL_ERROR
echo.
echo   Falha na instalacao das dependencias.
echo.

REM Verificar se eh problema de compatibilidade com Python 3.13
findstr /i "torch" setup_error.log >nul 2>&1
if errorlevel 1 goto UNKNOWN_ERROR

echo    Possivel incompatibilidade do PyTorch com Python %PYTHON_VERSION%.
echo.
echo    Opcoes:
echo    [1] Tentar instalar Python 3.12 via pyenv-win ^(automatico^)
echo    [2] Sair e instalar Python 3.12 manualmente
echo.
set /p CHOICE="    Sua escolha [1/2]: "
        
if not "%CHOICE%"=="1" goto INSTALL_MANUAL

echo.
echo    Instalando pyenv-win...
pip install pyenv-win --target "%USERPROFILE%\.pyenv" 2>nul
            
REM Adicionar pyenv ao PATH da sessao
set "PATH=%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
            
echo    Instalando Python 3.12.8...
pyenv install 3.12.8
if errorlevel 1 (
    echo     Falha ao instalar Python 3.12.8 via pyenv.
    echo    Instale manualmente: https://www.python.org/downloads/release/python-3128/
    pause
    exit /b 1
)
            
echo    Recriando ambiente virtual com Python 3.12...
if exist ".venv" rmdir /s /q .venv
"%USERPROFILE%\.pyenv\pyenv-win\versions\3.12.8\python.exe" -m venv .venv
call .venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo     Falha persistente na instalacao.
    echo    Verifique o arquivo setup_error.log para detalhes.
    pause
    exit /b 1
)
echo     Todas as dependencias instaladas com sucesso no novo ambiente 3.12.
goto INSTALL_SUCCESS

:INSTALL_MANUAL
echo.
echo    Instale Python 3.12 manualmente e execute setup.bat novamente.
pause
exit /b 1

:UNKNOWN_ERROR
echo    Erro desconhecido. Verifique setup_error.log para detalhes.
type setup_error.log
pause
exit /b 1

:INSTALL_SUCCESS

:: Limpar log de erro se tudo deu certo
if exist setup_error.log del setup_error.log

:: ============================================================
:: 4. Verificar FFmpeg
:: ============================================================
echo.
echo [4/4] Verificando FFmpeg...

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo      FFmpeg nao encontrado no PATH.
    echo.
    echo    Tentando instalar via winget...
    
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo.
        echo     Nao foi possivel instalar FFmpeg automaticamente.
        echo.
        echo    Instale manualmente:
        echo    1. Acesse https://www.gyan.dev/ffmpeg/builds/
        echo    2. Baixe "ffmpeg-release-essentials.zip"
        echo    3. Extraia e adicione a pasta "bin" ao PATH do sistema
        echo    4. Reinicie o terminal e execute setup.bat novamente
        echo.
        pause
        exit /b 1
    ) else (
        echo     FFmpeg instalado via winget.
        echo      Pode ser necessario reiniciar o terminal para o FFmpeg ficar disponivel.
    )
) else (
    for /f "tokens=3 delims= " %%v in ('ffmpeg -version 2^>^&1 ^| findstr /i "ffmpeg version"') do set FFMPEG_VERSION=%%v
    echo     FFmpeg !FFMPEG_VERSION! encontrado.
)

:: ============================================================
:: Concluido
:: ============================================================
echo.
echo 
echo    Setup concluido com sucesso!            
echo                                              
echo   Para usar:                                 
echo   limpar_audio.bat seu_video.mp4             
echo 
echo.

pause
endlocal
