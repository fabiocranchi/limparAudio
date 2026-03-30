---
date: 2026-03-30T12:23:45-03:00
researcher: Claude (Antigravity)
topic: "Análise do código Limpar Áudio"
tags: [research, codebase, limpar-audio, ffmpeg, demucs, python]
status: complete
---

# Research: Análise do código Limpar Áudio

## Research Question
"veja o que o código faz"

## Summary
O projeto **Limpar Áudio** é uma ferramenta de linha de comando (CLI) focada em extrair, limpar e aprimorar o áudio de vídeos. Ele roda localmente e utiliza uma combinação de IA (Demucs, Resemble Enhance) e processamento de sinal tradicional (Noisereduce, equalização, compressão) para garantir uma qualidade de som de nível "Broadcast". Ele é projetado especificamente para rodar no Windows utilizando apenas a CPU e lidando com armadilhas comuns (instalação de dependências pesadas, falta do FFmpeg, etc).

## Detailed Findings

### Pipeline de Processamento de Áudio (`limpar_audio.py`)
O coração do sistema reside no arquivo `limpar_audio.py`, que executa as seguintes etapas sequenciais:

1. **Extração:** Extrai a faixa de áudio do arquivo de vídeo usando `FFmpeg` (em formato WAV 16-bit PCM estéreo a 44.1kHz).
2. **Separação de Voz (IA):** Utiliza o modelo `htdemucs` da Meta para isolar a voz humana (vocals) dos outros elementos (bateria, baixo, outros).
3. **Limpeza Espectral:** Opcionalmente aplica a biblioteca `noisereduce` para remover qualquer ruído estático residual usando filtragem espectral.
4. **Resemble Enhance (IA):** Opcionalmente passa o áudio por um modelo de difusão de IA (`resemble-enhance`) para desruidar e aplicar "super resolução", adicionando corpo e eliminando artefatos para alcançar qualidade de estúdio de gravação. Há até um *monkey patch* interno no `limpar_audio.py` para rodar o pacote sem a engine `deepspeed` (que causava conflitos no Windows).
5. **Cadeia Broadcast:** Aplica uma sequência de processamento tradicional de rádio e TV:
   - *High-Pass Filter:* Remove frequências indesejadas abaixo de 80Hz.
   - *EQ Paramétrico:* Dá +2dB em 300Hz para corpo e +3dB em 3.5kHz para clareza/presença da voz.
   - *De-Esser:* Atenua sibilâncias excessivas dinamicamente na faixa de 5kHz a 8kHz.
   - *Compressor:* Achata a dinâmica (ratio 3:1) com threshold de -20dB, estabilizando subidas bruscas de volume.
   - *Normalização LUFS:* Garante o padrão EBU-R128 (-14 LUFS).
   - *Limiter:* Previne estalos ou cortes na saída de áudio.
6. **Remontagem (Merge):** Mixa o arquivo de vídeo original (cópia raw) com a nova trilha de áudio aprimorada (codificada em AAC 320kbps) via `FFmpeg`.

### Solução anti-crash para o Windows (`setup.bat` e `limpar_audio.bat`)
- O arquivo `setup.bat` cria automaticamente o ambiente virtual (venv), detecta a versão do Python, instala a versão CPU do PyTorch (para evitar o download de Gigabytes do CUDA inútil sem GPU), instala as dependências via *pip*, checa o estado do `ffmpeg` com um auto-fallback instalando-o via `winget` e até propõe um fallback embutido do Python 3.12 usando `pyenv-win` se problemas aparecerem no Python 3.13.
- O arquivo `limpar_audio.bat` funciona como um inicializador rápido (wrapper) ativando o VENV silenciosamente e passando os argumentos pra a CLI. Aceita sistema de arrastar e soltar (Drag and Drop) no Windows.

## Code References
- `limpar_audio.py:231-250` - `extrair_audio`: Extração do source via FFmpeg.
- `limpar_audio.py:253-312` - `separar_voz_demucs`: IA Demucs com carregamento na CPU e detecção de vozes.
- `limpar_audio.py:428-521` - `processar_broadcast`: Excelente demonstração de cadeia analógica adaptada para Python com NumPy/SciPy (Low-pass, Compressor, De-Esser, etc).
- `limpar_audio.py:60-90` - `_patch_deepspeed`: Monkey patching genial que cria um deepspeed fantasma na sessão do Python para driblar amarras de ambiente nativo C++.
- `setup.bat:153-181` - Detecção inteligente de FFmpeg e auto-reparo via `winget`.

## Architecture Insights
- **Forte Resiliência:** Foco extremo em usabilidade direta com `.bat`. Trata problemas como PyTorch, FFmpeg e limitações do Windows transparente ao usuário.
- **Eficiência e Qualidade:** Oferece opções no CLI para que o usuário escolha métodos mais rápidos (Noisereduce = 30s) ou mais lentos e de alta qualidade (Demucs e Resemble Enhance = minutos) adaptando-se às necessidades do hardware (rodando 100% na CPU) e do caso de uso.
- **Processamento de Áudio Sem Frameworks Gulosos:** A fase de processamento "Broadcast" dispensa softwares grandes como Audacity ou VSTs e faz toda cadeia via algoritmos criados sobre `scipy.signal` (filtros Butterworth em cascata, IIR Peaking, detecção de envlopes RMS), sendo autossuficiente.

## Open Questions
- Nenhum. O sistema é bem contido, com interface interativa (via CLI com a lib `rich`) bem documentada no README e nos scripts.
