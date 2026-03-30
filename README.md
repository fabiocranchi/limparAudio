# 🎙️ Limpar Áudio v1.1

Ferramenta CLI para limpeza e *enhancement* de áudio de vídeos usando Inteligência Artificial.  
Remove ruídos de fundo, isola a voz principal, restaura frequências perdidas via Super-Resolução Neural e processa o áudio em cadeia de efeitos com qualidade **Broadcast (Estúdio)**.

Tudo é **processado localmente** na sua máquina sem necessidade de API de nuvens ou Placa de Vídeo (Roda 100% na CPU do Windows!).

## ⚡ Instalação Rápida

### 1. Execute o setup (uma única vez)

Dê um clique duplo ou rode no seu terminal (recomendado usar o PowerShell ou CMD):
```bat
setup.bat
```

Isso irá:
- Verificar e instalar **Python 3.12** automaticamente caso não seja encontrado.
- Criar um ambiente virtual Python (`.venv`) autônomo.
- Instalar todas as dependências complexas (Demucs, PyTorch, noisereduce).
- Baixar o módulo modificado **Resemble Enhance AI** que ignora compilações nativas de Linux no Windows.
- Verificar e instalar **FFmpeg** se necessário.

### 2. Pronto! Use assim:

Basta arrastar o arquivo de vídeo original sobre o arquivo **`limpar_audio.bat`**, ou execute no terminal com o caminho do vídeo:
```bat
limpar_audio.bat seu_video.mp4
```

## 🎯 Como o Pipeline Funciona

A ferramenta usa uma arquitetura sequencial de *Audio Processing*:

```
seu_video.mp4
     │
     ▼
┌─────────────┐
│   FFmpeg    │──► Extrai áudio (WAV Float32)
└─────────────┘
     │
     ▼
┌─────────────┐
│   Demucs    │──► IA da Meta separa voz dos outros sons
│  (htdemucs) │    (ruído, música, ambiência, cliques)
└─────────────┘
     │
     ▼
┌─────────────┐
│ Noisereduce │──► Filtragem espectral clássica remove ruído residual cego
└─────────────┘
     │
     ▼
┌─────────────┐
│ Resemble IA │──► Enhancement Neural (Denoising + Super Resolução 44.1kHz)
└─────────────┘
     │
     ▼
┌─────────────┐
│  Broadcast  │──► EQ Paramétrico + Compressor + De-esser + Limiter (-14 LUFS)
└─────────────┘
     │
     ▼
┌─────────────┐
│   FFmpeg    │──► Junta vídeo original + O NOVO Áudio (AAC de alta taxa: >220kbps)
└─────────────┘
     │
     ▼
  seu_video_limpo.mp4
```

## 🔧 Opções Disponíveis

### Modelos de Limpeza (Passo 1)
| Modelo | Descrição | Velocidade |
|---|---|---|
| **Demucs (IA)** | Separação de voz com deep learning da Meta | ~3 min/vídeo |
| **Noisereduce** | Filtragem espectral clássica | ~30 seg/vídeo |
| **Ambos** | Demucs + Noisereduce em sequência | ~4 min/vídeo |

### Níveis de Agressividade (Passo 2)
| Nível | Resultado |
|---|---|
| 🟢 **Suave** | Preserva máxima naturalidade da voz |
| 🟡 **Médio** | Equilíbrio entre limpeza e naturalidade |
| 🔴 **Agressivo** | Máxima remoção de ruído |

### Aprimoramento de Voz (Novo! - Passo 3)
| Opção | Benefício | Custo de Tempo |
|---|---|---|
| 🔇 **Nenhum** | Apenas passa pelo filtro de ruído | Instantâneo |
| 🎛️ **Broadcast** | Tratamento clássico (EQ 300Hz e 3kHz, Compressor de dinâmica) | Instantâneo |
| 🧠 **Resemble Enhance** | IA avança para recuperar brilho da voz e resolução de estúdio | ~4 a 10 min |
| 🌟 **Máximo** | Resemble + Broadcast na sequência | ~4 a 10 min |

## 📋 Requisitos do Sistema

- **Windows 10 ou 11**
- **Python 3.10+** (Instalado automaticamente via Winget se você não o possui; compatível também com fallback para 3.12 usando Pyenv via Setup)
- **FFmpeg** (Instalado automaticamente via Winget se você não o possui)
- **CPU Módica e ~4 GB espaço no disco** (Para download automático do Modelo de IA de Resemble e Demucs durante o primeiro uso da função).

## 📁 Arquitetura Anti-Crash para Windows

Esse projeto foi especialmente desenvolvido para **não craxar em prompts do Windows CMD ou Powershell**.
O `setup.bat` e `limpar_audio.bat` utilizam estrutura baseada em `GOTO` no lugar dos vulneráveis laços `IF/ELSE` do Windows, assegurando que o parsing nunca intercepte lixo Unicode/UTF-8 em computadores com codepages não padrão!

## ❓ Solução de Problemas

### Demora Demais a Limpar?
A etapa "**Resemble Enhance**" é um modelo Diffusion Basead pesado. Para a maioria dos processadores modernos da Intel/AMD rodar 1 minuto de vídeo leva de 5 a 10 minutos para concluir a inferência. Pule essa etapa caso queira rapidez!

### "Erro de compatibilidade com PyTorch"
Se alguma complicação sobre `torch` ocorrer, é incompatibilidade isolada do Python 3.13 via C++. Mas não se preocupe: o `setup.bat` capturará o erro no log, perguntará se você quer baixar automaticamente o Python 3.12 via `pyenv` silenciosamente, e fará a ponte sozinho!

### Logs
Caso o processamento falhe e aborte a operação subitamente, veja o arquivo invisível `limpar_audio.log` ou `setup_error.log` para os logs limpos registrados pelo sistema.
