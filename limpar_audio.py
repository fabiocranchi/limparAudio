#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎙️ LIMPAR ÁUDIO v1.1
Limpeza de áudio de vídeos com IA (Demucs) e filtragem espectral (noisereduce).
Isola a voz e remove ruídos de fundo com qualidade broadcast.
Inclui aprimoramento de voz com Resemble Enhance (IA) e cadeia broadcast.
"""

import sys
import os
import signal
import shutil
import subprocess
import json
import tempfile
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, IntPrompt
from rich import box

# ============================================================
# Configuração global
# ============================================================

console = Console()
VERSION = "1.1"
SUPPORTED_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}
TEMP_FILES = []  # Rastrear arquivos temporários para limpeza

# Configurar logging
log_file = Path(__file__).parent / "limpar_audio.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger("limpar_audio")

# Silenciar loggers verbosos de bibliotecas externas
for noisy_logger in ["numba", "librosa", "torch", "torchaudio", "demucs", "resemble_enhance"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ============================================================
# Monkey-patch deepspeed para Windows (só necessário para Resemble Enhance)
# ============================================================

def _patch_deepspeed():
    """Cria stubs do deepspeed para permitir Resemble Enhance rodar sem ele."""
    import types
    ds = types.ModuleType('deepspeed')
    ds.DeepSpeedConfig = type('DeepSpeedConfig', (), {})
    ds.init_distributed = lambda *a, **k: None
    sys.modules['deepspeed'] = ds

    dsa = types.ModuleType('deepspeed.accelerator')
    _acc_cls = type('Acc', (), {
        'device_name': lambda s: 'cpu',
        'communication_backend_name': lambda s: 'nccl',
    })
    dsa.get_accelerator = lambda: _acc_cls()
    sys.modules['deepspeed.accelerator'] = dsa

    dsr = types.ModuleType('deepspeed.runtime')
    sys.modules['deepspeed.runtime'] = dsr

    dsre = types.ModuleType('deepspeed.runtime.engine')
    dsre.DeepSpeedEngine = type('DeepSpeedEngine', (object,), {})
    sys.modules['deepspeed.runtime.engine'] = dsre

    dsru = types.ModuleType('deepspeed.runtime.utils')
    dsru.clip_grad_norm_ = lambda *a, **k: None
    sys.modules['deepspeed.runtime.utils'] = dsru

try:
    import deepspeed
except ImportError:
    _patch_deepspeed()


# ============================================================
# Tratamento de sinais (Ctrl+C)
# ============================================================

def cleanup_handler(signum, frame):
    """Limpa arquivos temporários quando o usuário cancela (Ctrl+C)."""
    console.print("\n\n⚠️  Operação cancelada pelo usuário.", style="yellow")
    cleanup_temp_files()
    sys.exit(1)


def cleanup_temp_files():
    """Remove todos os arquivos temporários criados durante o processamento."""
    for f in TEMP_FILES:
        try:
            if os.path.isfile(f):
                os.remove(f)
                logger.info(f"Arquivo temporário removido: {f}")
            elif os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
                logger.info(f"Diretório temporário removido: {f}")
        except Exception as e:
            logger.warning(f"Falha ao remover temporário {f}: {e}")


signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)


# ============================================================
# Fase 2 — Engine Core
# ============================================================

def verificar_ffmpeg() -> bool:
    """Verifica se o FFmpeg está disponível no PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def obter_info_video(caminho: str) -> dict:
    """Obtém informações detalhadas do vídeo usando FFprobe."""
    logger.info(f"Obtendo informações do vídeo: {caminho}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        caminho
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe falhou: {result.stderr}")

        data = json.loads(result.stdout)
        info = {
            "arquivo": Path(caminho).name,
            "tamanho_bytes": int(data.get("format", {}).get("size", 0)),
            "duracao_seg": float(data.get("format", {}).get("duration", 0)),
        }

        # Extrair info dos streams
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and "video_codec" not in info:
                info["video_codec"] = stream.get("codec_name", "desconhecido")
                info["largura"] = stream.get("width", 0)
                info["altura"] = stream.get("height", 0)
                fps = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = map(int, fps.split("/"))
                    info["fps"] = round(num / den, 2) if den > 0 else 0
                except (ValueError, ZeroDivisionError):
                    info["fps"] = 0

            elif stream.get("codec_type") == "audio" and "audio_codec" not in info:
                info["audio_codec"] = stream.get("codec_name", "desconhecido")
                info["audio_sample_rate"] = int(stream.get("sample_rate", 0))
                info["audio_channels"] = stream.get("channels", 0)
                info["audio_bitrate"] = int(stream.get("bit_rate", 0))

        logger.info(f"Info do vídeo: {info}")
        return info

    except Exception as e:
        logger.error(f"Erro ao obter info do vídeo: {e}")
        raise


def formatar_duracao(segundos: float) -> str:
    """Formata segundos em MM:SS."""
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    return f"{minutos}:{segs:02d}"


def formatar_tamanho(bytes_val: int) -> str:
    """Formata bytes em unidade legível."""
    if bytes_val >= 1_073_741_824:
        return f"{bytes_val / 1_073_741_824:.1f} GB"
    elif bytes_val >= 1_048_576:
        return f"{bytes_val / 1_048_576:.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


def verificar_espaco_disco(caminho: str, multiplicador: int = 5) -> bool:
    """Verifica se há espaço suficiente em disco."""
    tamanho_arquivo = os.path.getsize(caminho)
    espaco_necessario = tamanho_arquivo * multiplicador

    disco = shutil.disk_usage(os.path.dirname(os.path.abspath(caminho)))

    if disco.free < espaco_necessario:
        console.print(
            f"\n⚠️  Espaço em disco pode ser insuficiente.\n"
            f"   Livre: {formatar_tamanho(disco.free)}\n"
            f"   Estimado necessário: {formatar_tamanho(espaco_necessario)}\n",
            style="yellow"
        )
        continuar = Prompt.ask(
            "   Continuar mesmo assim?",
            choices=["s", "n"],
            default="s"
        )
        return continuar == "s"
    return True


def extrair_audio(caminho_video: str, caminho_audio: str) -> None:
    """Extrai o áudio do vídeo como WAV 48kHz para máxima qualidade."""
    logger.info(f"Extraindo áudio: {caminho_video} -> {caminho_audio}")

    cmd = [
        "ffmpeg",
        "-i", caminho_video,
        "-vn",                    # Sem vídeo
        "-acodec", "pcm_s16le",   # PCM 16-bit (compatível com demucs)
        "-ar", "44100",           # 44.1kHz (requerido pelo Demucs)
        "-ac", "2",               # Estéreo
        "-y",                     # Sobrescrever se existir
        caminho_audio
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao extrair áudio: {result.stderr}")

    logger.info("Áudio extraído com sucesso.")


def separar_voz_demucs(caminho_audio: str, dir_saida: str) -> str:
    """Usa Demucs (htdemucs) para separar a voz dos outros sons."""
    logger.info(f"Iniciando separação de voz com Demucs: {caminho_audio}")

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Carregar modelo
    model = get_model("htdemucs")
    model.cpu()
    model.eval()

    # Carregar áudio com soundfile (mais confiável que torchaudio.load)
    audio_np, sample_rate = sf.read(caminho_audio, dtype='float32')

    # Converter para tensor PyTorch: (channels, samples)
    if len(audio_np.shape) == 1:
        # Mono -> estéreo
        waveform = torch.from_numpy(audio_np).unsqueeze(0).repeat(2, 1)
    else:
        # (samples, channels) -> (channels, samples)
        waveform = torch.from_numpy(audio_np.T)

    # Garantir que é estéreo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Resample se necessário (Demucs espera 44100Hz)
    if sample_rate != model.samplerate:
        logger.info(f"Reamostrando de {sample_rate}Hz para {model.samplerate}Hz")
        import torchaudio
        resampler = torchaudio.transforms.Resample(sample_rate, model.samplerate)
        waveform = resampler(waveform)

    # Aplicar modelo
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    with torch.no_grad():
        sources = apply_model(model, waveform[None], device="cpu", progress=True)

    # sources shape: (1, num_sources, channels, samples)
    sources = sources[0]

    # Desnormalizar
    sources = sources * ref.std() + ref.mean()

    # O índice do vocal depende do modelo. Para htdemucs: drums, bass, other, vocals
    source_names = model.sources
    vocal_idx = source_names.index("vocals")

    vocal_audio = sources[vocal_idx].cpu().numpy()

    # Salvar áudio vocal com soundfile: (channels, samples) -> (samples, channels)
    caminho_vocal = os.path.join(dir_saida, "vocals.wav")
    sf.write(caminho_vocal, vocal_audio.T, model.samplerate)

    logger.info(f"Voz separada salva em: {caminho_vocal}")
    return caminho_vocal


def reduzir_ruido(caminho_audio: str, caminho_saida: str, nivel: float = 0.7) -> None:
    """Aplica redução de ruído espectral com noisereduce."""
    logger.info(f"Aplicando redução de ruído (nível: {nivel}): {caminho_audio}")

    import noisereduce as nr

    audio, sr = sf.read(caminho_audio)

    # Se estéreo, processar cada canal separadamente
    if len(audio.shape) > 1 and audio.shape[1] == 2:
        canal_esq = nr.reduce_noise(
            y=audio[:, 0], sr=sr,
            prop_decrease=nivel,
            stationary=False,
            n_std_thresh_stationary=1.5,
        )
        canal_dir = nr.reduce_noise(
            y=audio[:, 1], sr=sr,
            prop_decrease=nivel,
            stationary=False,
            n_std_thresh_stationary=1.5,
        )
        audio_limpo = np.column_stack([canal_esq, canal_dir])
    else:
        audio_limpo = nr.reduce_noise(
            y=audio, sr=sr,
            prop_decrease=nivel,
            stationary=False,
            n_std_thresh_stationary=1.5,
        )

    sf.write(caminho_saida, audio_limpo, sr)
    logger.info("Redução de ruído concluída.")


def pos_processar(caminho_audio: str, caminho_saida: str) -> None:
    """Pós-processamento: suavização anti-metálico + normalização."""
    logger.info(f"Pós-processando áudio: {caminho_audio}")

    from scipy.signal import butter, sosfilt

    audio, sr = sf.read(caminho_audio)

    # 1. Filtro passa-baixa suave para remover artefatos de alta frequência
    # Corte em 15kHz — preserva toda a faixa vocal mas remove artefatos HF
    nyquist = sr / 2
    freq_corte = min(15000, nyquist * 0.95)  # Segurança para não exceder Nyquist
    sos = butter(4, freq_corte / nyquist, btype='low', output='sos')

    if len(audio.shape) > 1 and audio.shape[1] == 2:
        audio[:, 0] = sosfilt(sos, audio[:, 0])
        audio[:, 1] = sosfilt(sos, audio[:, 1])
    else:
        audio = sosfilt(sos, audio)

    # 2. Normalização de loudness (target: -14 LUFS para broadcast)
    # Aproximação simples via RMS normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        # Target RMS para aproximar -14 LUFS
        target_rms = 0.1  # ~= -20 dBFS, conservador para evitar clipping
        ganho = target_rms / rms
        audio = audio * ganho

    # 3. Limitar para evitar clipping
    audio = np.clip(audio, -0.99, 0.99)

    sf.write(caminho_saida, audio, sr)


def aprimorar_voz_resemble(caminho_audio: str, caminho_saida: str) -> None:
    """Usa Resemble Enhance para melhorar qualidade da voz (qualidade de estúdio)."""
    logger.info(f"Aprimorando voz com Resemble Enhance: {caminho_audio}")

    import torch
    from resemble_enhance.enhancer.inference import denoise, enhance

    # Carregar áudio
    audio_np, sr = sf.read(caminho_audio, dtype='float32')

    # Resemble Enhance espera mono
    is_stereo = len(audio_np.shape) > 1 and audio_np.shape[1] == 2
    if is_stereo:
        # Processar como mono (média dos canais), depois duplicar
        audio_mono = audio_np.mean(axis=1)
    else:
        audio_mono = audio_np

    # Converter para tensor
    dwav = torch.from_numpy(audio_mono).float()

    # Primeiro: denoise
    dwav_denoised, new_sr = denoise(dwav, sr, device="cpu", run_dir=None)

    # Depois: enhance (super-resolution + qualidade)
    # nfe=32 é bom equilíbrio qualidade/velocidade, tau=0.5 é moderado
    dwav_enhanced, new_sr = enhance(
        dwav_denoised, new_sr, device="cpu",
        nfe=32, solver="midpoint", lambd=0.1, tau=0.5,
        run_dir=None,
    )

    # Converter de volta para numpy
    audio_enhanced = dwav_enhanced.cpu().numpy()

    # Se era estéreo, duplicar o canal
    if is_stereo:
        audio_enhanced = np.column_stack([audio_enhanced, audio_enhanced])

    sf.write(caminho_saida, audio_enhanced, new_sr)
    logger.info(f"Voz aprimorada salva em: {caminho_saida} (sr={new_sr})")


def processar_broadcast(caminho_audio: str, caminho_saida: str) -> None:
    """Cadeia de processamento broadcast profissional: EQ + compressor + de-esser + normalização."""
    logger.info(f"Processando cadeia broadcast: {caminho_audio}")

    from scipy.signal import butter, sosfilt, iirpeak

    audio, sr = sf.read(caminho_audio, dtype='float32')
    is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2

    def process_channel(ch):
        nyquist = sr / 2

        # 1. HIGH-PASS FILTER (80Hz) - Remove rumble e grave indesejado
        hp_freq = 80 / nyquist
        if hp_freq < 1.0:
            sos_hp = butter(4, hp_freq, btype='high', output='sos')
            ch = sosfilt(sos_hp, ch)

        # 2. EQ PARAMÉTRICO VOCAL
        # 2a. Realce de corpo (200-400Hz) - leve boost de +2dB
        center_body = 300 / nyquist
        if center_body < 1.0:
            b_body, a_body = iirpeak(center_body, 2.0)
            from scipy.signal import lfilter
            ch = ch + 0.12 * lfilter(b_body, a_body, ch)  # ~+2dB

        # 2b. Realce de presença/clareza (2-5kHz) - boost de +3dB
        center_presence = 3500 / nyquist
        if center_presence < 1.0:
            b_pres, a_pres = iirpeak(center_presence, 1.5)
            ch = ch + 0.2 * lfilter(b_pres, a_pres, ch)  # ~+3dB

        # 3. DE-ESSER (atenua sibilância 5-8kHz)
        sib_low = min(5000 / nyquist, 0.99)
        sib_high = min(8000 / nyquist, 0.99)
        if sib_low < sib_high:
            sos_sib = butter(4, [sib_low, sib_high], btype='band', output='sos')
            sibilance = sosfilt(sos_sib, ch)
            # Detecção de envelope da sibilância
            envelope = np.abs(sibilance)
            # Suavizar envelope
            window = int(sr * 0.005)  # 5ms
            if window > 0:
                envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
            # Threshold adaptativo
            threshold = np.percentile(envelope, 85)
            # Atenuação onde sibilância é alta
            mask = np.where(envelope > threshold, 0.4, 1.0)  # Atenua 60%
            ch = ch - sibilance * (1.0 - mask)

        # 4. COMPRESSOR DE DINÂMICA
        # Compressão suave para nivelar volume
        threshold_db = -20  # dB
        ratio = 3.0  # 3:1 compression
        threshold_linear = 10 ** (threshold_db / 20)

        # Envelope follower simples
        abs_ch = np.abs(ch)
        attack_samples = int(sr * 0.01)   # 10ms attack
        release_samples = int(sr * 0.1)   # 100ms release

        gain = np.ones_like(ch)
        env = 0.0
        for i in range(len(ch)):
            if abs_ch[i] > env:
                env += (abs_ch[i] - env) / max(attack_samples, 1)
            else:
                env += (abs_ch[i] - env) / max(release_samples, 1)

            if env > threshold_linear:
                gain_reduction = threshold_linear * (env / threshold_linear) ** (1.0 / ratio) / max(env, 1e-10)
                gain[i] = gain_reduction

        ch = ch * gain

        # 5. NORMALIZAÇÃO EBU R128 (-14 LUFS)
        rms = np.sqrt(np.mean(ch ** 2))
        if rms > 0:
            target_rms = 10 ** (-14 / 20) * 0.5  # Aproximação -14 LUFS
            ch = ch * (target_rms / rms)

        # 6. LIMITER (previne clipping)
        ch = np.clip(ch, -0.95, 0.95)

        return ch

    if is_stereo:
        audio[:, 0] = process_channel(audio[:, 0])
        audio[:, 1] = process_channel(audio[:, 1])
    else:
        audio = process_channel(audio)

    sf.write(caminho_saida, audio, sr)
    logger.info("Cadeia broadcast concluída.")


def aplicar_eq(caminho_audio: str, caminho_saida: str, preset: str) -> None:
    """Aplica equalização de 7 níveis pré-determinados baseados em graves e agudos."""
    logger.info(f"Aplicando equalização (preset: {preset}): {caminho_audio}")
    
    from scipy.signal import iirpeak, lfilter
    audio, sr = sf.read(caminho_audio, dtype='float32')
    is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2
    nyquist = sr / 2
    
    # Frequências centrais para a voz
    freq_grave = 150 / nyquist
    freq_agudo = 4500 / nyquist
    
    if freq_grave >= 1.0 or freq_agudo >= 1.0:
        sf.write(caminho_saida, audio, sr)
        return

    ganho_grave = 0.0
    ganho_agudo = 0.0
    
    if preset == "1": # Extremo Grave
        ganho_grave = 0.6  
        ganho_agudo = -0.3 
    elif preset == "2": # Muito Grave
        ganho_grave = 0.4
        ganho_agudo = -0.15
    elif preset == "3": # Grave
        ganho_grave = 0.25
        ganho_agudo = 0.0
    elif preset == "4": # Neutro
        ganho_grave = 0.0
        ganho_agudo = 0.0
    elif preset == "5": # Agudo
        ganho_grave = 0.0
        ganho_agudo = 0.25
    elif preset == "6": # Muito Agudo
        ganho_grave = -0.15
        ganho_agudo = 0.4
    elif preset == "7": # Extremo Agudo
        ganho_grave = -0.3
        ganho_agudo = 0.6
        
    if ganho_grave == 0.0 and ganho_agudo == 0.0:
        sf.write(caminho_saida, audio, sr)
        return
        
    b_grave, a_grave = iirpeak(freq_grave, 0.7) 
    b_agudo, a_agudo = iirpeak(freq_agudo, 0.7)

    def process_channel(ch):
        if ganho_grave != 0.0:
            ch_g = lfilter(b_grave, a_grave, ch)
            ch = ch + ganho_grave * ch_g
        if ganho_agudo != 0.0:
            ch_a = lfilter(b_agudo, a_agudo, ch)
            ch = ch + ganho_agudo * ch_a
        ch = np.clip(ch, -0.99, 0.99)
        return ch
        
    if is_stereo:
        audio[:, 0] = process_channel(audio[:, 0])
        audio[:, 1] = process_channel(audio[:, 1])
    else:
        audio = process_channel(audio)
        
    sf.write(caminho_saida, audio, sr)
    logger.info("Equalização concluída.")


def detectar_artefatos(caminho_audio: str) -> tuple[bool, float]:
    """
    Detecta possíveis artefatos metálicos no áudio usando spectral flatness.
    Retorna (tem_artefatos: bool, score: float).
    """
    logger.info(f"Detectando artefatos: {caminho_audio}")

    import librosa

    audio, sr = librosa.load(caminho_audio, sr=None, mono=True)

    # Calcular spectral flatness
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    media_flatness = float(np.mean(flatness))

    # Calcular spectral contrast para detecção complementar
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    media_contrast = float(np.mean(contrast))

    logger.info(f"Spectral flatness média: {media_flatness:.4f}")
    logger.info(f"Spectral contrast média: {media_contrast:.4f}")

    # Threshold: flatness alta + contrast baixo = possível som metálico
    # Esses valores foram calibrados para detecção conservadora
    tem_artefatos = media_flatness > 0.15 and media_contrast < 10.0

    return tem_artefatos, media_flatness


def merge_video_audio(caminho_video: str, caminho_audio: str, caminho_saida: str) -> None:
    """Junta o vídeo original com o áudio limpo. Vídeo copy, áudio AAC 320kbps."""
    logger.info(f"Merge: {caminho_video} + {caminho_audio} -> {caminho_saida}")

    cmd = [
        "ffmpeg",
        "-i", caminho_video,       # Input: vídeo original
        "-i", caminho_audio,       # Input: áudio limpo
        "-c:v", "copy",            # Copiar vídeo sem recodificar
        "-c:a", "aac",             # Codificar áudio como AAC
        "-b:a", "320k",            # Bitrate alto para qualidade broadcast
        "-map", "0:v:0",           # Usar vídeo do primeiro input
        "-map", "1:a:0",           # Usar áudio do segundo input
        "-movflags", "+faststart", # Otimizar para streaming web
        "-y",                      # Sobrescrever se existir
        caminho_saida
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Falha no merge: {result.stderr}")

    logger.info("Merge concluído com sucesso.")


# ============================================================
# Fase 3 — CLI Interativa
# ============================================================

def exibir_banner():
    """Exibe o banner do programa."""
    console.print()
    console.print(Panel(
        f"[bold cyan]🎙️  LIMPAR ÁUDIO v{VERSION}[/bold cyan]\n"
        "[dim]Limpeza de áudio com IA[/dim]",
        box=box.DOUBLE,
        style="cyan",
        padding=(1, 4),
    ))
    console.print()


def exibir_info_video(info: dict):
    """Exibe as informações do vídeo em uma tabela formatada."""
    table = Table(
        title="📋 Informações do vídeo",
        box=box.ROUNDED,
        show_header=False,
        title_style="bold white",
        padding=(0, 2),
    )

    table.add_column("Campo", style="bold cyan", width=16)
    table.add_column("Valor", style="white")

    table.add_row("Arquivo", info.get("arquivo", "—"))
    table.add_row("Duração", formatar_duracao(info.get("duracao_seg", 0)))

    if "largura" in info and "altura" in info:
        table.add_row("Resolução", f"{info['largura']}x{info['altura']}")

    if "video_codec" in info:
        table.add_row("Codec Vídeo", info["video_codec"].upper())

    if "audio_codec" in info:
        bitrate_str = ""
        if info.get("audio_bitrate"):
            bitrate_str = f" {info['audio_bitrate'] // 1000}kbps"
        table.add_row("Codec Áudio", f"{info['audio_codec'].upper()}{bitrate_str}")

    table.add_row("Tamanho", formatar_tamanho(info.get("tamanho_bytes", 0)))

    console.print(table)
    console.print()


def menu_modelo() -> str:
    """Menu para escolher o modelo de limpeza."""
    console.print("[bold white]🔧 Escolha o modelo de limpeza:[/bold white]")
    console.print("  [cyan][1][/cyan] 🧠 Demucs (IA)        — Melhor qualidade, ~3 min de processamento")
    console.print("  [cyan][2][/cyan] 📊 Noisereduce        — Bom resultado, ~30 seg de processamento")
    console.print("  [cyan][3][/cyan] 🔗 Ambos em sequência  — Máxima limpeza, ~4 min de processamento")
    console.print()

    escolha = Prompt.ask(
        "  Sua escolha",
        choices=["1", "2", "3"],
        default="3"
    )

    modelos = {"1": "demucs", "2": "noisereduce", "3": "ambos"}
    return modelos[escolha]


def menu_nivel() -> tuple[str, float]:
    """Menu para escolher o nível de agressividade da limpeza."""
    console.print()
    console.print("[bold white]🎚️  Escolha o nível de limpeza:[/bold white]")
    console.print("  [green][1][/green] 🟢 Suave     — Preserva naturalidade máxima")
    console.print("  [yellow][2][/yellow] 🟡 Médio     — Balanço entre limpeza e naturalidade")
    console.print("  [red][3][/red] 🔴 Agressivo — Máxima remoção, risco de artefatos")
    console.print()

    escolha = Prompt.ask(
        "  Sua escolha",
        choices=["1", "2", "3"],
        default="2"
    )

    niveis = {
        "1": ("suave", 0.4),
        "2": ("medio", 0.7),
        "3": ("agressivo", 0.95),
    }
    return niveis[escolha]


def menu_aprimoramento() -> str:
    """Menu para escolher o aprimoramento de voz."""
    console.print()
    console.print("[bold white]🎙️  Aprimoramento de voz (opcional):[/bold white]")
    console.print("  [cyan][1][/cyan] 🔇 Sem aprimoramento   — Apenas limpeza de ruído")
    console.print("  [cyan][2][/cyan] 🎛️  Broadcast           — EQ + compressor + de-esser")
    console.print("  [cyan][3][/cyan] 🧠 Resemble Enhance    — IA para qualidade de estúdio, ~3 min")
    console.print("  [cyan][4][/cyan] ✨ Máximo              — Resemble + Broadcast [bold green](recomendado)[/bold green]")
    console.print()

    escolha = Prompt.ask(
        "  Sua escolha",
        choices=["1", "2", "3", "4"],
        default="4"
    )

    modos = {"1": "nenhum", "2": "broadcast", "3": "resemble", "4": "maximo"}
    return modos[escolha]


def menu_eq() -> str:
    """Menu para escolher a equalização."""
    console.print()
    console.print("[bold white]🎛️  Ajuste de Equalização (Graves e Agudos):[/bold white]")
    console.print("  [cyan][1][/cyan] Extremo Grave   (+Grave, -Agudo) [dim]Voz de rádio AM pesada[/dim]")
    console.print("  [cyan][2][/cyan] Muito Grave     (+Grave)")
    console.print("  [cyan][3][/cyan] Grave           (Leve reforço)")
    console.print("  [green][4][/green] Neutro          (Som original sem ajustes)")
    console.print("  [cyan][5][/cyan] Agudo           (Leve reforço de clareza)")
    console.print("  [cyan][6][/cyan] Muito Agudo     (+Agudo, -Grave)")
    console.print("  [cyan][7][/cyan] Extremo Agudo   (+Agudo, -Grave) [dim]Para microfones abafados[/dim]")
    console.print()

    escolha = Prompt.ask(
        "  Sua escolha",
        choices=["1", "2", "3", "4", "5", "6", "7"],
        default="4"
    )
    return escolha


def prompt_artefatos() -> str:
    """Pergunta ao usuário o que fazer quando artefatos são detectados."""
    console.print()
    console.print(Panel(
        "[bold yellow]⚠️  Possíveis artefatos metálicos detectados no áudio.[/bold yellow]\n"
        "[dim]Isso pode ocorrer com nível agressivo de limpeza.[/dim]",
        style="yellow",
        box=box.ROUNDED,
    ))
    console.print()
    console.print("  [cyan][1][/cyan] 🔄 Reprocessar com nível mais suave")
    console.print("  [cyan][2][/cyan] ✅ Manter este resultado mesmo assim")
    console.print("  [cyan][3][/cyan] ❌ Cancelar e manter o vídeo original")
    console.print()

    return Prompt.ask(
        "  Sua escolha",
        choices=["1", "2", "3"],
        default="1"
    )


# ============================================================
# Pipeline principal
# ============================================================

def processar_video(caminho_video: str, modelo: str, nome_nivel: str, nivel: float, aprimoramento: str = "nenhum", eq_preset: str = "4", pasta_saida: str = None) -> str | None:
    """
    Pipeline completo de processamento.
    Retorna o caminho do arquivo de saída ou None se cancelado.
    """
    caminho_video = os.path.abspath(caminho_video)
    pasta_video = os.path.dirname(caminho_video)
    nome_base = Path(caminho_video).stem
    extensao = Path(caminho_video).suffix
    
    if pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}_limpo{extensao}")
    else:
        caminho_saida = os.path.join(pasta_video, f"{nome_base}_limpo{extensao}")

    # Criar diretório temporário
    dir_temp = tempfile.mkdtemp(prefix="limpar_audio_")
    TEMP_FILES.append(dir_temp)

    caminho_wav = os.path.join(dir_temp, "audio_original.wav")
    caminho_vocal = None
    caminho_nr = os.path.join(dir_temp, "audio_noisereduced.wav")
    caminho_resemble = os.path.join(dir_temp, "audio_resemble.wav")
    caminho_broadcast = os.path.join(dir_temp, "audio_broadcast.wav")
    caminho_eq = os.path.join(dir_temp, "audio_eq.wav")
    caminho_final = os.path.join(dir_temp, "audio_final.wav")

    TEMP_FILES.extend([caminho_wav, caminho_nr, caminho_resemble, caminho_broadcast, caminho_eq, caminho_final])

    # Determinar etapas
    etapas = []
    etapas.append(("Extraindo áudio", "extrair"))

    if modelo in ("demucs", "ambos"):
        etapas.append(("Separando voz (Demucs IA)", "demucs"))

    if modelo in ("noisereduce", "ambos"):
        etapas.append(("Reduzindo ruído residual", "noisereduce"))

    etapas.append(("Suavizando áudio", "pos_processar"))

    if aprimoramento in ("resemble", "maximo"):
        etapas.append(("Aprimorando voz (Resemble IA)", "resemble"))

    if aprimoramento in ("broadcast", "maximo"):
        etapas.append(("Processando cadeia broadcast", "broadcast"))

    if eq_preset != "4":
        etapas.append(("Aplicando equalização tonal", "eq"))

    etapas.append(("Remontando vídeo", "merge"))

    total_etapas = len(etapas)

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Processando...", total=total_etapas)

        for i, (descricao, tipo) in enumerate(etapas):
            step_label = f"[{i+1}/{total_etapas}] {descricao}"
            progress.update(task, description=step_label)

            try:
                if tipo == "extrair":
                    extrair_audio(caminho_video, caminho_wav)
                    audio_atual = caminho_wav

                elif tipo == "demucs":
                    caminho_vocal = separar_voz_demucs(caminho_wav, dir_temp)
                    audio_atual = caminho_vocal
                    TEMP_FILES.append(caminho_vocal)

                elif tipo == "noisereduce":
                    entrada_nr = audio_atual if 'audio_atual' in dir() else caminho_wav
                    reduzir_ruido(entrada_nr, caminho_nr, nivel)
                    audio_atual = caminho_nr

                elif tipo == "pos_processar":
                    pos_processar(audio_atual, caminho_final)
                    audio_atual = caminho_final

                elif tipo == "resemble":
                    aprimorar_voz_resemble(audio_atual, caminho_resemble)
                    audio_atual = caminho_resemble

                elif tipo == "broadcast":
                    processar_broadcast(audio_atual, caminho_broadcast)
                    audio_atual = caminho_broadcast

                elif tipo == "eq":
                    aplicar_eq(audio_atual, caminho_eq, eq_preset)
                    audio_atual = caminho_eq

                elif tipo == "merge":
                    # Detectar artefatos antes do merge
                    tem_artefatos, score = detectar_artefatos(audio_atual)

                    if tem_artefatos and nome_nivel != "suave":
                        progress.stop()
                        acao = prompt_artefatos()

                        if acao == "1":
                            # Reprocessar com nível suave
                            console.print("\n🔄 Reprocessando com nível suave...\n")
                            cleanup_temp_files()
                            TEMP_FILES.clear()
                            return processar_video(caminho_video, modelo, "suave", 0.4, aprimoramento, eq_preset, pasta_saida=pasta_saida)

                        elif acao == "3":
                            console.print("\n❌ Processamento cancelado.", style="red")
                            return None

                        # acao == "2": continuar com o resultado atual
                        progress.start()

                    merge_video_audio(caminho_video, audio_atual, caminho_saida)

            except Exception as e:
                logger.error(f"Erro na etapa '{descricao}': {e}", exc_info=True)
                progress.stop()
                console.print(f"\n❌ Erro na etapa '{descricao}':", style="red bold")
                console.print(f"   {e}", style="red")
                console.print(f"\n   Detalhes no log: {log_file}", style="dim")
                return None

            progress.advance(task)

    return caminho_saida


# ============================================================
# Main
# ============================================================

def menu_escolher_pasta() -> str:
    """Menu para listar pastas locais ou receber um caminho absoluto."""
    console.print("\n[bold white]📂 Procurando pastas no diretório atual...[/bold white]")
    # Listar diretórios válidos na pasta atual (ignorando ocultas e venv)
    pastas = []
    base_dir = os.getcwd()
    try:
        for entry in os.scandir(base_dir):
            if entry.is_dir() and not entry.name.startswith("."):
                pastas.append(entry.path)
    except Exception as e:
        pass
    
    pastas.sort()
    
    if pastas:
        table = Table(box=box.MINIMAL, show_header=False)
        table.add_column("Num", style="cyan", justify="right")
        table.add_column("Pasta", style="white")
        for i, p in enumerate(pastas):
            table.add_row(f"[{i+1}]", Path(p).name)
        console.print(table)
        console.print("\nDigite o número da lista acima, ou digite um [yellow]caminho absoluto[/yellow] se a sua pasta estiver em outro lugar.")
    else:
        console.print("  Nenhuma pasta compatível visualizada. Digite o [yellow]caminho absoluto[/yellow] dos vídeos.")

    escolha = Prompt.ask("  Sua escolha (número ou caminho)")
    
    if escolha.isdigit() and 1 <= int(escolha) <= len(pastas):
        return pastas[int(escolha)-1]
    
    # Assumir que é um caminho escrito
    escolha_limpa = escolha.strip('"').strip("'")
    if os.path.isdir(escolha_limpa):
        return os.path.abspath(escolha_limpa)
    else:
        console.print(f"❌ Diretório não encontrado: [white]{escolha_limpa}[/white]\n", style="red")
        sys.exit(1)


def fluxo_lote():
    """Gerencia o processamento em lote para mutiplos arquivos."""
    pasta_alvo = menu_escolher_pasta()
    
    videos = []
    for f in os.listdir(pasta_alvo):
        caminho_completo = os.path.join(pasta_alvo, f)
        if os.path.isfile(caminho_completo) and Path(f).suffix.lower() in SUPPORTED_FORMATS:
            videos.append(caminho_completo)
            
    if not videos:
        console.print(f"❌ Nenhum vídeo suportado encontrado em: [white]{pasta_alvo}[/white]\n", style="red")
        sys.exit(1)
        
    console.print(f"\n[bold green]Encontrados {len(videos)} vídeos na pasta para processar![/bold green]")
    
    modelo = menu_modelo()
    nome_nivel, nivel = menu_nivel()
    aprimoramento = menu_aprimoramento()
    eq_preset = menu_eq()
    
    pasta_processados = os.path.join(pasta_alvo, "processados")
    
    aprimoramento_label = {
        "nenhum": "NENHUM", "broadcast": "BROADCAST",
        "resemble": "RESEMBLE IA", "maximo": "MÁXIMO"
    }
    
    console.print()
    console.print(
        f"[bold]⏳ Iniciando processamento em Lote ({len(videos)} arquivos)...[/bold]\n"
        f"   Destino: [cyan]{pasta_processados}[/cyan]\n"
        f"   Modelo: [cyan]{modelo.upper()}[/cyan] | "
        f"Nível: [cyan]{nome_nivel.upper()}[/cyan] | "
        f"Voz: [cyan]{aprimoramento_label.get(aprimoramento, aprimoramento.upper())}[/cyan] | "
        f"EQ: [cyan]{eq_preset}[/cyan]\n"
    )
    
    inicio_lote = time.time()
    sucessos = 0
    falhas = 0
    
    for i, cod_vid in enumerate(videos, 1):
        nome_arq = Path(cod_vid).name
        console.print(f"\n[bold blue]▶ Processando ({i}/{len(videos)}):[/bold blue] {nome_arq}")
        
        # Verificar disco por vídeo avulso (usando a constante multiplicadora leve do processo atual)
        if not verificar_espaco_disco(cod_vid):
            console.print(f"❌ Abortando processamento em lote devido a espaço insuficiente no HD.\n", style="red")
            break
            
        try:
            req_info = obter_info_video(cod_vid)
            exibir_info_video(req_info)
            res = processar_video(cod_vid, modelo, nome_nivel, nivel, aprimoramento, eq_preset, pasta_saida=pasta_processados)
            cleanup_temp_files()
            if res:
                sucessos += 1
            else:
                falhas += 1
        except Exception as e:
            logger.error(f"Falha ao processar {cod_vid} em lote: {e}")
            console.print(f"❌ Falha inesperada no vídeo {nome_arq}. Pulando...\n", style="red")
            cleanup_temp_files()
            falhas += 1
            continue
            
    tempo_lote = time.time() - inicio_lote
    console.print()
    console.print(Panel(
        f"[bold green]🏁 Lote Concluído![/bold green]\n\n"
        f"   ✅ Sucessos: [green]{sucessos}[/green]\n"
        f"   ❌ Falhas: [red]{falhas}[/red]\n"
        f"   ⏱️  Tempo total: [white]{formatar_duracao(tempo_lote)}[/white]\n"
        f"   📁 Salvos em: [white]{pasta_processados}[/white]",
        style="blue",
        box=box.DOUBLE,
        padding=(1, 2),
    ))


def main():
    """Ponto de entrada principal."""
    # Banner
    exibir_banner()

    # Validar argumento
    if len(sys.argv) < 2:
        console.print("❌ Nenhum arquivo especificado.\n", style="red")
        console.print("   Uso: [cyan]limpar_audio.bat seu_video.mp4[/cyan] ou [cyan]limpar_audio.bat -l[/cyan] para lote\n")
        sys.exit(1)

    caminho_input = sys.argv[1]

    if caminho_input in ("-l", "--lote"):
        # Fluxo de lote bloqueia aqui ate o modulo encerrar.
        if not verificar_ffmpeg():
            console.print("❌ FFmpeg não encontrado. Execute [cyan]setup.bat[/cyan] primeiro.\n", style="red")
            sys.exit(1)
        fluxo_lote()
        sys.exit(0)

    # Validar arquivo no modo unitário
    caminho_video = caminho_input
    if not os.path.isfile(caminho_video):
        console.print(f"❌ Arquivo não encontrado: [white]{caminho_video}[/white]\n", style="red")
        sys.exit(1)

    extensao = Path(caminho_video).suffix.lower()
    if extensao not in SUPPORTED_FORMATS:
        console.print(
            f"❌ Formato não suportado: [white]{extensao}[/white]\n"
            f"   Formatos aceitos: {', '.join(sorted(SUPPORTED_FORMATS))}\n",
            style="red"
        )
        sys.exit(1)

    # Verificar FFmpeg
    if not verificar_ffmpeg():
        console.print("❌ FFmpeg não encontrado. Execute [cyan]setup.bat[/cyan] primeiro.\n", style="red")
        sys.exit(1)

    # Verificar espaço em disco
    if not verificar_espaco_disco(caminho_video):
        console.print("❌ Operação cancelada pelo usuário.\n", style="red")
        sys.exit(1)

    # Obter info do vídeo
    try:
        info = obter_info_video(caminho_video)
    except Exception as e:
        console.print(f"❌ Falha ao ler informações do vídeo: {e}\n", style="red")
        sys.exit(1)

    # Exibir info
    exibir_info_video(info)

    # Menus interativos
    modelo = menu_modelo()
    nome_nivel, nivel = menu_nivel()
    aprimoramento = menu_aprimoramento()
    eq_preset = menu_eq()

    # Confirmar
    aprimoramento_label = {
        "nenhum": "NENHUM", "broadcast": "BROADCAST",
        "resemble": "RESEMBLE IA", "maximo": "MÁXIMO"
    }
    console.print()
    console.print(
        f"[bold]⏳ Iniciando processamento...[/bold]\n"
        f"   Modelo: [cyan]{modelo.upper()}[/cyan] | "
        f"Nível: [cyan]{nome_nivel.upper()}[/cyan] | "
        f"Voz: [cyan]{aprimoramento_label.get(aprimoramento, aprimoramento.upper())}[/cyan] | "
        f"EQ: [cyan]{eq_preset}[/cyan]\n"
    )

    # Processar
    inicio = time.time()
    caminho_saida = processar_video(caminho_video, modelo, nome_nivel, nivel, aprimoramento, eq_preset)
    tempo_total = time.time() - inicio

    # Limpeza de temporários
    cleanup_temp_files()

    # Resultado
    if caminho_saida and os.path.isfile(caminho_saida):
        tamanho_saida = os.path.getsize(caminho_saida)

        console.print()
        console.print(Panel(
            f"[bold green]✅ Concluído![/bold green]\n\n"
            f"   📁 Saída: [white]{caminho_saida}[/white]\n"
            f"   📊 Tamanho: [white]{formatar_tamanho(tamanho_saida)}[/white]\n"
            f"   ⏱️  Tempo total: [white]{formatar_duracao(tempo_total)}[/white]",
            style="green",
            box=box.DOUBLE,
            padding=(1, 2),
        ))
        console.print()

        logger.info(
            f"Processamento concluído com sucesso. "
            f"Saída: {caminho_saida}, Tempo: {tempo_total:.1f}s"
        )
    else:
        if caminho_saida is None:
            # Cancelado ou com erro — já tratado
            pass
        else:
            console.print("❌ Algo deu errado. Verifique o log para detalhes.\n", style="red")

        logger.warning("Processamento não gerou arquivo de saída.")


if __name__ == "__main__":
    main()
