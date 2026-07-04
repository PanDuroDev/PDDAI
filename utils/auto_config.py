import torch
import os
import platform
import time

def detect_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        return device, name, vram_gb
    return torch.device('cpu'), 'cpu', 0

def optimal_num_workers(is_cpu=False, is_memory_bound=True):
    """Choose optimal num_workers.

    Args:
        is_cpu: If True, compensate for slow CPU compute with more workers.
        is_memory_bound: If True (in-memory tensor), workers add IPC overhead.
                         If False (disk I/O), workers speed up loading.
    """
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4

    # On Windows, multiprocessing uses 'spawn' which has significant overhead.
    # For in-memory datasets (GPU-bound training), workers add IPC latency
    # with zero throughput benefit — GPU is the bottleneck, not data loading.
    if is_memory_bound or platform.system() == 'Windows':
        return 0
    if is_cpu:
        return max(1, cores - 1)
    return min(8, max(1, cores // 2))

def optimal_grad_accum(batch_size, is_cpu=False):
    target = 64 if is_cpu else 256
    steps = max(1, target // batch_size)
    return steps

def use_amp_on_device(device_name):
    return torch.cuda.is_available()

def is_windows():
    return platform.system() == 'Windows'

def tune_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

_dynamic_state = {
    'min_threads': 1,
    'max_threads': None,
    'last_adjust': 0.0,
    'interval': 30.0,
}


def set_process_priority():
    """Set process to below-normal priority so other apps stay responsive."""
    if platform.system() == 'Windows':
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL)
        except Exception:
            pass
    else:
        try:
            os.nice(5)
        except Exception:
            pass


def _get_cpu_load():
    """Return CPU load fraction (0.0–1.0). Returns 0 if unavailable."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.0) / 100.0
    except Exception:
        return 0.0


def _calc_thread_target(load, cores):
    """Map load → threads: high load = fewer threads, low load = more."""
    if load >= 0.8:
        return _dynamic_state['min_threads']
    if load >= 0.5:
        return max(_dynamic_state['min_threads'], cores // 6)
    return _dynamic_state['max_threads']


def set_num_threads(good_neighbor=True):
    """Set CPU threads. Uses most cores when training alone."""
    cores = os.cpu_count() or 4
    _dynamic_state['max_threads'] = max(10, cores - 2) if not good_neighbor else max(4, min(cores - 4, 12))
    _dynamic_state['min_threads'] = max(2, _dynamic_state['max_threads'] // 2)

    load = _get_cpu_load()
    target = _calc_thread_target(load, cores)
    torch.set_num_threads(target)


def dynamic_adjust_threads():
    """Periodically adjust threads based on system load. Call every N batches."""
    now = time.time()
    if now - _dynamic_state['last_adjust'] < _dynamic_state['interval']:
        return
    _dynamic_state['last_adjust'] = now

    cores = os.cpu_count() or 4
    load = _get_cpu_load()
    target = _calc_thread_target(load, cores)
    current = torch.get_num_threads()

    if target != current:
        torch.set_num_threads(target)


def optimal_batch_size(vram_gb, good_neighbor=True):
    if vram_gb >= 40:
        return 256
    elif vram_gb >= 24:
        return 128
    elif vram_gb >= 16:
        return 64
    elif vram_gb >= 12:
        return 32
    elif vram_gb >= 8:
        return 24
    elif vram_gb >= 6:
        return 16
    elif vram_gb >= 4:
        return 12
    return 8


def auto_config():
    device, device_name, vram_gb = detect_device()
    is_cpu = not torch.cuda.is_available()
    batch_size = optimal_batch_size(vram_gb)
    num_workers = optimal_num_workers(is_cpu=is_cpu, is_memory_bound=True)
    grad_accum = optimal_grad_accum(batch_size, is_cpu=is_cpu)
    amp = use_amp_on_device(device_name)

    set_process_priority()

    config = {
        'device': device,
        'device_name': device_name,
        'vram_gb': round(vram_gb, 1),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'grad_accum_steps': grad_accum,
        'use_amp': amp,
        'is_windows': is_windows(),
    }
    return config
