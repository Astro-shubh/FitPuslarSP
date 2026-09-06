import numpy as np
import scipy.signal as signal

def get_bootstrap_mod_idx_error(data_vector, n_bootstraps=500):
    """Computes statistical uncertainty of the modulation index using bootstrap resampling."""
    if len(data_vector) == 0 or np.mean(data_vector) == 0:
        return 0.0

    boot_mod_indices = []
    n_samples = len(data_vector)
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(data_vector, size=n_samples, replace=True)
        mean_val = np.mean(boot_sample)
        if mean_val != 0:
            boot_mod_idx = np.std(boot_sample) / mean_val
            boot_mod_indices.append(boot_mod_idx)
    return np.std(boot_mod_indices)

def get_mod_idx_summed(stack, on1, on2, threshold):
    """Computes the integrated on-pulse intensity modulation index."""
    on_summ = []
    numpulses = 0
    for pulse in stack:
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        on_window = pulse[on1:on2]
        on_len = on2 - on1
        if np.sum(on_window) > threshold * np.std(off_window) * np.sqrt(on_len):
            on_summ.append(np.sum(on_window))
            numpulses += 1
    on_summ = np.array(on_summ)
    mod_idx = np.std(on_summ) / np.mean(on_summ) if len(on_summ) > 0 else 0
    error_midx = get_bootstrap_mod_idx_error(on_summ)
    return mod_idx, error_midx, numpulses

def get_mod_idx_resolved(stack, on1, on2, threshold):
    """Isolates bright single pulses and tracks their original chronological indices."""
    bright_on_windows = []
    bright_pulses_full = []
    bright_indices = []
    
    for idx, pulse in enumerate(stack):
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        on_window = pulse[on1:on2]
        on_len = on2 - on1
        if np.sum(on_window) > threshold * np.std(off_window) * np.sqrt(on_len):
            bright_on_windows.append(on_window)  
            bright_pulses_full.append(pulse)     
            bright_indices.append(idx)
            
    bright_on_windows = np.array(bright_on_windows)
    bright_pulses_full = np.array(bright_pulses_full)
    
    if len(bright_on_windows) == 0:
        return np.array([]), 0, 0, np.array([]), []
        
    down_factor = 1
    if (on2 - on1 > 200):
        down_factor = int((on2 - on1) / 100.0)

    bright_on_decimated = signal.decimate(bright_on_windows, down_factor, axis=1)
    global_noise_floor = np.std(bright_on_windows) * 0.01
    
    mod_idx = []
    error_midx = []
    for i in range(len(bright_on_decimated[0, :])):
        current_phase = bright_on_decimated[:, i]
        mean_val = np.mean(current_phase)
        if mean_val <= global_noise_floor:
            mod_idx.append(0.0)
            error_midx.append(0.0)
        else:
            mod_idx.append(np.std(current_phase) / mean_val)
            error_midx.append(get_bootstrap_mod_idx_error(current_phase))
        
    return np.array(mod_idx), np.array(error_midx), len(bright_on_decimated[0, :]), bright_pulses_full, bright_indices
