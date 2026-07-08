import os
import sys
import glob
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from psrqpy import QueryATNF

# --- 1. CORE ANALYSIS FUNCTIONS ---

def get_bootstrap_mod_idx_error(data_vector, n_bootstraps=500):
    """
    Computes statistical uncertainty of the modulation index using bootstrap resampling.
    """
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
    """
    Computes the integrated on-pulse intensity modulation index.
    """
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
    """
    Isolates bright single pulses and tracks their original chronological indices 
    to preserve chronological continuity during visual reconstruction.
    """
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


# --- GENERATIVE MULTI-GAUSSIAN MODEL ---
def multi_gaussian_profile(x, *params):
    """
    Generates a composite model profile from unpacked parameters.
    """
    model = np.zeros(len(x), dtype=float)
    for i in range(0, len(params), 3):
        amp = params[i]
        mean = params[i+1]
        sigma = params[i+2]
        model += amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return model


# --- 2. OPTIMIZED BATCH PEAK FINDING ENGINE ---

def find_optimized_generative_peaks_batch(bright_pulses_full, bright_indices, on1, on2, total_bins, threshold_multiplier=3.0):
    """
    Optimizes component detections across individual bright single pulses.
    """
    if len(bright_pulses_full) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], []
        
    on_len = on2 - on1
    x_axis = np.arange(on_len)
    
    duty_cycle = on_len / total_bins
    if duty_cycle < 0.04:
        calculated_kernel = int(np.floor(total_bins / 500.0))
    else:
        calculated_kernel = int(np.floor(total_bins / 300.0))
    if calculated_kernel % 2 == 0:
        calculated_kernel += 1  
    kernel_size = max(5, calculated_kernel)

    min_scale = int(max(2.0, kernel_size / 3.0))
    max_scale = int(on_len * 0.30)
    
    trial_widths = []
    current_w = float(min_scale)
    while current_w <= max_scale:
        trial_widths.append(int(np.round(current_w)))
        current_w *= 1.5
        
    trial_widths = np.unique(trial_widths)
    num_stages = len(trial_widths)

    all_peak_phases = []
    peaks_per_pulse_count = []
    filtered_pulses_full = []
    approx_shapes = []  
    peaks_2d = []       
    successful_indices = []
    
    for b_idx, pulse in enumerate(bright_pulses_full):
        smoothed_pulse = signal.medfilt(pulse, kernel_size=kernel_size)
        
        off_window = np.concatenate([smoothed_pulse[:on1], smoothed_pulse[on2:]])
        baseline_noise = np.std(off_window)
        baseline_mean = np.mean(off_window)
        absolute_height_threshold = baseline_mean + (threshold_multiplier * baseline_noise)
        
        on_window_raw = smoothed_pulse[on1:on2] - baseline_mean  
        
        full_2d_cwt = []
        for w in trial_widths:
            points = min(10 * w, on_len)
            if points % 2 == 0: 
                points += 1
            x = np.arange(points) - (points - 1) / 2.0
            x_scaled_sq = (x / w) ** 2
            wavelet = (1.0 - x_scaled_sq) * np.exp(-0.5 * x_scaled_sq)
            
            full_raw_convolution = np.convolve(smoothed_pulse, wavelet, mode='same')
            layer = full_raw_convolution[on1:on2]
            full_2d_cwt.append(layer)
            
        full_2d_cwt = np.array(full_2d_cwt)
        
        best_loss = np.inf
        winning_peaks = []
        winning_model = np.zeros_like(on_window_raw, dtype=float)
        
        for i in range(num_stages):
            active_widths = trial_widths[:i+1]
            collapsed_shape = np.mean(full_2d_cwt[:i+1, :], axis=0)
            
            reconstructed_trend = (collapsed_shape - np.mean(collapsed_shape)) * \
                                  (np.std(smoothed_pulse[on1:on2]) / (np.std(collapsed_shape) + 1e-6)) + np.mean(smoothed_pulse[on1:on2])
            
            stage_raw_peaks, _ = signal.find_peaks(
                reconstructed_trend, 
                height=absolute_height_threshold,
                prominence=baseline_noise * 1.0  
            )
            
            stage_valid_peaks = [pk for pk in stage_raw_peaks if (on_window_raw[pk] + baseline_mean) >= absolute_height_threshold]
            k = len(stage_valid_peaks)
            
            if k == 0:
                continue
                
            initial_guesses = []
            param_bounds_lower = []
            param_bounds_upper = []
            stage_min_w = float(np.min(active_widths))
            stage_max_w = float(np.max(active_widths))
            allowed_phase_variation = int(on_len/20.0) 
            
            for pk in stage_valid_peaks:
                pk_height = on_window_raw[pk]
                init_sigma = stage_max_w * 0.4
                initial_guesses.extend([pk_height, float(pk), init_sigma])
                param_bounds_lower.extend([0.0, float(int(max(0, pk - allowed_phase_variation))), stage_min_w])
                param_bounds_upper.extend([pk_height, float(int(min(on_len, pk + allowed_phase_variation))), stage_max_w])
                
            try:
                popt, _ = curve_fit(
                    multi_gaussian_profile, x_axis, on_window_raw, 
                    p0=initial_guesses, bounds=(param_bounds_lower, param_bounds_upper),
                    maxfev=1500  
                )
                synthetic_model = multi_gaussian_profile(x_axis, *popt)
                rss = np.sum((on_window_raw - synthetic_model) ** 2)
                if rss <= 0: 
                    rss = 1e-6
                    
                loss_score = (2 * (3 * k)) + (on_len * np.log(rss / on_len))
                
                if loss_score < best_loss:
                    best_loss = loss_score
                    winning_peaks = [int(np.round(popt[j])) for j in range(1, len(popt), 3)]
                    winning_model = synthetic_model + baseline_mean
            except Exception:
                continue
                    
        if len(winning_peaks) <= 0:
            continue

        peaks_per_pulse_count.append(len(winning_peaks))
        filtered_pulses_full.append(pulse)
        approx_shapes.append(winning_model)
        peaks_2d.append(winning_peaks)
        successful_indices.append(bright_indices[b_idx])
        
        for peak in winning_peaks:
            all_peak_phases.append(peak)
            
    return (np.array(peaks_2d, dtype=object), 
            np.array(peaks_per_pulse_count), 
            np.array(filtered_pulses_full), 
            np.array(approx_shapes),
            all_peak_phases,
            successful_indices)


# --- 3. REFACTORED TUNED HISTOGRAM FITTING FUNCTION ---

def fit_profile_with_histogram_peaks(single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses):
    """
    Builds an adaptive histogram of peak detections from single-pulse traces,
    applies Gaussian smoothing to suppress statistical discretization noise,
    identifies prominent peaks on the distribution using noise-resilient criteria,
    and runs a least-squares multi-component Gaussian fit to reconstruct the composite reference profile.
    """
    hist_fitted_profile = np.zeros(on_len)
    histogram_peak_locations = []
    raw_counts = np.array([])
    smoothed_counts = np.array([])
    centers = np.array([])
    num_bins = 0
    raw_peak_phases = []

    if len(single_pulse_peak_bins) == 0:
        return hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases

    # Dynamic bin selection rule based on total detected peak events (minimum 50)
    n_points = len(single_pulse_peak_bins)
    num_bins = int(np.round(max(50, np.sqrt(n_points))))
    
    counts, edges = np.histogram(single_pulse_peak_bins, bins=np.linspace(0, on_len, num_bins + 1))
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Process raw count intensities directly 
    raw_counts = counts.astype(float)
    smoothening_scale = int(max(1.5, num_bins/30.0))
    
    # Apply gentle Gaussian smoothing to isolate components without underestimating broad structures
    smoothed_counts = gaussian_filter1d(raw_counts, sigma=smoothening_scale)
    
    # Dynamic intensity gate with a hard user-defined minimum safety floor of 4.0
    height = max(4.0, num_bright_pulses / 100.0)
    
    # Minimum separation distance constraint (3% of window or at least 2 bins)
    min_distance = max(2, int(num_bins * 0.03))
    
    # Prominence threshold with a matching safety floor of 4.0 to reject small plateau ripples
    prominence = max(4.0, height*0.5)
    
    # Run peak finder on smoothed counts
    hist_pks, _ = signal.find_peaks(
        smoothed_counts, 
        height=height, 
        distance=min_distance
    )
    num_hist_components = len(hist_pks)
    allowed_phase_variation = int(0.1 * on_len)
    
    if num_hist_components > 0:
        raw_peak_phases = centers[hist_pks]
        hist_init_guesses = []
        hist_bounds_lower = []
        hist_bounds_upper = []
        max_profile_amp = np.max(raw_folded_on_pulse)
        
        for p_idx in hist_pks:
            guess_phase = centers[p_idx]
            hist_init_guesses.extend([max_profile_amp * 0.5, guess_phase, on_len * 0.05])
            hist_bounds_lower.extend([0.0, max(0.0, guess_phase - allowed_phase_variation), 2.0])
            hist_bounds_upper.extend([max_profile_amp, min(on_len, guess_phase + allowed_phase_variation), on_len * 0.5])
            
        try:
            x_axis = np.arange(on_len)
            hist_popt, _ = curve_fit(
                multi_gaussian_profile, x_axis, raw_folded_on_pulse,
                p0=hist_init_guesses, bounds=(hist_bounds_lower, hist_bounds_upper), maxfev=2000
            )
            hist_fitted_profile = multi_gaussian_profile(x_axis, *hist_popt)
            
            for j in range(1, len(hist_popt), 3):
                histogram_peak_locations.append(hist_popt[j])
                
            print(f"Successfully generated histogram-driven fit with {num_hist_components} components.")
        except Exception as e:
            print(f"Warning: Histogram-seeded curve_fit dropped execution: {e}")
            
    return hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases


# --- 4. BATCH DIRECTORY CONTROLLER ---

def find_unique_pulsars(folder):
    stats_files = glob.glob(os.path.join(folder, "*_stats.txt"))
    pulsars = []
    for f in stats_files:
        pulsars.append(os.path.basename(f).replace("_stats.txt", ""))
    return sorted(pulsars)


def process_pulsar(folder, pulsar_name, threshold):
    """
    Ingests and processes a single pulsar folder's stack, running
    both single-pulse and integrated profile fitting pipelines.
    """
    pulse_stackI = os.path.join(folder, f"{pulsar_name}_pulsestackI.npy")
    pulse_stats = os.path.join(folder, f"{pulsar_name}_stats.txt")

    if not (os.path.exists(pulse_stackI) and os.path.exists(pulse_stats)):
        return None

    p0_val, p1_val = None, None
    p1_string = r", $\dot{{E}}$ = N/A"
    try:
        query = QueryATNF(params=['EDOT', 'P0', 'P1'], psrs=[pulsar_name])
        if len(query.table) > 0:
            edot_val = query.table['EDOT'][0]
            p0_val = query.table['P0'][0]
            p1_val = query.table['P1'][0]
            
            if edot_val is not None and not np.isnan(edot_val):
                exponent = int(np.floor(np.log10(abs(edot_val))))
                base = edot_val / (10 ** exponent)
                p1_string = rf", $\dot{{E}}$ = {base:.2f} $\times$ 10$^{{{exponent}}}$ erg/s"
    except Exception as e:
        print(f"  Warning: ATNF query failed for {pulsar_name}: {e}")

    with open(pulse_stats, 'r') as f:
        for line in f:
            s = [int(r) for r in line.split()]
    on1, on2 = int(s[0]), int(s[1])
    on_len = on2 - on1

    SP_I = np.load(pulse_stackI)
    total_pulses_count, total_bins = SP_I.shape
    
    modidxI, modidxI_error, _ = get_mod_idx_summed(SP_I, on1, on2, threshold)
    mod_idx_resolved, mod_idx_resolved_error, window_size, bright_pulses_full, bright_indices = get_mod_idx_resolved(SP_I, on1, on2, threshold)
    
    # Run structural optimization over single pulses (gathers row mapping indices)
    peaks, peaks_per_pulse, peak_filtered_pulses, model_pulses, single_pulse_peak_bins, successful_indices = find_optimized_generative_peaks_batch(
        bright_pulses_full, bright_indices, on1, on2, total_bins, threshold_multiplier=3.0
    )
    
    if len(peak_filtered_pulses) > 0 and peak_filtered_pulses.ndim == 1:
        peak_filtered_pulses = np.atleast_2d(peak_filtered_pulses)
        model_pulses = np.atleast_2d(model_pulses)
        peaks = np.array([peaks], dtype=object)

    raw_pulses = peak_filtered_pulses[:, on1:on2] if len(peak_filtered_pulses) > 0 else np.array([])
    folded_fitted_pulses = np.mean(model_pulses, axis=0) if len(model_pulses) > 0 else np.zeros(on_len)
    num_bright_pulses = raw_pulses.shape[0] if len(raw_pulses) > 0 else 0

    # --- HISTOGRAM-DRIVEN GAUSSIAN FOLDED PROFILE FIT METHOD ---
    x_axis = np.arange(on_len)
    phase_axis = (x_axis + on1) / total_bins

    folded_profile_full = np.mean(bright_pulses_full, axis=0) if len(bright_pulses_full) > 0 else np.mean(SP_I, axis=0)
    raw_folded_on_pulse = folded_profile_full[on1:on2]

    # Execute the updated, smoothed fitting function
    hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases = fit_profile_with_histogram_peaks(
        single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses
    )

    phase_histogram_peaks_count = len(raw_peak_phases)

    single_pulse_peaks_80th = np.percentile(peaks_per_pulse, 80) if len(peaks_per_pulse) > 0 else 0.0

    # Generate global index mapped coordinates for plotting
    global_peak_phases_list = [(p + on1)/total_bins for p in single_pulse_peak_bins]
    smoothed_display_folded = signal.medfilt(folded_profile_full, kernel_size=5)[on1:on2]
    global_profile = np.mean(SP_I[:, on1:on2], axis=0)

    # =========================================================================
    # --- DOCUMENT PATH A: DIAGNOSTICS STATISTICS GRID PANEL ---
    # =========================================================================
    fig_stats = plt.figure(figsize=(9.5, 15))
    fig_stats.suptitle(f"Single pulse statistics: {pulsar_name}{p1_string}", fontsize=15, fontweight='bold', y=0.97)
    
    ax1 = fig_stats.add_subplot(4, 1, 1)
    ax2 = fig_stats.add_subplot(4, 1, 2)
    ax3a = fig_stats.add_subplot(4, 2, 5) 
    ax3b = fig_stats.add_subplot(4, 2, 6) 
    ax4 = fig_stats.add_subplot(4, 1, 4)

    phase_range0 = np.linspace(on1 / total_bins, on2 / total_bins, on_len)

    ax1.plot(phase_range0, global_profile, color='black', label="Global Mean Profile", zorder=2)
    # Overlay raw candidate peak markers directly
    for r_phase in raw_peak_phases:
        global_phase_pos = (r_phase + on1) / total_bins
        ax1.axvline(x=global_phase_pos, color='tab:red', linestyle='--', alpha=0.85, zorder=1)
        
    ax1.set_title(f"Folded Profile (Histogram candidate components detected = {phase_histogram_peaks_count})")
    ax1.set_xlabel("Pulse Phase")
    ax1.set_ylabel("Average Intensity")
    ax1.grid(True, linestyle=':', alpha=0.6)

    mod_idx_resolved_error = np.clip(mod_idx_resolved_error, 0, 3.0)
    if window_size > 0:
        phase_range = np.linspace(on1 / total_bins, on2 / total_bins, window_size)
        ax2.errorbar(phase_range, mod_idx_resolved, yerr=mod_idx_resolved_error, ls='None', color='tab:blue', alpha=0.7)
        ax2.scatter(phase_range, mod_idx_resolved, color='tab:blue', s=15)
        ax2.axhline(y=modidxI, color='r', linestyle='--', label=f"On-pulse Summed Mod (I={modidxI:.2f})")
        ax2.set_title(f"Modulation index of bright pulses (sigma > {threshold})")
        ax2.set_xlabel("Pulse Phase")
        ax2.set_ylabel("Modulation Index")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=':', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No pulses survived thresholding", ha='center', va='center')

    if len(peak_filtered_pulses) > 0:
        ax3a.plot(phase_range0, global_profile, color='black', linewidth=1.5)
        ax3a.set_title(f"Peak Locations Distribution (Smooth Count Peaks = {phase_histogram_peaks_count})")
        ax3a.set_xlabel("Pulse Phase")
        ax3a.set_ylabel("Average Intensity")
        if len(global_peak_phases_list) > 0 and num_bins > 0:
            ax3a_twin = ax3a.twinx()
            ax3a_twin.hist(global_peak_phases_list, bins=np.linspace(on1/total_bins, on2/total_bins, num_bins + 1), color='tab:green', alpha=0.15, edgecolor='tab:green')
            
            # Map the continuous smoothed trends curve overlay matching phase axes
            centers_phase = (centers + on1) / total_bins
            ax3a_twin.plot(centers_phase, smoothed_counts, color='tab:green', linewidth=1.5, linestyle='-')
            ax3a_twin.set_ylabel("Total Peak Counts", color='tab:green')
            ax3a_twin.tick_params(axis='y', labelcolor='tab:green')
        ax3a.grid(True, linestyle=':', alpha=0.6)
        
        if len(peaks_per_pulse) > 0:
            percentile_90_cutoff = np.percentile(peaks_per_pulse, 90)
            filtered_multiplicity = [p for p in peaks_per_pulse if p <= percentile_90_cutoff]
            max_peaks = np.max(filtered_multiplicity) if len(filtered_multiplicity) > 0 else 1
            bin_edges = np.arange(0.5, max_peaks + 1.5, 1) 
            ax3b.hist(filtered_multiplicity, bins=bin_edges, color='tab:purple', alpha=0.6, edgecolor='purple', rwidth=0.8)
            ax3b.set_xticks(np.arange(1, max_peaks + 1, 1))
        ax3b.set_title("Number of peaks detected per individual pulse frame")
        ax3b.set_xlabel("Peaks Count")
        ax3b.set_ylabel("Pulse Count")
        ax3b.grid(True, linestyle=':', alpha=0.6)
    else:
        ax3a.text(0.5, 0.5, "No pulses matched criteria", ha='center', va='center')
        ax3b.text(0.5, 0.5, "No pulses matched criteria", ha='center', va='center')

    if len(peak_filtered_pulses) > 0:
        bright_on_windows = peak_filtered_pulses[:, on1:on2]
        ax4.fill_between(phase_range0, np.percentile(bright_on_windows, 5, axis=0), np.percentile(bright_on_windows, 95, axis=0), color='tab:orange', alpha=0.25)
        ax4.plot(phase_range0, np.percentile(bright_on_windows, 50, axis=0), color='black', linewidth=2.5)
        ax4.set_title("Intensity variations")
        ax4.set_xlabel("Pulse Phase")
        ax4.set_ylabel("Intensity")
        ax4.set_xlim(phase_range0[0], phase_range0[-1])
        ax4.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # =========================================================================
    # --- DOCUMENT PATH B: CHRONOLOGICAL DOUBLE PULSE STACKS ---
    # =========================================================================
    fig_stack = plt.figure(figsize=(13, 9.5))
    fig_stack.suptitle(f"Continuous Pulse Stack Topography: {pulsar_name}", fontsize=14, fontweight='bold', y=0.96)
    gs = fig_stack.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.15, wspace=0.18)

    prof_ax1 = fig_stack.add_subplot(gs[0, 0])
    prof_ax2 = fig_stack.add_subplot(gs[0, 1])
    stack_ax1 = fig_stack.add_subplot(gs[1, 0], sharex=prof_ax1)
    stack_ax2 = fig_stack.add_subplot(gs[1, 1], sharex=prof_ax2)

    # Reconstruct chronological double pulse stack layout
    continuous_raw_stack = SP_I[:, on1:on2]
    continuous_model_stack = np.zeros((total_pulses_count, on_len))
    if len(model_pulses) > 0:
        for b_idx, orig_row in enumerate(successful_indices):
            if b_idx < len(model_pulses):
                continuous_model_stack[orig_row, :] = model_pulses[b_idx]

    # Column 1 Top: Raw Profile Data Trace
    prof_ax1.plot(phase_axis, raw_folded_on_pulse, color='purple', linewidth=2)
    prof_ax1.set_title("Raw Integrated Profile Data", fontsize=11, fontweight='bold')
    prof_ax1.set_ylabel("Average Intensity")
    prof_ax1.grid(True, linestyle=':', alpha=0.4)
    prof_ax1.tick_params(labelbottom=False)

    # Column 1 Bottom: Continuous Raw Heatmap Stack
    stack_ax1.imshow(continuous_raw_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax1.set_title("Continuous Raw Data Pulse Stack", fontsize=11)
    stack_ax1.set_xlabel("Pulsar Rotation Phase ($\phi$)")
    stack_ax1.set_ylabel("Continuous Pulse Number")

    # Column 2 Top: Verification Summary Trace
    prof_ax2.plot(phase_axis, folded_fitted_pulses, color='purple', linewidth=2, linestyle=':', label='Averaged Single Fits')
    if np.any(hist_fitted_profile):
        prof_ax2.plot(phase_axis, hist_fitted_profile, color='tab:green', linewidth=2, linestyle='--', label='Histogram-Driven Fit')
    prof_ax2.set_title("Optimized Fitted Models Summary", fontsize=11, fontweight='bold')
    prof_ax2.grid(True, linestyle=':', alpha=0.4)
    prof_ax2.legend(loc='upper right', fontsize=9)
    prof_ax2.tick_params(labelbottom=False)

    # Column 2 Bottom: Continuous Zero-Filled Fitted Model Stack
    stack_ax2.imshow(continuous_model_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax2.set_title("Continuous Fitted Model Stack (Zero-Filled)", fontsize=11)
    stack_ax2.set_xlabel("Pulsar Rotation Phase ($\phi$)")

    # Return elements matching processing structure
    return fig_stats, fig_stack, p0_val, p1_val, phase_histogram_peaks_count, modidxI, single_pulse_peaks_80th


# --- 5. MAIN RUNNER CONTROLLER ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_pulsar_analysis.py <Threshold> <directory>")
        sys.exit(1)

    threshold_val = float(sys.argv[1])
    folder_path = sys.argv[2]
    
    # Dual output file paths
    output_pdf_stats = os.path.join(folder_path, "compiled_pulsar_plots.pdf")
    output_pdf_stacks = os.path.join(folder_path, "compiled_pulse_stacks.pdf")
    output_txt_summary = os.path.join(folder_path, "analysis_summary_table.txt")

    pulsar_list = find_unique_pulsars(folder_path)
    print(f"Found {len(pulsar_list)} pulsars to process...")

    with open(output_txt_summary, "w") as txt_out:
        txt_out.write("Pulsar_Name\tPeriod_P(s)\tP_dot(s/s)\tSummed_Mod_Index\tHistogram_Peaks_Count\tSingle_Pulse_Peaks_80th_Percentile\n")

    # Compile files simultaneously
    with PdfPages(output_pdf_stats) as pdf_stats, PdfPages(output_pdf_stacks) as pdf_stacks:
        for psr in pulsar_list:
            print(f"Processing {psr}...")
            try:
                result = process_pulsar(folder_path, psr, threshold_val)
                if result is not None:
                    fig_stats, fig_stack, p0, p1, hist_pks, mod_idx_sum, single_pulse_80th = result
                    
                    pdf_stats.savefig(fig_stats)
                    pdf_stacks.savefig(fig_stack)
                    
                    plt.close(fig_stats)
                    plt.close(fig_stack)
                    
                    # Output summary tables directly
                    with open(output_txt_summary, "a") as txt_out:
                        p0_str = f"{p0:.6f}" if p0 is not None else "N/A"
                        p1_str = f"{p1:.6e}" if p1 is not None else "N/A"
                        txt_out.write(f"{psr}\t{p0_str}\t{p1_str}\t{mod_idx_sum:.4f}\t{hist_pks}\t{single_pulse_80th:.2f}\n")
            except Exception as e:
                print(f"Error processing {psr}: {e}")

    print(f"\nSuccessfully compiled statistical analysis into: {output_pdf_stats}")
    print(f"Successfully compiled comparative pulse stack views into: {output_pdf_stacks}")
    print(f"Successfully generated clean summary database matrix file to: {output_txt_summary}")
