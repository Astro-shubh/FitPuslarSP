import os
import sys
import glob
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# --- 1. CORE ANALYSIS FUNCTIONS ---

def get_bootstrap_mod_idx_error(data_vector, n_bootstraps=500):
    """
    Computes the statistical uncertainty of the modulation index using bootstrap resampling.
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

def get_mod_idx_resolved(stack, on1, on2, threshold):
    """
    Extracts high signal-to-noise pulse frames and computes the phase-resolved
    modulation index across the on-pulse region. Tracks original chronological indices.
    """
    bright_on_windows = []
    bright_pulses_full = []
    bright_indices = []  # Track original chronological indices
    
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
    
    mod_idx = []
    error_midx = []
    for i in range(len(bright_on_decimated[0, :])):
        current_phase = bright_on_decimated[:, i]
        mod_idx.append(np.std(current_phase) / np.mean(current_phase))
        error_midx.append(get_bootstrap_mod_idx_error(current_phase))
        
    return np.array(mod_idx), np.array(error_midx), len(bright_on_decimated[0, :]), bright_pulses_full, bright_indices


# --- DYNAMIC GENERATIVE GAUSSIAN MODEL ---
def multi_gaussian_profile(x, *params):
    """Generates a composite profile from an arbitrary number of unpacked component parameters."""
    model = np.zeros(len(x), dtype=float)
    for i in range(0, len(params), 3):
        amp = params[i]
        mean = params[i+1]
        sigma = params[i+2]
        model += amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return model


def find_optimized_generative_peaks(bright_pulses_full, bright_indices, on1, on2, total_bins, kernel_size, threshold_multiplier=3.0):
    """
    Performs automated peak tracking on individual single pulses using a multi-scale 
    Continuous Wavelet Transform (CWT) followed by Akaike Information Criterion (AIC) 
    regularization. Tracks successfully fitted chronological indices.
    """
    if len(bright_pulses_full) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], []
        
    on_len = on2 - on1
    x_axis = np.arange(on_len)
    
    min_scale = int(max(2.0, kernel_size/3.0))
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
    successful_indices = []  # Track indices of pulses that were successfully fitted
    
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
            
            for pk in stage_valid_peaks:
                pk_height = on_window_raw[pk]
                init_sigma = stage_max_w * 0.4
                allowed_phase_variation = int(on_len/20.0)
                
                initial_guesses.extend([pk_height, float(pk), init_sigma])
                param_bounds_lower.extend([0.0, float(int(max(0, pk - allowed_phase_variation))), stage_min_w])
                param_bounds_upper.extend([1.0 * pk_height, float(int(min(on_len, pk + allowed_phase_variation))), stage_max_w * 1.0])
                
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


# --- 2. REFACTORED PEAK HISTOGRAM AND PROFILE FITTER ---

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
    smoothening_scale = int(max(1, num_bins/30.0))
    
    # NEW: Apply gentle Gaussian smoothing to isolate components without underestimating broad structures
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
        distance=min_distance, 
 #       prominence=prominence
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


# --- 3. MAIN RUNNER EXECUTION ---

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <folder> <psrname> <threshold>")
        sys.exit(1)

    folder = sys.argv[1]
    psr_name = sys.argv[2]
    threshold = float(sys.argv[3])

    totalI_stack = folder + '/' + psr_name + '_pulsestackI.npy'
    on_off = folder + '/' + psr_name + '_stats.txt'

    stack = np.load(totalI_stack)
    total_pulses_count, total_bins = stack.shape
    stats_data = np.loadtxt(on_off, dtype=int)
    on1, on2 = int(stats_data[0]), int(stats_data[1])
    on_len = on2 - on1

    duty_cycle = (on2 - on1)/total_bins
    print(f"On-pulse boundaries: {on1} to {on2}")

    if(duty_cycle < 0.04):
        calculated_kernel = int(np.floor(total_bins / 500.0))
    else:
        calculated_kernel = int(np.floor(total_bins / 300.0))
    if calculated_kernel % 2 == 0:
        calculated_kernel += 1  
    kernel_size = max(5, calculated_kernel)
    print(f"Running Median Kernel Size: {kernel_size} bins")

    idx, error, length, bright_pulses, bright_indices = get_mod_idx_resolved(stack, on1, on2, threshold)

    # Single-pulse analysis running through the median-smoothed engine
    peaks, npeaks, raw_pulses_full, model_pulses, single_pulse_peak_bins, successful_indices = find_optimized_generative_peaks(
        bright_pulses, bright_indices, on1, on2, total_bins, kernel_size, threshold_multiplier=3.0
    )

    if raw_pulses_full.ndim == 1:
        raw_pulses_full = np.atleast_2d(raw_pulses_full)
        model_pulses = np.atleast_2d(model_pulses)
        peaks = np.array([peaks], dtype=object)

    raw_pulses = raw_pulses_full[:, on1:on2]
    folded_fitted_pulses = np.mean(model_pulses, axis=0) if len(model_pulses) > 0 else np.zeros(on_len)
    folded_profile_full = np.mean(bright_pulses, axis=0) if len(bright_pulses) > 0 else np.mean(stack, axis=0)
    raw_folded_on_pulse = folded_profile_full[on1:on2]
    num_bright_pulses = raw_pulses.shape[0]

    # --- DUAL MATRIX RECONSTRUCTION (CHRONOLOGICAL RESTORATION) ---
    continuous_raw_stack = stack[:, on1:on2]
    
    continuous_model_stack = np.zeros((total_pulses_count, on_len))
    if len(model_pulses) > 0:
        for b_idx, orig_row in enumerate(successful_indices):
            if b_idx < len(model_pulses):
                continuous_model_stack[orig_row, :] = model_pulses[b_idx]

    # --- HISTOGRAM-DRIVEN GAUSSIAN FOLDED PROFILE FIT METHOD ---
    print("Executing Histogram-driven profile fitting engine...")
    x_axis = np.arange(on_len)
    phase_axis = (x_axis + on1) / total_bins

    # Call extracted histogram fitting optimization routine (now with Gaussian filter inside)
    hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases = fit_profile_with_histogram_peaks(
        single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses
    )

    # Apply filter to diagnostic display raw overlays for baseline plot synchronization
    smoothed_display_folded = signal.medfilt(folded_profile_full, kernel_size=kernel_size)[on1:on2]

    print("Generating side-by-side comparative pulse stack file panel...")
    fig_stacks = plt.figure(figsize=(14, 10))
    gs = fig_stacks.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.15, wspace=0.18)

    prof_ax1 = fig_stacks.add_subplot(gs[0, 0])
    prof_ax2 = fig_stacks.add_subplot(gs[0, 1])
    stack_ax1 = fig_stacks.add_subplot(gs[1, 0], sharex=prof_ax1)
    stack_ax2 = fig_stacks.add_subplot(gs[1, 1], sharex=prof_ax2)

    # Column 1 Top: Raw Folded Profile
    prof_ax1.plot(phase_axis, raw_folded_on_pulse, color='purple', linewidth=2)
    prof_ax1.set_title(f"Raw Integrated Profile ({psr_name})", fontsize=11, fontweight='bold')
    prof_ax1.set_ylabel("Average Intensity")
    prof_ax1.grid(True, linestyle=':', alpha=0.4)
    prof_ax1.tick_params(labelbottom=False)

    # Column 1 Bottom: UNBROKEN CONTINUOUS RAW STACK (Includes faint pulses)
    stack_ax1.imshow(continuous_raw_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax1.set_title("Continuous Raw Data Pulse Stack", fontsize=11)
    stack_ax1.set_xlabel("Pulsar Rotation Phase ($\phi$)")
    stack_ax1.set_ylabel("Continuous Pulse Number")

    # Column 2 Top: MODEL PROFILES INTEGRATION PANEL (Displays both fitting models together)
    prof_ax2.plot(phase_axis, folded_fitted_pulses, color='purple', linewidth=2, linestyle=':', label='Averaged Single Fits')
    if np.any(hist_fitted_profile):
        prof_ax2.plot(phase_axis, hist_fitted_profile, color='tab:green', linewidth=2, linestyle='--', label='Histogram-Driven Fit')
    prof_ax2.set_title("Optimized Fitted Models Summary", fontsize=11, fontweight='bold')
    prof_ax2.grid(True, linestyle=':', alpha=0.4)
    prof_ax2.legend(loc='upper right', fontsize=9)
    prof_ax2.tick_params(labelbottom=False)

    # Column 2 Bottom: UNBROKEN CONTINUOUS MODEL STACK (Faint pulses filled with zeros)
    stack_ax2.imshow(continuous_model_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax2.set_title("Continuous Fitted Model Stack (Zero-Filled)", fontsize=11)
    stack_ax2.set_xlabel("Pulsar Rotation Phase ($\phi$)")

    output_stack_png = f"{folder}/{psr_name}_pulse_stacks.png"
    plt.savefig(output_stack_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean diagnostic comparison matrix saved successfully to: {output_stack_png}")


    # --- 4. INDIVIDUAL DISPLAY LOOP ---
    for i in range(len(peaks)):
        smoothed_display_single = signal.medfilt(raw_pulses_full[i], kernel_size=kernel_size)[on1:on2]
        off_window = np.concatenate((raw_pulses_full[i, 0:on1], raw_pulses_full[i, on2:-1]))
        rms = np.std(off_window)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        
        # ---- SUBPLOT 1: Single Pulse Generative Fit Lines ----
        ax1.plot(raw_pulses[i], label='Raw Single Pulse Data', alpha=0.25, color='gray')
        ax1.plot(smoothed_display_single, label='Median Filtered Signal', color='black', alpha=0.5, linestyle=':')
        ax1.plot(model_pulses[i], label='Optimized Generative Model', color='tab:blue', linewidth=2)
        
        upper_limit = np.mean(off_window) + 3 * rms
        lower_limit = np.mean(off_window) - 3 * rms
        ax1.axhline(y=np.mean(off_window), color='black', alpha=0.3)
        ax1.axhline(y=upper_limit, color='red', linestyle=':', alpha=0.5)
        ax1.axhline(y=lower_limit, color='red', linestyle=':', alpha=0.5)
        
        for j in range(len(peaks[i])):
            ax1.axvline(x=peaks[i][j], c='black', linestyle='--', alpha=0.8)
            
        ax1.set_title(f"Pulse Stack Row {i} Generative Least-Squares Optimization")
        ax1.set_ylabel("Intensity")
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        # ---- SUBPLOT 2: High S/N Integrated Reference Profile Fit Lines with Peak Histogram ----
        ax2.plot(raw_folded_on_pulse, label='Raw Folded Profile Data', alpha=0.25, color='purple')
        ax2.plot(smoothed_display_folded, label='Median Filtered Folded Signal', color='black', alpha=0.5, linestyle=':')
        ax2.plot(folded_fitted_pulses, label='Folded Model (Averaged Single Fits)', color='purple', linewidth=2)
        
        # Plot the histogram-driven folded model profile line
        if np.any(hist_fitted_profile):
            ax2.plot(hist_fitted_profile, label='Histogram-Driven Folded Model', color='tab:green', linewidth=1.8, linestyle='--')
        
        # Overlay the single pulse peak distribution histogram underneath profiles as a twin Y-axis
        if num_bins > 0 and len(single_pulse_peak_bins) > 0:
            ax2_twin = ax2.twinx()
            # Raw histogram counts
            ax2_twin.hist(
                single_pulse_peak_bins, 
                bins=np.linspace(0, on_len, num_bins + 1), 
                color='tab:green', 
                alpha=0.15, 
                edgecolor='tab:green',
                label='Raw Peak counts'
            )
            # Overlay the 1D Gaussian Smoothed Trend curve on top of the bars
            ax2_twin.plot(
                centers, 
                smoothed_counts, 
                color='tab:green', 
                linewidth=1.5, 
                linestyle='-', 
                label='Gaussian Smoothed Trend'
            )
            ax2_twin.set_ylabel("Peak Count Frequency", color='tab:green')
            ax2_twin.tick_params(axis='y', labelcolor='tab:green')
            
        # Draw vertical guidelines representing raw candidate histogram peaks found by find_peaks (before fit)
        for raw_pk in raw_peak_phases:
            ax2.axvline(x=raw_pk, c='tab:red', linestyle='--', alpha=0.85, linewidth=1.5, label='Raw Hist Peak' if raw_pk == raw_peak_phases[0] else "")
            
        ax2.set_title(f"Optimized Folded Reference Profile (Histogram Peaks Found = {len(raw_peak_phases)})")
        ax2.set_xlabel("Relative Window Bins")
        ax2.set_ylabel("Average Intensity")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.4)
        
        plt.tight_layout()
        plt.show()

