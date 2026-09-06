import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

def multi_gaussian_profile(x, *params):
    """Generates a composite model profile from unpacked parameters."""
    model = np.zeros(len(x), dtype=float)
    for i in range(0, len(params), 3):
        amp = params[i]
        mean = params[i+1]
        sigma = params[i+2]
        model += amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return model

def find_optimized_generative_peaks_batch(bright_pulses_full, bright_indices, on1, on2, total_bins, threshold_multiplier=3.0):
    """
    Optimizes component detections across individual bright single pulses,
    internally computing the running median kernel size based on duty cycle.
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
            np.array(all_peak_phases),
            successful_indices)

def fit_profile_with_histogram_peaks(single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses):
    """Builds an adaptive histogram of peak detections and fits a multi-component Gaussian model."""
    hist_fitted_profile = np.zeros(on_len)
    histogram_peak_locations = []
    raw_counts = np.array([])
    smoothed_counts = np.array([])
    centers = np.array([])
    num_bins = 0
    raw_peak_phases = []

    if len(single_pulse_peak_bins) == 0:
        return hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases

    n_points = len(single_pulse_peak_bins)
    num_bins = int(np.round(max(50, np.sqrt(n_points))))
    
    counts, edges = np.histogram(single_pulse_peak_bins, bins=np.linspace(0, on_len, num_bins + 1))
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    raw_counts = counts.astype(float)
    smoothening_scale = int(max(1.5, num_bins/30.0))
    
    smoothed_counts = gaussian_filter1d(raw_counts, sigma=smoothening_scale)
    
    height = max(4.0, num_bright_pulses / 100.0)
    min_distance = max(2, int(num_bins * 0.03))
    
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
        except Exception:
            pass
            
    return hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases
