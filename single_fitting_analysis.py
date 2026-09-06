import os
import sys
import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Import from modular files
from modules.mod_index import get_mod_idx_summed, get_mod_idx_resolved
from modules.pulse_fitting import find_optimized_generative_peaks_batch, fit_profile_with_histogram_peaks

def main():
    parser = argparse.ArgumentParser(description="Process single pulse fitting and analysis for a single pulsar.")
    parser.add_argument("folder", type=str, help="Directory containing the pulsar data files.")
    parser.add_argument("psrname", type=str, help="Name of the pulsar.")
    parser.add_argument("threshold", type=float, help="Threshold multiplier for pulse selection.")
    args = parser.parse_args()

    folder = args.folder
    psr_name = args.psrname
    threshold = args.threshold

    totalI_stack = os.path.join(folder, f"{psr_name}_pulsestackI.npy")
    on_off = os.path.join(folder, f"{psr_name}_stats.txt")

    if not (os.path.exists(totalI_stack) and os.path.exists(on_off)):
        print(f"Error: Required files for {psr_name} not found in {folder}.")
        sys.exit(1)

    stack = np.load(totalI_stack)
    total_pulses_count, total_bins = stack.shape
    stats_data = np.loadtxt(on_off, dtype=int)
    on1, on2 = int(stats_data[0]), int(stats_data[1])
    on_len = on2 - on1

    duty_cycle = (on2 - on1) / total_bins
    print(f"On-pulse boundaries: {on1} to {on2}")

    if duty_cycle < 0.04:
        calculated_kernel = int(np.floor(total_bins / 500.0))
    else:
        calculated_kernel = int(np.floor(total_bins / 300.0))
    if calculated_kernel % 2 == 0:
        calculated_kernel += 1  
    kernel_size = max(5, calculated_kernel)
    print(f"Running Median Kernel Size: {kernel_size} bins")

    idx, error, length, bright_pulses, bright_indices = get_mod_idx_resolved(stack, on1, on2, threshold)
    modidxI, modidxI_error, _ = get_mod_idx_summed(stack, on1, on2, threshold)

    peaks, npeaks, raw_pulses_full, model_pulses, single_pulse_peak_bins, successful_indices = find_optimized_generative_peaks_batch(
        bright_pulses, bright_indices, on1, on2, total_bins, threshold_multiplier=3.0
    )

    if len(raw_pulses_full) > 0 and raw_pulses_full.ndim == 1:
        raw_pulses_full = np.atleast_2d(raw_pulses_full)
        model_pulses = np.atleast_2d(model_pulses)
        peaks = np.array([peaks], dtype=object)

    raw_pulses = raw_pulses_full[:, on1:on2] if len(raw_pulses_full) > 0 else np.array([])
    folded_fitted_pulses = np.mean(model_pulses, axis=0) if len(model_pulses) > 0 else np.zeros(on_len)
    folded_profile_full = np.mean(bright_pulses, axis=0) if len(bright_pulses) > 0 else np.mean(stack, axis=0)
    raw_folded_on_pulse = folded_profile_full[on1:on2]
    num_bright_pulses = raw_pulses.shape[0] if len(raw_pulses) > 0 else 0

    continuous_raw_stack = stack[:, on1:on2]
    continuous_model_stack = np.zeros((total_pulses_count, on_len))
    if len(model_pulses) > 0:
        for b_idx, orig_row in enumerate(successful_indices):
            if b_idx < len(model_pulses):
                continuous_model_stack[orig_row, :] = model_pulses[b_idx]

    print("Executing Histogram-driven profile fitting engine...")
    x_axis = np.arange(on_len)
    phase_axis = (x_axis + on1) / total_bins

    hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases = fit_profile_with_histogram_peaks(
        single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses
    )

    smoothed_display_folded = signal.medfilt(folded_profile_full, kernel_size=kernel_size)[on1:on2]

    print("Generating side-by-side comparative pulse stack file panel...")
    fig_stacks = plt.figure(figsize=(14, 10))
    gs = fig_stacks.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.15, wspace=0.18)

    prof_ax1 = fig_stacks.add_subplot(gs[0, 0])
    prof_ax2 = fig_stacks.add_subplot(gs[0, 1])
    stack_ax1 = fig_stacks.add_subplot(gs[1, 0], sharex=prof_ax1)
    stack_ax2 = fig_stacks.add_subplot(gs[1, 1], sharex=prof_ax2)

    prof_ax1.plot(phase_axis, raw_folded_on_pulse, color='purple', linewidth=2)
    prof_ax1.set_title(f"Raw Integrated Profile ({psr_name})", fontsize=11, fontweight='bold')
    prof_ax1.set_ylabel("Average Intensity")
    prof_ax1.grid(True, linestyle=':', alpha=0.4)
    prof_ax1.tick_params(labelbottom=False)

    stack_ax1.imshow(continuous_raw_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax1.set_title("Continuous Raw Data Pulse Stack", fontsize=11)
    stack_ax1.set_xlabel(r"Pulsar Rotation Phase ($\phi$)")
    stack_ax1.set_ylabel("Continuous Pulse Number")

    prof_ax2.plot(phase_axis, folded_fitted_pulses, color='purple', linewidth=2, linestyle=':', label='Averaged Single Fits')
    if np.any(hist_fitted_profile):
        prof_ax2.plot(phase_axis, hist_fitted_profile, color='tab:green', linewidth=2, linestyle='--', label='Histogram-Driven Fit')
    prof_ax2.set_title("Optimized Fitted Models Summary", fontsize=11, fontweight='bold')
    prof_ax2.grid(True, linestyle=':', alpha=0.4)
    prof_ax2.legend(loc='upper right', fontsize=9)
    prof_ax2.tick_params(labelbottom=False)

    stack_ax2.imshow(continuous_model_stack, aspect='auto', cmap='viridis', origin='lower', extent=[phase_axis[0], phase_axis[-1], 0, total_pulses_count])
    stack_ax2.set_title("Continuous Fitted Model Stack (Zero-Filled)", fontsize=11)
    stack_ax2.set_xlabel(r"Pulsar Rotation Phase ($\phi$)")

    output_stack_png = os.path.join(folder, f"{psr_name}_pulse_stacks.png")
    plt.savefig(output_stack_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean diagnostic comparison matrix saved successfully to: {output_stack_png}")

    for i in range(len(peaks)):
        smoothed_display_single = signal.medfilt(raw_pulses_full[i], kernel_size=kernel_size)[on1:on2]
        off_window = np.concatenate((raw_pulses_full[i, 0:on1], raw_pulses_full[i, on2:-1]))
        rms = np.std(off_window)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        
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
        
        ax2.plot(raw_folded_on_pulse, label='Raw Folded Profile Data', alpha=0.25, color='purple')
        ax2.plot(smoothed_display_folded, label='Median Filtered Folded Signal', color='black', alpha=0.5, linestyle=':')
        ax2.plot(folded_fitted_pulses, label='Folded Model (Averaged Single Fits)', color='purple', linewidth=2)
        
        if np.any(hist_fitted_profile):
            ax2.plot(hist_fitted_profile, label='Histogram-Driven Folded Model', color='tab:green', linewidth=1.8, linestyle='--')
        
        if num_bins > 0 and len(single_pulse_peak_bins) > 0:
            ax2_twin = ax2.twinx()
            ax2_twin.hist(
                single_pulse_peak_bins, 
                bins=np.linspace(0, on_len, num_bins + 1), 
                color='tab:green', 
                alpha=0.15, 
                edgecolor='tab:green',
                label='Raw Peak counts'
            )
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
            
        for raw_pk in raw_peak_phases:
            ax2.axvline(x=raw_pk, c='tab:red', linestyle='--', alpha=0.85, linewidth=1.5, label='Raw Hist Peak' if raw_pk == raw_peak_phases[0] else "")
            
        ax2.set_title(f"Optimized Folded Reference Profile (Histogram Peaks Found = {len(raw_peak_phases)})")
        ax2.set_xlabel("Relative Window Bins")
        ax2.set_ylabel("Average Intensity")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.4)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
