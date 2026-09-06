import os
import argparse
import glob
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from psrqpy import QueryATNF

from modules.mod_index import get_mod_idx_summed, get_mod_idx_resolved
from modules.pulse_fitting import find_optimized_generative_peaks_batch, fit_profile_with_histogram_peaks

def find_unique_pulsars(folder):
    stats_files = glob.glob(os.path.join(folder, "*_stats.txt"))
    pulsars = []
    for f in stats_files:
        pulsars.append(os.path.basename(f).replace("_stats.txt", ""))
    return sorted(pulsars)

def process_pulsar(folder, pulsar_name, threshold):
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

    x_axis = np.arange(on_len)
    phase_axis = (x_axis + on1) / total_bins

    folded_profile_full = np.mean(bright_pulses_full, axis=0) if len(bright_pulses_full) > 0 else np.mean(SP_I, axis=0)
    raw_folded_on_pulse = folded_profile_full[on1:on2]

    hist_fitted_profile, histogram_peak_locations, raw_counts, smoothed_counts, centers, num_bins, raw_peak_phases = fit_profile_with_histogram_peaks(
        single_pulse_peak_bins, raw_folded_on_pulse, on_len, num_bright_pulses
    )

    phase_histogram_peaks_count = len(raw_peak_phases)
    single_pulse_peaks_80th = np.percentile(peaks_per_pulse, 80) if len(peaks_per_pulse) > 0 else 0.0

    global_peak_phases_list = [(p + on1)/total_bins for p in single_pulse_peak_bins]
    global_profile = np.mean(SP_I[:, on1:on2], axis=0)

    fig_stats = plt.figure(figsize=(9.5, 15))
    fig_stats.suptitle(f"Single pulse statistics: {pulsar_name}{p1_string}", fontsize=15, fontweight='bold', y=0.97)
    
    ax1 = fig_stats.add_subplot(4, 1, 1)
    ax2 = fig_stats.add_subplot(4, 1, 2)
    ax3a = fig_stats.add_subplot(4, 2, 5) 
    ax3b = fig_stats.add_subplot(4, 2, 6) 
    ax4 = fig_stats.add_subplot(4, 1, 4)

    phase_range0 = np.linspace(on1 / total_bins, on2 / total_bins, on_len)

    ax1.plot(phase_range0, global_profile, color='black', label="Global Mean Profile", zorder=2)
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

    fig_stack = plt.figure(figsize=(13, 9.5))
    fig_stack.suptitle(f"Continuous Pulse Stack Topography: {pulsar_name}", fontsize=14, fontweight='bold', y=0.96)
    gs = fig_stack.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.15, wspace=0.18)

    prof_ax1 = fig_stack.add_subplot(gs[0, 0])
    prof_ax2 = fig_stack.add_subplot(gs[0, 1])
    stack_ax1 = fig_stack.add_subplot(gs[1, 0], sharex=prof_ax1)
    stack_ax2 = fig_stack.add_subplot(gs[1, 1], sharex=prof_ax2)

    continuous_raw_stack = SP_I[:, on1:on2]
    continuous_model_stack = np.zeros((total_pulses_count, on_len))
    if len(model_pulses) > 0:
        for b_idx, orig_row in enumerate(successful_indices):
            if b_idx < len(model_pulses):
                continuous_model_stack[orig_row, :] = model_pulses[b_idx]

    prof_ax1.plot(phase_axis, raw_folded_on_pulse, color='purple', linewidth=2)
    prof_ax1.set_title("Raw Integrated Profile Data", fontsize=11, fontweight='bold')
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

    return fig_stats, fig_stack, p0_val, p1_val, phase_histogram_peaks_count, modidxI, single_pulse_peaks_80th

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs a single pulse fitting for bright pulses (above threshold) of all the pulsars in a folder.")
    parser.add_argument("threshold", type=float, help="Signal-to-noise threshold for bright pulse selection.")
    parser.add_argument("directory", type=str, help="Path to the directory containing pulsar data files.")
    args = parser.parse_args()

    threshold_val = args.threshold
    folder_path = args.directory
    
    output_pdf_stats = os.path.join(folder_path, "compiled_pulsar_plots.pdf")
    output_pdf_stacks = os.path.join(folder_path, "compiled_pulse_stacks.pdf")
    output_txt_summary = os.path.join(folder_path, "analysis_summary_table.txt")

    pulsar_list = find_unique_pulsars(folder_path)
    print(f"Found {len(pulsar_list)} pulsars to process...")

    with open(output_txt_summary, "w") as txt_out:
        txt_out.write("Pulsar_Name\tPeriod_P(s)\tP_dot(s/s)\tSummed_Mod_Index\tHistogram_Peaks_Count\tSingle_Pulse_Peaks_80th_Percentile\n")

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
                    
                    with open(output_txt_summary, "a") as txt_out:
                        p0_str = f"{p0:.6f}" if p0 is not None else "N/A"
                        p1_str = f"{p1:.6e}" if p1 is not None else "N/A"
                        txt_out.write(f"{psr}\t{p0_str}\t{p1_str}\t{mod_idx_sum:.4f}\t{hist_pks}\t{single_pulse_80th:.2f}\n")
            except Exception as e:
                print(f"Error processing {psr}: {e}")

    print(f"\nSuccessfully compiled statistical analysis into: {output_pdf_stats}")
    print(f"Successfully compiled comparative pulse stack views into: {output_pdf_stacks}")
    print(f"Successfully generated summary database matrix file to: {output_txt_summary}")
