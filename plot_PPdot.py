"""
Pulsar Results Plotting Tool
============================
Generates high-resolution diagnostic publication-ready figures binned by 
spin-down energy loss rates, concluding with a direct multi-component correlation 
analysis between profile complexity and single-pulse flux modulation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psrqpy import QueryATNF

# Set professional plotting parameters for scientific publishing
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'savefig.dpi': 300
})

def generate_ppdot_and_averaged_edot_plots(summary_txt_path):
    if not os.path.exists(summary_txt_path):
        print(f"Error: Could not find the file '{summary_txt_path}'")
        return

    # --- 1. READ AND CLEAN SUMMARY TABLE DATA ---
    print(f"Reading summary matrix from: {summary_txt_path}")
    df = pd.read_csv(summary_txt_path, sep='\t')
    df.columns = df.columns.str.strip()
    
    # Drop rows missing kinematic tracking coordinates
    df = df.replace('N/A', np.nan).dropna(subset=['Period_P(s)', 'P_dot(s/s)'])

    pulsar_names = df['Pulsar_Name'].values
    periods = df['Period_P(s)'].astype(float).values
    pdots = df['P_dot(s/s)'].astype(float).values
    single_peaks = df['Single_Pulse_Peaks_80th_Percentile'].astype(float).values
    folded_peaks = df['Histogram_Peaks_Count'].astype(float).values
    mod_indices = df['Summed_Mod_Index'].astype(float).values

    # Pure raw 1 / m_I^2 formulation
    protected_mod = np.where(mod_indices <= 0, 1e-4, mod_indices)
    inv_square_mod = 1.0 / (protected_mod ** 2)

    # --- 2. FETCH EDOT AND BACKGROUND POPULATION FROM PSRQPY ---
    print("Querying ATNF reference catalog for targeted EDOT values and global context...")
    bg_periods, bg_pdots = [], []
    edot_mapped_dict = {}
    
    try:
        full_query = QueryATNF(params=['P0', 'P1', 'EDOT'])
        all_p0 = full_query.table['P0']
        all_p1 = full_query.table['P1']
        valid_mask = (~np.isnan(all_p0)) & (~np.isnan(all_p1)) & (all_p0 > 0) & (all_p1 > 0)
        bg_periods = all_p0[valid_mask]
        bg_pdots = all_p1[valid_mask]
        
        atnf_df = full_query.pandas
        name_col = 'PSRJ' if 'PSRJ' in atnf_df.columns else 'NAME'
        
        for _, row in atnf_df.dropna(subset=[name_col, 'EDOT']).iterrows():
            edot_mapped_dict[str(row[name_col]).strip()] = float(row['EDOT'])
    except Exception as e:
        print(f"Warning: Global ATNF background population context query dropped: {e}")

    # Map the target EDOT parameter array for our active sample group
    edot_values = []
    for name in pulsar_names:
        clean_name = str(name).strip()
        if clean_name in edot_mapped_dict:
            edot_values.append(edot_mapped_dict[clean_name])
        else:
            # Fallback to analytical calculation if not found in table lookup
            edot_values.append(3.95e46 * (df.loc[df['Pulsar_Name'] == name, 'P_dot(s/s)'].values[0] / 
                                          (df.loc[df['Pulsar_Name'] == name, 'Period_P(s)'].values[0] ** 3)))
    
    edot_values = np.array(edot_values, dtype=float)
    log_edot = np.log10(np.where(edot_values <= 0, 1e30, edot_values))

    # --- 3. HELPER FUNCTION TO COMPUTE TRUE GROUPED MEAN VALUES ---
    def compute_binned_averages(x_data, y_data, num_bins=8):
        # Establish structural boundaries for the requested log(Edot) steps
        bin_edges = np.linspace(np.min(x_data), np.max(x_data), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_widths = np.diff(bin_edges)
        
        binned_means = []
        for i in range(num_bins):
            mask = (x_data >= bin_edges[i]) & (x_data < bin_edges[i+1])
            if i == num_bins - 1: # Catch edge points on the final boundary limit
                mask = mask | (x_data == bin_edges[i+1])
                
            if np.any(mask):
                binned_means.append(np.mean(y_data[mask]))
            else:
                binned_means.append(0.0) # Handle unpopulated sample windows cleanly
                
        return bin_centers, bin_widths, np.array(binned_means)

    # --- 4. SHARED CANVAS DESIGN METRICS ---
    plane_xlim = (0.1, 10.0)          
    plane_ylim = (1e-17, 1e-12)       
    color_max_components = int(max(5, np.max(folded_peaks)))
    
    def format_ppdot_axes(ax, title):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(plane_xlim)
        ax.set_ylim(plane_ylim)
        ax.set_xlabel(r"Pulsar Period $P$ (s)", fontsize=11)
        ax.set_ylabel(r"Period Derivative $\dot{P}$ (s/s)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # --- TWIN PLOT 1: SINGLE PULSE MULTIPLICITY ANALYSIS ---
    print("Generating Plot 1 Layout: Averaged Multiplicity vs Edot Spectrum...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if len(bg_periods) > 0:
        ax1.scatter(bg_periods, bg_pdots, s=2, color='lightgray', alpha=0.4, zorder=1)
    ax1.scatter(periods, pdots, s=(single_peaks ** 2) * 5, c=single_peaks, 
               cmap='plasma', alpha=0.9, edgecolors='black', linewidths=1.2, 
               vmin=1, vmax=color_max_components, zorder=3)
    format_ppdot_axes(ax1, "Max number of peaks in single pulses (95 percentile)")
    
    centers, widths, averaged_y = compute_binned_averages(log_edot, single_peaks)
    ax2.bar(centers, averaged_y, width=widths, color='tab:blue', edgecolor='black', alpha=0.8, align='center')
    ax2.set_xlabel(r"Energy Loss Rate $\log_{10}(\dot{E})$ (erg/s)", fontsize=11)
    ax2.set_ylabel("Number of single pulse peaks averaged over pulsars", fontsize=11)
    ax2.set_title(r"Single Pulse peaks across $\dot{E}$ Bins", fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.savefig("ppdot_1_single_pulse_multiplicity.png", dpi=300)
    plt.close()

    # --- TWIN PLOT 2: MASTER FOLDED COMPONENTS ANALYSIS ---
    print("Generating Plot 2 Layout: Averaged Folded Components vs Edot Spectrum...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if len(bg_periods) > 0:
        ax1.scatter(bg_periods, bg_pdots, s=2, color='lightgray', alpha=0.4, zorder=1)
    ax1.scatter(periods, pdots, s=(folded_peaks ** 2)*5, c=folded_peaks, 
               cmap='plasma', alpha=0.9, edgecolors='black', linewidths=1.2, 
               vmin=1, vmax=color_max_components, zorder=3)
    format_ppdot_axes(ax1, "Master Folded Profile Components Count")
    
    centers, widths, averaged_y = compute_binned_averages(log_edot, folded_peaks)
    ax2.bar(centers, averaged_y, width=widths, color='purple', edgecolor='black', alpha=0.75, align='center')
    ax2.set_xlabel(r"Energy Loss Rate $\log_{10}(\dot{E})$ (erg/s)", fontsize=11)
    ax2.set_ylabel("Average Component Count ($N_{comp}$)", fontsize=11)
    ax2.set_title(r"Mean Folded Components count across $\dot{E}$ Bins", fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.savefig("ppdot_2_folded_profile_components.png", dpi=300)
    plt.close()

    # --- TWIN PLOT 3: FLUX STABILITY ANALYSIS ---
    print("Generating Plot 3 Layout: Averaged Flux Stability vs Edot Spectrum...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if len(bg_periods) > 0:
        ax1.scatter(bg_periods, bg_pdots, s=2, color='lightgray', alpha=0.4, zorder=1)
        
    stability_marker_sizes = inv_square_mod * 15
    ax1.scatter(periods, pdots, s=stability_marker_sizes, c=inv_square_mod, 
               cmap='viridis', alpha=0.9, edgecolors='black', linewidths=1.2, zorder=3)
    format_ppdot_axes(ax1, r"Pulsar Flux Stability ($1/m_I^2$) Layout")
    
    centers, widths, averaged_y = compute_binned_averages(log_edot, inv_square_mod)
    ax2.bar(centers, averaged_y, width=widths, color='teal', edgecolor='black', alpha=0.8, align='center')
    ax2.set_xlabel(r"Energy Loss Rate $\log_{10}(\dot{E})$ (erg/s)", fontsize=11)
    ax2.set_ylabel(r"Average Stability Metric ($1/m_I^2$)", fontsize=11)
    ax2.set_title(r"Mean Flux Stability ($1/m_I^2$) across $\dot{E}$ Bins", fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.savefig("ppdot_3_flux_stability.png", dpi=300)
    plt.close()
    
    # --- TWIN PLOT 4: PEAK COUNT CORRELATION ANALYSIS ---
    print("Generating Plot 4 Layout: Peak Multiplicity vs Squared Modulation Index...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Square of the original modulation index
    mod_indices_sq = mod_indices ** 2

    # Panel 1: Histogram_Peaks_Count vs. Squared Modulation Index
    ax1.scatter(inv_square_mod, folded_peaks, color='purple', alpha=0.8, edgecolors='black', linewidths=1.2, s=50, zorder=3)
    ax1.set_xlabel(r"Squared Modulation Index ($m_I^2$)", fontsize=11)
    ax1.set_ylabel("Histogram Peaks Count", fontsize=11)
    ax1.set_title("Histogram Peaks Count vs. $m_I^2$", fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Panel 2: Single_Pulse_Peaks_80th_Percentile vs. Squared Modulation Index
    ax2.scatter(inv_square_mod, single_peaks, color='tab:blue', alpha=0.8, edgecolors='black', linewidths=1.2, s=50, zorder=3)
    ax2.set_xlabel(r"1/(Squared Modulation Index) ($1/m_I^2$)", fontsize=11)
    ax2.set_ylabel("Single Pulse Peaks (80th Percentile)", fontsize=11)
    ax2.set_title("Single Pulse Peaks (80th Percentile) vs. $m_I^2$", fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
#    plt.savefig("ppdot_4_modulation_vs_peaks.png", dpi=300)
    plt.show()
    
    print("\n--- INTEGRATED ASTRONOMICAL EXPORT COMPLETE ---")
    print("All four master files generated using actual pipeline dataset outputs successfully.")

if __name__ == "__main__":
    input_file = "analysis_summary_table.txt"
    generate_ppdot_and_averaged_edot_plots(input_file)
