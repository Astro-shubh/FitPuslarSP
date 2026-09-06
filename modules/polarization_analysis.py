import numpy as np
import matplotlib.pyplot as plt

def subtract_base(stack, on1, on2):
    """Subtracts the off-pulse baseline from each pulse in the stack."""
    new_stack = []
    for pulse in stack:
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        baseline = np.mean(off_window)
        pulse1 = pulse - baseline
        new_stack.append(np.array(pulse1))
    return np.array(new_stack)

def get_linear(Q, U, rms):
    """Computes linear polarization intensity and position angle (PA)."""
    Q = np.array(Q)
    U = np.array(U)
    l = np.sqrt(Q**2 + U**2)
    psi = []
    for i in range(len(l)):
        if l[i] > 3.0 * rms:
            psi.append(0.5 * np.degrees(np.arctan2(U[i], Q[i])))
        else:
            psi.append(np.nan)
    psi = np.array(psi)
    return l, psi

def get_mod_idx(stack, on1, on2):
    """Computes modulation index and significant pulse counts for polarization data."""
    on_summ = []
    numpulses = 0
    for pulse in stack:
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        baseline = np.mean(off_window)
        on_window = pulse[on1:on2]
        off_window = off_window - baseline
        on_window = on_window - baseline
        on_len = on2 - on1
        if np.sum(on_window) > 7.0 * np.std(off_window) * np.sqrt(on_len):
            on_summ.append(np.sum(on_window))
            numpulses += 1
    on_summ = np.array(on_summ)
    mod_idx = np.std(on_summ) / np.mean(on_summ) if len(on_summ) > 0 else 0.0
    return mod_idx, numpulses

def plot_folded(I, Q, U, V):
    """Interactive plotting function for folded polarization profiles."""
    folded_I = np.array(I) - np.median(I)
    folded_Q = np.array(Q) - np.median(Q)
    folded_U = np.array(U) - np.median(U)
    folded_V = np.array(V) - np.median(V)

    plt.plot(folded_I)
    plt.show()
    off1 = int(input("Enter off start:"))
    off2 = int(input("Enter off end: "))
    rms = np.std(folded_I[off1:off2])

    plt.plot(folded_I)
    plt.axhline(y=3 * rms, label="3rms line")
    plt.show()

    l, psi = get_linear(folded_Q, folded_U, rms)
    x = np.arange(0, len(folded_I))
    x = x / max(x)

    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel='Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels=[], ylabel='Position angle (PA)')
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c='k', label='Intensity')
    ax1.axhline(y=3 * rms, label="3rms line")
    ax1.plot(x, l, c='r', label='Linear polarization')
    ax1.plot(x, folded_V, c='g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()

def plot_SP(I, Q, U, V, on1, on2):
    """Plots polarization characteristics for single pulses."""
    folded_I = np.array(I) - np.median(I)
    folded_Q = np.array(Q) - np.median(Q)
    folded_U = np.array(U) - np.median(U)
    folded_V = np.array(V) - np.median(V)

    on1 = int(on1)
    on2 = int(on2)
    end_bin = int(len(folded_I) - 1)
    rms = np.std(np.concatenate([folded_I[0:on1], folded_I[on2:end_bin]]))

    l, psi = get_linear(folded_Q, folded_U, rms)
    x = np.arange(0, len(folded_I))
    x = x / max(x)

    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel='Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels=[], ylabel='Position angle (PA)')
    ax1.set_xlim(x[on1], x[on2])
    ax2.set_xlim(x[on1], x[on2])
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c='k', label='Intensity')
    ax1.plot(x, l, c='r', label='Linear polarization')
    ax1.plot(x, folded_V, c='g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()

    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel='Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels=[], ylabel='Position angle (PA)')
    ax1.set_xlim(x[on1], x[on2])
    ax2.set_xlim(x[on1], x[on2])
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c='k', label='Intensity')
    ax1.plot(x, folded_Q, c='r', label='Q')
    ax1.plot(x, folded_U, c='r', label='U')
    ax1.plot(x, folded_V, c='g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()
