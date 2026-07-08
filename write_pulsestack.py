import numpy as np
import sys
import matplotlib.pyplot as plt


def subtract_base(stack, on1, on2):
    new_stack = []
    for pulse in stack:
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        baseline = np.mean(off_window)
        pulse1 = pulse - baseline
        new_stack.append(np.array(pulse1))
    return np.array(new_stack)


def get_linear(Q, U, rms):
    Q=np.array(Q)
    U=np.array(U)
    l = np.sqrt(Q**2 + U**2)
    psi = []
    for i in range(len(l)):
        if(l[i] > 3.0*rms):
            psi.append(0.5*np.degrees(np.arctan2(U[i], Q[i])))
        else:
            psi.append(np.nan)
    psi = np.array(psi)
    return l, psi

def plot_folded(I, Q, U, V):
    folded_I = np.array(I)
    folded_Q = np.array(Q)
    folded_U = np.array(U)
    folded_V = np.array(V)

    folded_I = folded_I - np.median(folded_I)
    folded_Q = folded_Q - np.median(folded_Q)
    folded_U = folded_U - np.median(folded_U)
    folded_V = folded_V - np.median(folded_V)

    plt.plot(folded_I)
    plt.show()
    off1 = int(input("Enter off start:"))
    off2 = int(input("Enter off end: "))
    rms = np.std(folded_I[off1:off2])

    plt.plot(folded_I)
    plt.axhline(y=3*rms, label="3rms line")
    plt.show()

    l, psi = get_linear(folded_Q, folded_U, rms)
    x = np.arange(0, len(folded_I))
    x = x/max(x)

    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel = 'Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels = [], ylabel='Position angle (PA)')
    ax1.set_xlim(0,1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c= 'k', label='Intensity')
    ax1.axhline(y=3*rms, label="3rms line")
    ax1.plot(x, l, c='r', label='Linear polarization')
    ax1.plot(x, folded_V, c = 'g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()

def get_mod_idx(stack, on1, on2):
    on_summ = []
    numpulses = 0
    for pulse in stack:
        off_window = np.concatenate([pulse[:on1], pulse[on2:]])
        baseline = np.mean(off_window)
        on_window = pulse[on1:on2]
        off_window = off_window - baseline
        on_window = on_window - baseline
        on_len = on2 - on1
        if(np.sum(on_window) > 7.0*np.std(off_window)*np.sqrt(on_len)):
            on_summ.append(np.sum(on_window))
            numpulses += 1
    on_summ = np.array(on_summ)
    mod_idx = np.std(on_summ)/np.mean(on_summ)
    return mod_idx, numpulses

def plot_SP(I, Q, U, V, on1, on2):
    folded_I = np.array(I)
    folded_Q = np.array(Q)
    folded_U = np.array(U)
    folded_V = np.array(V)

    folded_I = folded_I - np.median(folded_I)
    folded_Q = folded_Q - np.median(folded_Q)
    folded_U = folded_U - np.median(folded_U)
    folded_V = folded_V - np.median(folded_V)

    on1 = int(on1)
    on2 = int(on2)
    end_bin = int(len(folded_I)-1)
    rms = np.std(np.concatenate([folded_I[0:on1], folded_I[on2:end_bin]]))


    l, psi = get_linear(folded_Q, folded_U, rms)
    x = np.arange(0, len(folded_I))
    x = x/max(x)

    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel = 'Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels = [], ylabel='Position angle (PA)')
    ax1.set_xlim(x[on1], x[on2])
    ax2.set_xlim(x[on1], x[on2])
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c= 'k', label='Intensity')
    ax1.plot(x, l, c='r', label='Linear polarization')
    ax1.plot(x, folded_V, c = 'g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.12, 0.12, 0.8, 0.6], xlabel = 'Pulse phase', ylabel='Polarization intensity')
    ax2 = fig.add_axes([0.12, 0.72, 0.8, 0.2], xticklabels = [], ylabel='Position angle (PA)')
    ax1.set_xlim(x[on1], x[on2])
    ax2.set_xlim(x[on1], x[on2])
    ax2.set_ylim(-90, 90)
    ax1.plot(x, folded_I, c= 'k', label='Intensity')
    ax1.plot(x, folded_Q, c='r', label='Q')
    ax1.plot(x, folded_U, c='r', label='U')
    ax1.plot(x, folded_V, c = 'g', label='Circular polarization')
    ax2.plot(x, psi)
    plt.show()


print("Python write_pulsestack.py <single_pulse_file> <folded_file> <pulsar_name> <write_directory>")
single_pulse_file = sys.argv[1]
folded_file = sys.argv[2]
pulsar_name = sys.argv[3]
directory = sys.argv[4]

pulse_stack_destI = directory+"/"+pulsar_name+"_pulsestackI.npy"
pulse_stack_destL = directory+"/"+pulsar_name+"_pulsestackL.npy"
pulse_stack_destV = directory+"/"+pulsar_name+"_pulsestackV.npy"
pulsar_stats = directory+"/"+pulsar_name+"_stats.txt"


folded_I = []
folded_Q = []
folded_U = []
folded_V = []

for line in open(folded_file, 'r'):
    if(line[0] != '#'):
        s = [float(r) for r in line.split()]
        folded_I.append(s[1])
        folded_Q.append(s[2])
        folded_U.append(s[3])
        folded_V.append(s[4])


plot_folded(folded_I, folded_Q, folded_U, folded_V)

on1 = int(input("Enter start of on pulse: "))
on2 = int(input("Enter end of on pulse: "))

f = open(pulsar_stats,'w')
f.write(str(on1)+"  "+str(on2)+"\n")
f.close()


SP_I = []
SP_Q = []
SP_U = []
SP_V = []
SP_L = []

pulse_num = 0
I=[]
Q=[]
U=[]
V=[]
L=[]
for line in open(single_pulse_file, 'r'):
    if(line[0] != '#'):
        s=[float(r) for r in line.split()]
        if(s[0] != pulse_num):
            print("Loading next pulse")
            SP_I.append(np.array(I))
            SP_Q.append(np.array(Q))
            SP_U.append(np.array(U))
            SP_V.append(np.array(V))
            SP_L.append(np.array(L))
            I=[]
            Q=[]
            U=[]
            V=[]
            L=[]
            pulse_num = s[0]
        I.append(s[2])
        Q.append(s[3])
        U.append(s[4])
        V.append(s[5])
        L.append(np.sqrt(s[3]*s[3] + s[4]*s[4]))
print("Total number of pulses loaded: "+str(len(SP_I)))

np.save(pulse_stack_destI, subtract_base(SP_I, on1, on2))
np.save(pulse_stack_destL, subtract_base(SP_L, on1, on2))
np.save(pulse_stack_destV, subtract_base(SP_V, on1, on2))

I_modidx, ni = get_mod_idx(SP_I, on1, on2)
L_modidx, nl = get_mod_idx(SP_L, on1, on2)
V_modidx, nv = get_mod_idx(SP_V, on1, on2)
print(f"Number of significant pulses and modulation index for total intensity: {ni} and {I_modidx}")
print(f"Number of significant pulses and modulation index for linear intensity: {nl} and {L_modidx}")
print(f"Number of significant pulses and modulation index for circular intensity: {nv} and {V_modidx}")
