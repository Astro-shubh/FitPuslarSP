import argparse
import numpy as np
from modules.polarization_analysis import subtract_base, plot_folded, get_mod_idx

parser = argparse.ArgumentParser(description="Process and write polarization pulsar stacks.")
parser.add_argument("single_pulse_file", type=str, help="Path to the single pulse data file.")
parser.add_argument("folded_file", type=str, help="Path to the folded profile file.")
parser.add_argument("pulsar_name", type=str, help="Name of the pulsar.")
parser.add_argument("write_directory", type=str, help="Directory to save output files.")
args = parser.parse_args()

single_pulse_file = args.single_pulse_file
folded_file = args.folded_file
pulsar_name = args.pulsar_name
directory = args.write_directory

pulse_stack_destI = directory + "/" + pulsar_name + "_pulsestackI.npy"
pulse_stack_destL = directory + "/" + pulsar_name + "_pulsestackL.npy"
pulse_stack_destV = directory + "/" + pulsar_name + "_pulsestackV.npy"
pulsar_stats = directory + "/" + pulsar_name + "_stats.txt"

folded_I = []
folded_Q = []
folded_U = []
folded_V = []

for line in open(folded_file, 'r'):
    if line[0] != '#':
        s = [float(r) for r in line.split()]
        folded_I.append(s[1])
        folded_Q.append(s[2])
        folded_U.append(s[3])
        folded_V.append(s[4])

plot_folded(folded_I, folded_Q, folded_U, folded_V)

on1 = int(input("Enter start of on pulse: "))
on2 = int(input("Enter end of on pulse: "))

with open(pulsar_stats, 'w') as f:
    f.write(str(on1) + "  " + str(on2) + "\n")

SP_I = []
SP_Q = []
SP_U = []
SP_V = []
SP_L = []

pulse_num = 0
I = []
Q = []
U = []
V = []
L = []

for line in open(single_pulse_file, 'r'):
    if line[0] != '#':
        s = [float(r) for r in line.split()]
        if s[0] != pulse_num:
            print("Loading next pulse")
            SP_I.append(np.array(I))
            SP_Q.append(np.array(Q))
            SP_U.append(np.array(U))
            SP_V.append(np.array(V))
            SP_L.append(np.array(L))
            I = []
            Q = []
            U = []
            V = []
            L = []
            pulse_num = s[0]
        I.append(s[2])
        Q.append(s[3])
        U.append(s[4])
        V.append(s[5])
        L.append(np.sqrt(s[3] * s[3] + s[4] * s[4]))

print("Total number of pulses loaded: " + str(len(SP_I)))

np.save(pulse_stack_destI, subtract_base(SP_I, on1, on2))
np.save(pulse_stack_destL, subtract_base(SP_L, on1, on2))
np.save(pulse_stack_destV, subtract_base(SP_V, on1, on2))

I_modidx, ni = get_mod_idx(SP_I, on1, on2)
L_modidx, nl = get_mod_idx(SP_L, on1, on2)
V_modidx, nv = get_mod_idx(SP_V, on1, on2)

print(f"Number of significant pulses and modulation index for total intensity: {ni} and {I_modidx}")
print(f"Number of significant pulses and modulation index for linear intensity: {nl} and {L_modidx}")
print(f"Number of significant pulses and modulation index for circular intensity: {nv} and {V_modidx}")
