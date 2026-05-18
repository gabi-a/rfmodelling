import numpy as np
import skrf as rf
from skrf.circuit import Circuit
from matplotlib import pyplot as plt

# Design a 4-gap Loop Gap Resonator
# Represent the LGR as a parallel RLC circuit, 
# and calculate the coupling capacitance

f0 = 500e6
w0 = 2 * np.pi * f0
Q0 = 600
beta = 1

L = 21.4e-9
C = 1 / (w0 ** 2 * L)
R = Q0 * w0 * L # Equivalent parallel resistance at resonance
Z_0 = 50
# β = R × Z0 × ω²C_c²
Cc = np.sqrt(beta / (R * Z_0 * w0 ** 2))
print(f'Coupling Capacitance: {Cc * 1e15:.2f} fF')

k = np.sqrt((R - Z_0) / (Z_0 * R**2))
wm = (-k + np.sqrt(k**2 + 4 * C / L)) / (2 * C)
Cc_exact = 1 / (wm * np.sqrt(Z_0 * (R - Z_0)))

print(f"Exact critical-coupling frequency: {wm / (2*np.pi) / 1e6:.2f} MHz")
print(f"Exact Cc: {Cc_exact * 1e15:.2f} fF")


freq = rf.Frequency(start=300, stop=600, unit='MHz', npoints=1000)
tline_media = rf.media.DefinedGammaZ0(freq, z0=Z_0)

port1 = Circuit.Port(freq, 'port1', z0=Z_0)
gnd = Circuit.Ground(freq, name='gnd')

C1 = tline_media.capacitor(C, name='C1')
L1 = tline_media.inductor(L, name='L1')
R1 = tline_media.resistor(R, name='R1')

C2 = tline_media.capacitor(Cc, name='C2')

cnx = [
    [(port1, 0), (C2, 0)],
    [(C2, 1), (C1, 0), (L1, 0), (R1, 0)],
    [(gnd, 0), (C1, 1), (L1, 1), (R1, 1)]
]

circuit = Circuit(cnx)
network = circuit.network

circuit.plot_graph(
    port_labels=True,
    network_labels=True,
    edge_labels=True,
    port_fontsize=5,
    network_fontsize=5,
    edge_fontsize=5,
)

plt.figure()
network.plot_s_db(m=0, n=0, lw=2)
plt.grid(True)
plt.title("Parallel RLC with coupling capacitor")
plt.show()