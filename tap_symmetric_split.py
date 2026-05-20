import numpy as np
from scipy.optimize import minimize, differential_evolution
import skrf as rf
from skrf.circuit import Circuit
import matplotlib.pyplot as plt
from utils import ideal_180_hybrid
# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

f0 = 492e6
w0 = 2 * np.pi * f0

Q0 = 600
beta = 2

L = 21.4e-9
Z0 = 50
Ngaps = 4

Ceq = 1 / (w0**2 * L)
Cgap = Ngaps * Ceq

Lseg = L / Ngaps
Rs_total = w0 * L / Q0
Rseg = Rs_total / Ngaps

Rpar = Q0 * w0 * L
Rgap  = Rpar / Ngaps**2
Rhalf = Rpar / (Ngaps / 2)**2

Zdiff_target = 2 * Z0

N_caps = 1
C_single_cap = Cgap / N_caps
Clump = (N_caps - 1) * C_single_cap
Csplit = Cgap - Clump    # Equivalent to C_single_cap


Ca = 2 * Cgap / (1 - (Zdiff_target * beta / Rgap)**0.5) / N_caps
# Equivalent defintion:
#   Ca = 2 * Csplit / (1 - 2 * (beta * Zdiff_target / Rhalf)**0.5) 
Cb = 1 / (1 / Csplit - 1 / Ca)

print("Ca = {:.2f} pF".format(Ca * 1e12))
print("Cb = {:.2f} pF".format(Cb * 1e12))

# Check total adds to gap cap
print(((1 / Ca + 1 / Cb)**-1 + Clump) * 1e12, "pF total across gap")
print(Cgap * 1e12, "pF original gap cap")

# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

freq = rf.Frequency(start=f0-5e6, stop=f0+5e6, unit='Hz', npoints=3001)
media = rf.media.DefinedGammaZ0(freq, z0=Z0) # type: ignore

def make_tapped_lgr(
        media, freq, Cgap, Rseg, Lseg, Z0, Ca, Cb, Clump) -> Circuit:
    """
    Floating 4-gap LGR driven through an ideal 180-degree hybrid.
    """

    port1 = Circuit.Port(freq, 'port1', z0=Z0)
    gnd = Circuit.Ground(freq, name='gnd')

    balun = ideal_180_hybrid(freq, z0=Z0, name='balun')

    Riso = media.resistor(Z0, name='Riso')

    Cg1_A = media.capacitor(Ca, name='Cg1_A')
    Cg1_B = media.capacitor(Cb, name='Cg1_B')

    Cg2 = media.capacitor(Cgap, name='Cg2')

    Cg3_A = media.capacitor(Ca, name='Cg3_A')
    Cg3_B = media.capacitor(Cb, name='Cg3_B')

    Cg4 = media.capacitor(Cgap, name='Cg4')

    C_lump1 = media.capacitor(Clump, name='Clump1')
    C_lump3 = media.capacitor(Clump, name='Clump3')

    R1 = media.resistor(Rseg, name='R1')
    R2 = media.resistor(Rseg, name='R2')
    R3 = media.resistor(Rseg, name='R3')
    R4 = media.resistor(Rseg, name='R4')

    L1 = media.inductor(Lseg, name='L1')
    L2 = media.inductor(Lseg, name='L2')
    L3 = media.inductor(Lseg, name='L3')
    L4 = media.inductor(Lseg, name='L4')

    cnx = [
        # Single-ended input into hybrid
        [(port1, 0), (balun, 0)],

        # Isolated port terminated
        [(balun, 3), (Riso, 0)],
        [(Riso, 1), (gnd, 0)],

        # Balun connected to tapped point in LGR
        [(balun, 1), (Cg1_B, 0), (Cg1_A, 1)],
        [(Cg1_B, 1), (R1, 0), (C_lump1, 1)],

        # Segment 1
        [(R1, 1), (L1, 0)],
        [(L1, 1), (Cg2, 0)],

        # Segment 2
        [(Cg2, 1), (R2, 0)],
        [(R2, 1), (L2, 0)],
        [(L2, 1), (Cg3_B, 0), (C_lump3, 0)],
        [(Cg3_B, 1), (Cg3_A, 0), (balun, 2)],

        # Segment 3
        [(Cg3_A, 1), (R3, 0), (C_lump3, 1)],
        [(R3, 1), (L3, 0)],
        [(L3, 1), (Cg4, 0)],

        # Segment 4, closing floating loop
        [(Cg4, 1), (R4, 0)],
        [(R4, 1), (L4, 0)],
        [(L4, 1), (Cg1_A, 0), (C_lump1, 0)],
    ]

    return Circuit(cnx)

circuit = make_tapped_lgr(
    media=media,
    freq=freq,
    Cgap=Cgap,
    Rseg=Rseg,
    Lseg=Lseg,
    Z0=Z0,
    Ca=Ca,
    Cb=Cb,
    Clump=Clump,
)

network = circuit.network

s11_db = 20 * np.log10(np.abs(network.s[:, 0, 0]))
Zin = network.z[:, 0, 0]

print(Zin[np.argmin(np.abs(freq.f - f0))])

circuit.plot_graph(
    port_labels=True,
    network_labels=True,
    edge_labels=True,
    port_fontsize=10,
    network_fontsize=10,
    edge_fontsize=5,
)

plt.figure()
plt.plot(freq.f / 1e6, s11_db, lw=2)
plt.axvline(f0 / 1e6, linestyle='--', label='500 MHz')
plt.xlabel("Frequency [MHz]")
plt.ylabel("S11 [dB]")
plt.grid(True)
plt.legend()

plt.show()