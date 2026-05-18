import numpy as np
import skrf as rf
from skrf.circuit import Circuit
from matplotlib import pyplot as plt


# ------------------------------------------------------------
# Resonator parameters
# ------------------------------------------------------------

f0 = 500e6
w0 = 2 * np.pi * f0

Q0 = 600
beta = 2.0

L = 21.4e-9
Z_0 = 50
Ngaps = 4

Ceq = 1 / (w0**2 * L)
Cgap = Ngaps * Ceq

Lseg = L / Ngaps

Rs_total = w0 * L / Q0
Rseg = Rs_total / Ngaps

Rpar = Q0 * w0 * L
Rgap = Rpar / Ngaps**2

# A 50 ohm 180-degree hybrid sees the differential load as 100 ohm.
Zdiff_target = 2 * Z_0

# Differential effective coupling estimate
Cc_eff_est = np.sqrt(beta / (Rgap * Zdiff_target * w0**2))

# Two equal coupling capacitors are in series differentially:
# Ceff = Cc_leg / 2
Cc_leg_est = 2 * Cc_eff_est

print(f"Ceq          = {Ceq:.4e} F = {Ceq * 1e12:.3f} pF")
print(f"Cgap         = {Cgap:.4e} F = {Cgap * 1e12:.3f} pF")
print(f"Lseg         = {Lseg:.4e} H = {Lseg * 1e9:.3f} nH")
print(f"Rseg         = {Rseg:.4f} ohm")
print(f"Rpar         = {Rpar:.2f} ohm")
print(f"Rgap         = {Rgap:.2f} ohm")
print(f"Zdiff target = {Zdiff_target:.1f} ohm")
print(f"Cc_eff_est   = {Cc_eff_est:.4e} F = {Cc_eff_est * 1e15:.2f} fF")
print(f"Cc_leg_est   = {Cc_leg_est:.4e} F = {Cc_leg_est * 1e15:.2f} fF per side")


# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

freq = rf.Frequency(start=350, stop=750, unit='MHz', npoints=5001)
media = rf.media.DefinedGammaZ0(freq, z0=Z_0)


# ------------------------------------------------------------
# Ideal 180-degree hybrid / balun
# ------------------------------------------------------------

def ideal_180_hybrid(freq, z0=50, name='ideal_180_hybrid'):
    """
    Ideal lossless matched 4-port 180-degree hybrid.

    Port definition:
        0: single-ended input
        1: balanced output +
        2: balanced output -
        3: isolated port

    Port 3 must be terminated in z0.
    """

    nfreq = len(freq)
    s = np.zeros((nfreq, 4, 4), dtype=complex)

    a = 1 / np.sqrt(2)

    Smat = a * np.array([
        [0,  1, -1,  0],
        [1,  0,  0,  1],
        [-1, 0,  0,  1],
        [0,  1,  1,  0],
    ], dtype=complex)

    s[:] = Smat

    return rf.Network(
        frequency=freq,
        s=s,
        z0=z0 * np.ones((nfreq, 4)),
        name=name
    )


# ------------------------------------------------------------
# Balanced 4-gap LGR
# ------------------------------------------------------------

def make_balanced_4_gap_lgr(Cc_leg):
    """
    Floating balanced 4-gap LGR driven through an ideal 180-degree hybrid.

    The resonator itself has no ground connection.
    """

    port1 = Circuit.Port(freq, 'port1', z0=Z_0)
    gnd = Circuit.Ground(freq, name='gnd')

    balun = ideal_180_hybrid(freq, z0=Z_0, name='balun')

    # Isolated-port termination
    Riso = media.resistor(Z_0, name='Riso')

    # Coupling capacitors
    Cc_plus = media.capacitor(Cc_leg, name='Cc_plus')
    Cc_minus = media.capacitor(Cc_leg, name='Cc_minus')

    # Gap capacitors
    Cg1 = media.capacitor(Cgap, name='Cg1')
    Cg2 = media.capacitor(Cgap, name='Cg2')
    Cg3 = media.capacitor(Cgap, name='Cg3')
    Cg4 = media.capacitor(Cgap, name='Cg4')

    # Distributed loop resistance
    R1 = media.resistor(Rseg, name='R1')
    R2 = media.resistor(Rseg, name='R2')
    R3 = media.resistor(Rseg, name='R3')
    R4 = media.resistor(Rseg, name='R4')

    # Distributed loop inductance
    L1 = media.inductor(Lseg, name='L1')
    L2 = media.inductor(Lseg, name='L2')
    L3 = media.inductor(Lseg, name='L3')
    L4 = media.inductor(Lseg, name='L4')

    cnx = [
        # Single-ended input into hybrid
        [(port1, 0), (balun, 0)],

        # Isolated port terminated in 50 ohm
        [(balun, 3), (Riso, 0)],
        [(Riso, 1), (gnd, 0)],

        # Hybrid balanced outputs
        [(balun, 1), (Cc_plus, 0)],
        [(balun, 2), (Cc_minus, 0)],

        # Floating LGR node A
        [(Cc_plus, 1), (Cg1, 0), (L4, 1)],

        # Floating LGR node B
        [(Cc_minus, 1), (Cg1, 1), (R1, 0)],

        # Segment 1
        [(R1, 1), (L1, 0)],
        [(L1, 1), (Cg2, 0)],

        # Gap 2 and segment 2
        [(Cg2, 1), (R2, 0)],
        [(R2, 1), (L2, 0)],
        [(L2, 1), (Cg3, 0)],

        # Gap 3 and segment 3
        [(Cg3, 1), (R3, 0)],
        [(R3, 1), (L3, 0)],
        [(L3, 1), (Cg4, 0)],

        # Gap 4 and segment 4, closing the floating loop
        [(Cg4, 1), (R4, 0)],
        [(R4, 1), (L4, 0)],
    ]

    circuit = Circuit(cnx)

    return circuit


# ------------------------------------------------------------
# First simulation using analytical estimate
# ------------------------------------------------------------

network_est = make_balanced_4_gap_lgr(Cc_leg_est).network

s11_est_db = 20 * np.log10(np.abs(network_est.s[:, 0, 0]))
idx_est = np.argmin(s11_est_db)

print()
print("Using analytical Cc estimate:")
print(f"Minimum S11 = {s11_est_db[idx_est]:.2f} dB")
print(f"Frequency   = {freq.f[idx_est] / 1e6:.3f} MHz")


# ------------------------------------------------------------
# Numerical tuning of Cc_leg
# ------------------------------------------------------------

Cc_sweep = Cc_leg_est * np.linspace(0.1, 5.0, 250)

best_Cc = None
best_s11 = np.inf
best_freq = None
best_circuit = None
best_network = None

if False:
    from tqdm import tqdm
    for Cc_test in tqdm(Cc_sweep):
        circ = make_balanced_4_gap_lgr(Cc_test)

        s11_db = 20 * np.log10(np.abs(circ.network.s[:, 0, 0]))
        idx = np.argmin(s11_db)

        if s11_db[idx] < best_s11:
            best_s11 = s11_db[idx]
            best_freq = freq.f[idx]
            best_Cc = Cc_test
            best_network = circ.network
            best_circuit = circ
else:
    best_Cc = Cc_eff_est * 2
    best_circuit = make_balanced_4_gap_lgr(best_Cc)
    best_network = best_circuit.network
    best_freq = best_network.frequency.f[np.argmin(20 * np.log10(np.abs(best_network.s[:, 0, 0])))]
    best_s11 = 20 * np.log10(np.abs(best_network.s[:, 0, 0]))[np.argmin(20 * np.log10(np.abs(best_network.s[:, 0, 0])))]
    # best_Cc = 1.3494e-12
    # best_freq = 497.920e6
    # best_s11 = -36.27
    # best_circuit = make_balanced_4_gap_lgr(best_Cc)
    # best_network = best_circuit.network

best_circuit.plot_graph(
    port_labels=True,
    network_labels=True,
    edge_labels=True,
    port_fontsize=5,
    network_fontsize=5,
    edge_fontsize=5,
)

print()
print("After numerical tuning:")
print(f"Best Cc_leg  = {best_Cc:.4e} F = {best_Cc * 1e15:.2f} fF per side")
print(f"Minimum S11  = {best_s11:.2f} dB")
print(f"Frequency    = {best_freq / 1e6:.3f} MHz")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

z11 = best_network.z[:, 0, 0]

plt.figure()
plt.plot(freq.f / 1e6, np.real(z11), label="Re(Zin)")
plt.plot(freq.f / 1e6, np.imag(z11), label="Im(Zin)")
plt.axhline(50, linestyle="--", label="50 ohm")
plt.axhline(0, linestyle=":")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Impedance [ohm]")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(freq.f / 1e6, s11_est_db, '--', lw=2,
         label=f"estimate: Cc_leg = {Cc_leg_est * 1e15:.1f} fF")

s11_best_db = 20 * np.log10(np.abs(best_network.s[:, 0, 0]))
plt.plot(freq.f / 1e6, s11_best_db, lw=2,
         label=f"tuned: Cc_leg = {best_Cc * 1e15:.1f} fF")

plt.xlabel("Frequency [MHz]")
plt.ylabel("S11 [dB]")
plt.title("Balanced 4-gap LGR driven through ideal 180° hybrid")
plt.grid(True)
plt.legend()
plt.show()