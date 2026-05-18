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
beta = 1.0

L = 21.4e-9
Z_0 = 50
Ngaps = 4

# Full equivalent capacitance of the LGR mode
Ceq = 1 / (w0**2 * L)

# Four identical gap capacitors in series:
# Ceq = Cgap / Ngaps
Cgap = Ngaps * Ceq

# Split total loop inductance into four equal segments
Lseg = L / Ngaps

# Convert Q0 to total series resistance, then split it
Rs_total = w0 * L / Q0
Rseg = Rs_total / Ngaps

# Equivalent full-resonator parallel resistance, useful for coupling estimate
Rpar = Q0 * w0 * L

# Effective resistance seen across one physical gap
Rgap = Rpar / Ngaps**2

# Coupling capacitor estimate for coupling to one gap
Cc_est = np.sqrt(beta / (Rgap * Z_0 * w0**2))

print(f"Ceq      = {Ceq:.4e} F = {Ceq * 1e12:.3f} pF")
print(f"Cgap     = {Cgap:.4e} F = {Cgap * 1e12:.3f} pF")
print(f"Lseg     = {Lseg:.4e} H = {Lseg * 1e9:.3f} nH")
print(f"Rs_total = {Rs_total:.4f} ohm")
print(f"Rseg     = {Rseg:.4f} ohm")
print(f"Rpar     = {Rpar:.2f} ohm")
print(f"Rgap     = {Rgap:.2f} ohm")
print(f"Cc est   = {Cc_est:.4e} F = {Cc_est * 1e15:.2f} fF")


# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

freq = rf.Frequency(start=430, stop=540, unit='MHz', npoints=3001)
media = rf.media.DefinedGammaZ0(freq, z0=Z_0)


# ------------------------------------------------------------
# 4-gap LGR with distributed series loss
# ------------------------------------------------------------

def make_4_gap_lgr_distributed(Cc):
    """
    4-gap LGR model with distributed segment losses.

    Topology, schematically:

        n0 -- Cg1 -- n1 -- R1 -- L1 -- n2 -- Cg2 -- n3 -- R2 -- L2
         |                                                              |
         +-- L4 -- R4 -- n7 -- Cg4 -- n6 -- L3 -- R3 -- n5 -- Cg3 -----+

    The port is capacitively coupled to node n1, across physical gap Cg1.
    Node n0 is the resonator reference node and is tied to circuit ground.
    """

    port1 = Circuit.Port(freq, 'port1', z0=Z_0)
    gnd = Circuit.Ground(freq, name='gnd')

    Ccouple = media.capacitor(Cc, name='Ccouple')

    Cg1 = media.capacitor(Cgap, name='Cg1')
    Cg2 = media.capacitor(Cgap, name='Cg2')
    Cg3 = media.capacitor(Cgap, name='Cg3')
    Cg4 = media.capacitor(Cgap, name='Cg4')

    R1 = media.resistor(Rseg, name='R1')
    R2 = media.resistor(Rseg, name='R2')
    R3 = media.resistor(Rseg, name='R3')
    R4 = media.resistor(Rseg, name='R4')

    L1 = media.inductor(Lseg, name='L1')
    L2 = media.inductor(Lseg, name='L2')
    L3 = media.inductor(Lseg, name='L3')
    L4 = media.inductor(Lseg, name='L4')

    cnx = [
        # Port coupling capacitor
        [(port1, 0), (Ccouple, 0)],

        # Reference side of first gap
        [(gnd, 0), (Cg1, 0), (L4, 1)],

        # Coupled side of first gap
        [(Ccouple, 1), (Cg1, 1), (R1, 0)],

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

        # Gap 4 and segment 4 closing the loop
        [(Cg4, 1), (R4, 0)],
        [(R4, 1), (L4, 0)],
    ]

    circuit = Circuit(cnx)
    return circuit


# ------------------------------------------------------------
# Simulate
# ------------------------------------------------------------

circuit = make_4_gap_lgr_distributed(Cc_est)
network = circuit.network

s11_db = 20 * np.log10(np.abs(network.s[:, 0, 0]))
idx = np.argmin(s11_db)

print()
print("Using analytical Cc estimate:")
print(f"Minimum S11 = {s11_db[idx]:.2f} dB")
print(f"Frequency   = {freq.f[idx] / 1e6:.3f} MHz")

circuit.plot_graph(
    port_labels=True,
    network_labels=True,
    edge_labels=True,
    port_fontsize=5,
    network_fontsize=5,
    edge_fontsize=5,
)

plt.figure()
network.plot_s_db(m=0, n=0, lw=2, label=f"Cc = {Cc_est * 1e15:.1f} fF")
plt.grid(True)
plt.title("4-gap LGR with distributed segment loss")
plt.legend()
plt.show()