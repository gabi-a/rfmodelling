import numpy as np
import skrf as rf
from skrf.circuit import Circuit
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution


# ------------------------------------------------------------
# Resonator parameters
# ------------------------------------------------------------

# f0 = 500e6
# w0 = 2 * np.pi * f0

# Q0 = 600
# beta = 1.0

# L = 21.4e-9
# Z_0 = 50
# Ngaps = 4

# Ceq = 1 / (w0**2 * L)
# Cgap = Ngaps * Ceq

# Lseg = L / Ngaps

# Rs_total = w0 * L / Q0
# Rseg = Rs_total / Ngaps

# Rpar = Q0 * w0 * L
# Rgap = Rpar / Ngaps**2

# Zdiff_target = 2 * Z_0

# # Fixed coupling capacitors
# Cc_eff_est = np.sqrt(beta / (Rgap * Zdiff_target * w0**2))
# Cc_leg_fixed = 2 * Cc_eff_est

# print(f"Ceq          = {Ceq:.4e} F = {Ceq * 1e12:.3f} pF")
# print(f"Cgap         = {Cgap:.4e} F = {Cgap * 1e12:.3f} pF")
# print(f"Lseg         = {Lseg:.4e} H = {Lseg * 1e9:.3f} nH")
# print(f"Rseg         = {Rseg:.4f} ohm")
# print(f"Rpar         = {Rpar:.2f} ohm")
# print(f"Rgap         = {Rgap:.2f} ohm")
# print(f"Cc_leg_fixed = {Cc_leg_fixed:.4e} F = {Cc_leg_fixed * 1e15:.2f} fF per side")


# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

# freq = rf.Frequency(start=430, stop=570, unit='MHz', npoints=3001)
# media = rf.media.DefinedGammaZ0(freq, z0=Z_0)

# f_array = freq.f
# idx_f0 = np.argmin(np.abs(f_array - f0))


# ------------------------------------------------------------
# Ideal 180-degree hybrid / balun
# ------------------------------------------------------------

def ideal_180_hybrid(freq, z0=50, name='ideal_180_hybrid'):
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
# Helpers for tunable elements
# ------------------------------------------------------------

def make_series_element(media, kind, value, name):
    if kind == "C":
        return media.capacitor(value, name=name)
    elif kind == "L":
        return media.inductor(value, name=name)
    else:
        raise ValueError("kind must be 'C' or 'L'")


def make_shunt_element(media, kind, value, name):
    if kind == "C":
        return media.capacitor(value, name=name)
    elif kind == "L":
        return media.inductor(value, name=name)
    else:
        raise ValueError("kind must be 'C' or 'L'")


def format_value(kind, value):
    if kind == "C":
        return f"{value * 1e12:.4f} pF"
    else:
        return f"{value * 1e9:.4f} nH"


# ------------------------------------------------------------
# Balanced LGR with arbitrary series/shunt tuning topology
# ------------------------------------------------------------

def make_balanced_tuned_lgr(media, freq, series_kind, series_value, shunt_kind, shunt_value, Cc_leg_fixed, Cgap, Rseg, Lseg, Z0) -> Circuit:
    """
    Floating balanced 4-gap LGR driven through an ideal 180-degree hybrid.

    Tuning network:
        shunt element is connected across the differential pair.
        one equal series element is inserted in each balanced leg.

    The shunt element is placed after the series elements, closer to the load.
    This is usually the better L-match orientation for a resonator load.
    """

    port1 = Circuit.Port(freq, 'port1', z0=Z0)
    gnd = Circuit.Ground(freq, name='gnd')

    balun = ideal_180_hybrid(freq, z0=Z0, name='balun')

    Riso = media.resistor(Z0, name='Riso')

    Xs_plus = make_series_element(media, series_kind, series_value, f'{series_kind}_plus')
    Xs_minus = make_series_element(media, series_kind, series_value, f'{series_kind}_minus')

    Xsh = make_shunt_element(media, shunt_kind, shunt_value, name=f'{shunt_kind}_shunt')

    Cc_plus = media.capacitor(Cc_leg_fixed, name='Cc_plus')
    Cc_minus = media.capacitor(Cc_leg_fixed, name='Cc_minus')

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
        # Single-ended input into hybrid
        [(port1, 0), (balun, 0)],

        # Isolated port terminated
        [(balun, 3), (Riso, 0)],
        [(Riso, 1), (gnd, 0)],

        # Series tuning elements at balanced output
        [(balun, 1), (Xs_plus, 0)],
        [(balun, 2), (Xs_minus, 0)],

        # Differential shunt tuning element after series elements
        [(Xs_plus, 1), (Xsh, 0), (Cc_plus, 0)],
        [(Xs_minus, 1), (Xsh, 1), (Cc_minus, 0)],

        # Floating LGR driven gap nodes
        [(Cc_plus, 1), (Cg1, 0), (L4, 1)],
        [(Cc_minus, 1), (Cg1, 1), (R1, 0)],

        # Segment 1
        [(R1, 1), (L1, 0)],
        [(L1, 1), (Cg2, 0)],

        # Segment 2
        [(Cg2, 1), (R2, 0)],
        [(R2, 1), (L2, 0)],
        [(L2, 1), (Cg3, 0)],

        # Segment 3
        [(Cg3, 1), (R3, 0)],
        [(R3, 1), (L3, 0)],
        [(L3, 1), (Cg4, 0)],

        # Segment 4, closing floating loop
        [(Cg4, 1), (R4, 0)],
        [(R4, 1), (L4, 0)],
    ]

    return Circuit(cnx)