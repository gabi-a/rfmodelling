import numpy as np
from scipy.optimize import minimize, differential_evolution
import skrf as rf
from skrf.circuit import Circuit
import matplotlib.pyplot as plt
# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

f0 = 500e6
w0 = 2 * np.pi * f0

Q0 = 600
beta = 1.0

L = 21.4e-9
Z0 = 50
Ngaps = 4

Ceq = 1 / (w0**2 * L)
Cgap = Ngaps * Ceq

Lseg = L / Ngaps
Rs_total = w0 * L / Q0
Rseg = Rs_total / Ngaps

Rpar = Q0 * w0 * L
Rgap = Rpar / Ngaps**2

Zdiff_target = 2 * Z0

Cc_eff_est = np.sqrt(beta / (Rgap * Zdiff_target * w0**2))
Cc_leg_fixed = 2 * Cc_eff_est

print(f"Cgap         = {Cgap * 1e12:.3f} pF")
print(f"Lseg         = {Lseg * 1e9:.3f} nH")
print(f"Rseg         = {Rseg:.4f} ohm")
print(f"Cc_leg_fixed = {Cc_leg_fixed * 1e15:.2f} fF per side")

# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

freq = rf.Frequency(start=430, stop=570, unit='MHz', npoints=3001)
media = rf.media.DefinedGammaZ0(freq, z0=Z0)

# ------------------------------------------------------------
# Fast analytic floating-loop impedance
# ------------------------------------------------------------

def z_gap_loop(omega):
    """
    Differential impedance across one physical gap of the floating 4-gap loop.

    The driven gap is Cgap in parallel with the rest of the loop:
        Cgap || (4 series segments? no: rest of loop has 3 gaps + 4 segments)

    Around the other side of the loop from node A to node B:
        segment + gap + segment + gap + segment + gap + segment

    So:
        Zgap = Z_Cgap || Z_rest
    """

    Zc = 1 / (1j * omega * Cgap)
    Zseg = Rseg + 1j * omega * Lseg

    Zrest = 4 * Zseg + 3 * Zc
    Zgap = 1 / (1 / Zc + 1 / Zrest)

    return Zgap


def element_impedance(kind, value, omega):
    if kind == "C":
        return 1 / (1j * omega * value)
    elif kind == "L":
        return 1j * omega * value
    else:
        raise ValueError("kind must be 'C' or 'L'")


def diff_load_impedance(series_kind, series_value, shunt_kind, shunt_value, omega):
    """
    Differential impedance at the balanced output of the hybrid.

    Topology:
        balun+ -- Xs/2? -- node P -- Cc -- LGR -- Cc -- node M -- Xs/2? -- balun-

    In the skrf model we put one equal series element in each leg.
    Differentially, those two leg elements add in series:
        Zseries_total = 2 * Zseries_leg

    Fixed coupling capacitors also add:
        Zcc_total = 2 * Zcc_leg

    Shunt element is across the differential pair after the series elements,
    so it is in parallel with:
        2*Zcc_leg + Zgap
    """

    Zs_leg = element_impedance(series_kind, series_value, omega)
    Zsh = element_impedance(shunt_kind, shunt_value, omega)

    Zcc_leg = 1 / (1j * omega * Cc_leg_fixed)
    Zgap = z_gap_loop(omega)

    Zload_after_series = 2 * Zcc_leg + Zgap
    Zafter = 1 / (1 / Zsh + 1 / Zload_after_series)

    Zdiff = 2 * Zs_leg + Zafter

    return Zdiff


def s11_from_zdiff(Zdiff):
    """
    A 50 ohm ideal 180 degree hybrid corresponds to a 100 ohm
    differential load target.

    The single-ended reflection coefficient is equivalent to:
        Gamma = (Zdiff - 2*Z0) / (Zdiff + 2*Z0)
    """
    return (Zdiff - 2 * Z0) / (Zdiff + 2 * Z0)


def objective(x, series_kind, shunt_kind):
    series_value = 10 ** x[0]
    shunt_value = 10 ** x[1]

    Zdiff = diff_load_impedance(
        series_kind, series_value,
        shunt_kind, shunt_value,
        w0,
    )

    gamma = s11_from_zdiff(Zdiff)

    # Minimize |S11|^2 directly
    return abs(gamma) ** 2


def bounds_for_kind(kind):
    if kind == "C":
        return (np.log10(0.05e-12), np.log10(100e-12))
    else:
        return (np.log10(0.5e-9), np.log10(1000e-9))


def format_value(kind, value):
    if kind == "C":
        return f"{value * 1e12:.4f} pF"
    else:
        return f"{value * 1e9:.4f} nH"


# ------------------------------------------------------------
# Try all four topologies quickly
# ------------------------------------------------------------

topologies = [
    ("C", "C"),
    ("C", "L"),
    ("L", "C"),
    ("L", "L"),
]

results = []

for series_kind, shunt_kind in topologies:
    bounds = [
        bounds_for_kind(series_kind),
        bounds_for_kind(shunt_kind),
    ]

    # Fast coarse global search
    result_de = differential_evolution(
        objective,
        bounds=bounds,
        args=(series_kind, shunt_kind),
        maxiter=80,
        popsize=10,
        polish=False,
        seed=1,
        workers=1,
    )

    # Fast local polish
    result = minimize(
        objective,
        result_de.x,
        args=(series_kind, shunt_kind),
        method="Nelder-Mead",
        options={
            "maxiter": 1000,
            "xatol": 1e-10,
            "fatol": 1e-14,
        },
    )

    series_value = 10 ** result.x[0]
    shunt_value = 10 ** result.x[1]

    Zdiff = diff_load_impedance(
        series_kind, series_value,
        shunt_kind, shunt_value,
        w0,
    )
    gamma = s11_from_zdiff(Zdiff)
    s11_db = 20 * np.log10(abs(gamma))

    results.append({
        "series_kind": series_kind,
        "shunt_kind": shunt_kind,
        "series_value": series_value,
        "shunt_value": shunt_value,
        "Zdiff": Zdiff,
        "Zin_equiv": Zdiff / 2,
        "S11_db": s11_db,
        "objective": abs(gamma) ** 2,
    })


results = sorted(results, key=lambda r: r["objective"])

print()
print("Fast analytic topology results at 500 MHz:")
print()

for r in results:
    print(f"Series {r['series_kind']} + shunt {r['shunt_kind']}")
    print(f"  series value = {format_value(r['series_kind'], r['series_value'])} per leg")
    print(f"  shunt value  = {format_value(r['shunt_kind'], r['shunt_value'])} differential")
    print(f"  Zdiff        = {r['Zdiff'].real:.2f} + j{r['Zdiff'].imag:.2f} ohm")
    print(f"  Zin equiv    = {r['Zin_equiv'].real:.2f} + j{r['Zin_equiv'].imag:.2f} ohm")
    print(f"  S11          = {r['S11_db']:.2f} dB")
    print()

best = results[0]

print("Best:")
print(f"  Series {best['series_kind']} + shunt {best['shunt_kind']}")
print(f"  series value = {format_value(best['series_kind'], best['series_value'])} per leg")
print(f"  shunt value  = {format_value(best['shunt_kind'], best['shunt_value'])} differential")

series_kind = best["series_kind"]
shunt_kind = best["shunt_kind"]
series_value = best["series_value"]
shunt_value = best["shunt_value"]

from utils import make_balanced_tuned_lgr
circuit_best = make_balanced_tuned_lgr(
    media=media,
    freq=freq,
    series_kind=series_kind,
    series_value=series_value,
    shunt_kind=shunt_kind,
    shunt_value=shunt_value,
    Cc_leg_fixed=Cc_leg_fixed,
    Cgap=Cgap,
    Rseg=Rseg,
    Lseg=Lseg,
    Z0=Z0,
)
network_best = circuit_best.network

idx_f0 = np.argmin(np.abs(freq.f - f0))

Zin_f0 = network_best.z[idx_f0, 0, 0]
S11_f0_db = 20 * np.log10(abs(network_best.s[idx_f0, 0, 0]))

print()
print("Full skrf verification:")
print(f"Topology     = series {series_kind} + shunt {shunt_kind}")
print(f"Zin @ 500MHz = {Zin_f0.real:.2f} + j{Zin_f0.imag:.2f} ohm")
print(f"S11 @ 500MHz = {S11_f0_db:.2f} dB")

# ------------------------------------------------------------
# Plot the network topology
# ------------------------------------------------------------

circuit_best.plot_graph(
    port_labels=True,
    network_labels=True,
    edge_labels=True,
    port_fontsize=10,
    network_fontsize=10,
    edge_fontsize=5,
)

# ------------------------------------------------------------
# Plot best S11 and input impedance
# ------------------------------------------------------------

s11_best_db = 20 * np.log10(np.abs(network_best.s[:, 0, 0]))
Zin_best = network_best.z[:, 0, 0]

plt.figure()
plt.plot(freq.f / 1e6, s11_best_db, lw=2)
plt.axvline(f0 / 1e6, linestyle='--', label='500 MHz')
plt.xlabel("Frequency [MHz]")
plt.ylabel("S11 [dB]")
plt.title(
    f"Best match: series {best['series_kind']} + shunt {best['shunt_kind']}"
)
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(freq.f / 1e6, np.real(Zin_best), label="Re(Zin)")
plt.plot(freq.f / 1e6, np.imag(Zin_best), label="Im(Zin)")
plt.axvline(f0 / 1e6, linestyle='--', label='500 MHz')
plt.axhline(50, linestyle=':')
plt.axhline(0, linestyle=':')
plt.xlabel("Frequency [MHz]")
plt.ylabel("Input impedance [ohm]")
plt.title("Single-ended input impedance after balun")
plt.grid(True)
plt.legend()
plt.show()