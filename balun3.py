import numpy as np
from scipy.optimize import minimize, differential_evolution
import skrf as rf

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

f0 = 500e6
w0 = 2 * np.pi * f0

Q0 = 600

# This is now the TARGET final loaded beta
beta = 2.0

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

print(f"Ceq          = {Ceq * 1e12:.3f} pF")
print(f"Cgap         = {Cgap * 1e12:.3f} pF")
print(f"Lseg         = {Lseg * 1e9:.3f} nH")
print(f"Rseg         = {Rseg:.4f} ohm")
print(f"Rpar         = {Rpar:.2f} ohm")
print(f"Rgap         = {Rgap:.2f} ohm")
print(f"Target beta  = {beta:.3f}")


# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

freq = rf.Frequency(start=430, stop=570, unit='MHz', npoints=3001)
media = rf.media.DefinedGammaZ0(freq, z0=Z0)
idx_f0 = np.argmin(np.abs(freq.f - f0))


# ------------------------------------------------------------
# Fast analytic floating-loop impedance
# ------------------------------------------------------------

def z_gap_loop(omega, Cgap, Rseg, Lseg):
    """
    Differential impedance across the driven physical gap of the floating
    4-gap resonator.

    Driven gap branch:
        Cgap

    Alternate path around the rest of the loop:
        4 series segments + 3 other gap capacitors
    """

    Zc = 1 / (1j * omega * Cgap)
    Zseg = Rseg + 1j * omega * Lseg

    Zrest = 4 * Zseg + 3 * Zc

    return 1 / (1 / Zc + 1 / Zrest)


def element_impedance(kind, value, omega):
    if kind == "C":
        return 1 / (1j * omega * value)
    elif kind == "L":
        return 1j * omega * value
    else:
        raise ValueError("kind must be 'C' or 'L'")


def external_gap_impedance(
    omega,
    Cc_leg,
    series_kind,
    series_value,
    shunt_kind,
    shunt_value,
    Z0,
):
    """
    External impedance seen by the LGR driven gap, including:

        two coupling capacitors
        differential shunt tuning element
        two series tuning elements
        ideal 100 ohm differential source termination

    Topology from LGR gap outward:

        gap A -- Cc -- node P -- series -- balun+
                         |
                       shunt
                         |
        gap B -- Cc -- node M -- series -- balun-

    The ideal 50-ohm single-ended hybrid corresponds to a 100-ohm
    differential termination.
    """

    Zcc = 1 / (1j * omega * Cc_leg)

    Zs_leg = element_impedance(series_kind, series_value, omega)
    Zsh = element_impedance(shunt_kind, shunt_value, omega)

    Zdiff_port = 2 * Z0

    # Looking from node P/M toward the source:
    Ztoward_port = 2 * Zs_leg + Zdiff_port

    # Shunt element is across P/M, in parallel with the source path
    Zpm_external = 1 / (1 / Zsh + 1 / Ztoward_port)

    # From the LGR driven gap, add two coupling capacitors
    Zext_gap = 2 * Zcc + Zpm_external

    return Zext_gap


def beta_loaded_from_network(
    omega,
    Cc_leg,
    series_kind,
    series_value,
    shunt_kind,
    shunt_value,
    Cgap,
    Rseg,
    Lseg,
    Z0,
):
    """
    beta_loaded = G_external / G_internal
    """

    Zgap_internal = z_gap_loop(omega, Cgap, Rseg, Lseg)
    Ygap_internal = 1 / Zgap_internal
    G_internal = np.real(Ygap_internal)

    Zext_gap = external_gap_impedance(
        omega=omega,
        Cc_leg=Cc_leg,
        series_kind=series_kind,
        series_value=series_value,
        shunt_kind=shunt_kind,
        shunt_value=shunt_value,
        Z0=Z0,
    )

    Yext_gap = 1 / Zext_gap
    G_external = np.real(Yext_gap)

    beta_loaded = G_external / G_internal

    return beta_loaded, Zgap_internal, Zext_gap, G_internal, G_external


def diff_input_impedance(
    omega,
    Cc_leg,
    series_kind,
    series_value,
    shunt_kind,
    shunt_value,
    Cgap,
    Rseg,
    Lseg,
):
    """
    Differential impedance seen by the balanced output of the hybrid.

    From the balun output:

        series elements
        then shunt element in parallel with:
            two coupling capacitors + LGR gap impedance
    """

    Zs_leg = element_impedance(series_kind, series_value, omega)
    Zsh = element_impedance(shunt_kind, shunt_value, omega)

    Zcc = 1 / (1j * omega * Cc_leg)
    Zgap = z_gap_loop(omega, Cgap, Rseg, Lseg)

    Zload_after_series = 2 * Zcc + Zgap
    Zafter = 1 / (1 / Zsh + 1 / Zload_after_series)

    Zdiff = 2 * Zs_leg + Zafter

    return Zdiff


def s11_from_zdiff(Zdiff, Z0):
    return (Zdiff - 2 * Z0) / (Zdiff + 2 * Z0)


def bounds_for_kind(kind):
    if kind == "C":
        # 0.05 pF to 100 pF
        return (np.log10(0.05e-12), np.log10(100e-12))
    else:
        # 0.5 nH to 1000 nH
        return (np.log10(0.5e-9), np.log10(1000e-9))


def format_value(kind, value):
    if kind == "C":
        return f"{value * 1e12:.4f} pF"
    else:
        return f"{value * 1e9:.4f} nH"


# ------------------------------------------------------------
# Objective: match + target final beta_loaded
# ------------------------------------------------------------

def objective_with_beta(x, series_kind, shunt_kind):
    """
    x[0] = log10(Cc_leg)
    x[1] = log10(series_value)
    x[2] = log10(shunt_value)

    Target:
        Zdiff = 100 + j0 ohm
        beta_loaded = beta
    """

    Cc_leg = 10 ** x[0]
    series_value = 10 ** x[1]
    shunt_value = 10 ** x[2]

    try:
        Zdiff = diff_input_impedance(
            omega=w0,
            Cc_leg=Cc_leg,
            series_kind=series_kind,
            series_value=series_value,
            shunt_kind=shunt_kind,
            shunt_value=shunt_value,
            Cgap=Cgap,
            Rseg=Rseg,
            Lseg=Lseg,
        )

        beta_loaded, *_ = beta_loaded_from_network(
            omega=w0,
            Cc_leg=Cc_leg,
            series_kind=series_kind,
            series_value=series_value,
            shunt_kind=shunt_kind,
            shunt_value=shunt_value,
            Cgap=Cgap,
            Rseg=Rseg,
            Lseg=Lseg,
            Z0=Z0,
        )

        # Match errors
        err_r = (Zdiff.real - 2 * Z0) / (2 * Z0)
        err_x = Zdiff.imag / (2 * Z0)

        # Coupling error
        err_beta = beta_loaded - beta # np.log(beta_loaded / beta)

        # Weight beta enough that the optimizer does not only chase match
        return err_r**2 + err_x**2 + 5 * err_beta**2

    except Exception:
        return 1e12


# ------------------------------------------------------------
# Try all four topologies
# ------------------------------------------------------------

topologies = [
    # ("C", "C"),
    ("C", "L"),
    # ("L", "C"),
    # ("L", "L"),
]

results = []

# Coupling-cap range.
# This is intentionally broad. Tighten it later for manufacturability.
Cc_bounds = (np.log10(0.05e-12), np.log10(20e-12))

for series_kind, shunt_kind in topologies:
    bounds = [
        Cc_bounds,
        bounds_for_kind(series_kind),
        bounds_for_kind(shunt_kind),
    ]

    result_de = differential_evolution(
        objective_with_beta,
        bounds=bounds,
        args=(series_kind, shunt_kind),
        maxiter=120,
        popsize=12,
        polish=False,
        seed=1,
        workers=1,
    )

    result = minimize(
        objective_with_beta,
        result_de.x,
        args=(series_kind, shunt_kind),
        method="Nelder-Mead",
        options={
            "maxiter": 2000,
            "xatol": 1e-11,
            "fatol": 1e-14,
        },
    )

    Cc_leg = 10 ** result.x[0]
    series_value = 10 ** result.x[1]
    shunt_value = 10 ** result.x[2]

    Zdiff = diff_input_impedance(
        omega=w0,
        Cc_leg=Cc_leg,
        series_kind=series_kind,
        series_value=series_value,
        shunt_kind=shunt_kind,
        shunt_value=shunt_value,
        Cgap=Cgap,
        Rseg=Rseg,
        Lseg=Lseg,
    )

    gamma = s11_from_zdiff(Zdiff, Z0)
    s11_db = 20 * np.log10(abs(gamma))

    beta_loaded, Zgap_internal, Zext_gap, Gint, Gext = beta_loaded_from_network(
        omega=w0,
        Cc_leg=Cc_leg,
        series_kind=series_kind,
        series_value=series_value,
        shunt_kind=shunt_kind,
        shunt_value=shunt_value,
        Cgap=Cgap,
        Rseg=Rseg,
        Lseg=Lseg,
        Z0=Z0,
    )

    results.append({
        "series_kind": series_kind,
        "shunt_kind": shunt_kind,
        "Cc_leg": Cc_leg,
        "series_value": series_value,
        "shunt_value": shunt_value,
        "Zdiff": Zdiff,
        "Zin_equiv": Zdiff / 2,
        "S11_db": s11_db,
        "beta_loaded": beta_loaded,
        "Zgap_internal": Zgap_internal,
        "Zext_gap": Zext_gap,
        "G_internal": Gint,
        "G_external": Gext,
        "objective": result.fun,
    })


results = sorted(results, key=lambda r: r["objective"])

print()
print("Topology results at 500 MHz, including beta_loaded constraint:")
print()

for r in results:
    print(f"Series {r['series_kind']} + shunt {r['shunt_kind']}")
    print(f"  Cc_leg       = {r['Cc_leg'] * 1e12:.4f} pF per side")
    print(f"  series value = {format_value(r['series_kind'], r['series_value'])} per leg")
    print(f"  shunt value  = {format_value(r['shunt_kind'], r['shunt_value'])} differential")
    print(f"  Zdiff        = {r['Zdiff'].real:.2f} + j{r['Zdiff'].imag:.2f} ohm")
    print(f"  Zin equiv    = {r['Zin_equiv'].real:.2f} + j{r['Zin_equiv'].imag:.2f} ohm")
    print(f"  S11          = {r['S11_db']:.2f} dB")
    print(f"  beta_loaded  = {r['beta_loaded']:.3f}")
    print(f"  Zext_gap     = {r['Zext_gap'].real:.2f} + j{r['Zext_gap'].imag:.2f} ohm")
    print()

best = results[0]

print("Best:")
print(f"  Series {best['series_kind']} + shunt {best['shunt_kind']}")
print(f"  Cc_leg       = {best['Cc_leg'] * 1e12:.4f} pF per side")
print(f"  series value = {format_value(best['series_kind'], best['series_value'])} per leg")
print(f"  shunt value  = {format_value(best['shunt_kind'], best['shunt_value'])} differential")
print(f"  Zdiff        = {best['Zdiff'].real:.2f} + j{best['Zdiff'].imag:.2f} ohm")
print(f"  S11          = {best['S11_db']:.2f} dB")
print(f"  beta_loaded  = {best['beta_loaded']:.3f}")