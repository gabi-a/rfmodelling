import numpy as np
from matplotlib import pyplot as plt

f0 = 500e6
w0 = 2 * np.pi * f0
Q0 = 600
L = 21.4e-9
Ngaps = 4

Ceq = 1 / (w0**2 * L)
Cgap = Ngaps * Ceq

Rpar = Q0 * w0 * L
Rgap = Rpar / Ngaps**2

Rs = w0 * L / Q0

freq = np.linspace(430e6, 540e6, 3001)
w = 2 * np.pi * freq

Zc_gap = 1 / (1j * w * Cgap)

# Exact one-gap impedance of the distributed loop
Z_rest = Rs + 1j * w * L + 3 * Zc_gap
Z_gap_exact = 1 / (1 / Zc_gap + 1 / Z_rest)

# Approximate one-gap parallel RLC model
# Use Cgap as the physical gap capacitance.
# Choose Lgap so that Lgap || Cgap resonates at f0.
Lgap = 1 / (w0**2 * Cgap)

Z_Lgap = 1j * w * Lgap
Z_Cgap = 1 / (1j * w * Cgap)

Z_gap_rlc = 1 / (
    1 / Rgap +
    1 / Z_Lgap +
    1 / Z_Cgap
)

plt.figure()
plt.plot(freq / 1e6, np.real(Z_gap_exact), label="Full loop: Re(Zgap)")
plt.plot(freq / 1e6, np.real(Z_gap_rlc), "--", label="RLC approx: Re(Zgap)")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Resistance [ohm]")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(freq / 1e6, np.imag(Z_gap_exact), label="Full loop: Im(Zgap)")
plt.plot(freq / 1e6, np.imag(Z_gap_rlc), "--", label="RLC approx: Im(Zgap)")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Reactance [ohm]")
plt.grid(True)
plt.legend()
plt.show()