import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps  # For smooth part

# Mock Gutzwiller: d(E) = bar d + sum orbits
def gutzwiller_trace(E, evals, primes, hbar=0.1):
    # Smooth: Volume / (2π ħ)^f (f=2D mock)
    bar_d = E / (2 * np.pi * hbar)**2
    # Oscillating: Sum A_γ e^{i S_γ / ħ - μ/2} (primes as orbits, S= log p * E)
    osc = 0.0
    for p in primes[:20]:
        S_gamma = np.log(p) * E
        A_gamma = 1.0 / np.sqrt(np.abs(p))  # Stability mock
        mu = 0  # Maslov index
        osc += A_gamma * np.exp(1j * S_gamma / hbar - mu / 2)
    return bar_d + np.real(osc)

# QFM Evals (from SMRK)
def qfm_evals(N=256):
    log_n = np.log(np.arange(1, N+1))
    H_diag = log_n + np.random.normal(0, 0.1, N)
    return np.sort(H_diag)

# Demo: Trace vs E
E_vals = np.linspace(1, 10, 100)
evals = qfm_evals(256)
primes = [p for p in range(2, 50) if sp.isprime(p)]

traces = [gutzwiller_trace(E, evals, primes, hbar=0.1) for E in E_vals]

plt.figure(figsize=(10,6))
plt.plot(E_vals, traces, 'b-', label='Gutzwiller Trace Approx')
plt.axhline(y=np.mean(traces), color='r', linestyle='--', label='Smooth bar d(E)')
plt.xlabel('Energy E')
plt.ylabel('Density of States d(E)')
plt.title('Gutzwiller Trace Formula v QFC: Spectral + Primes Orbits')
plt.legend()
plt.grid(True)
plt.show()

# Test: Oscillating amplitude ~ O(ħ^{-1})
hbar_vals = np.logspace(-2, 0, 5)
amps = [np.std([gutzwiller_trace(5, evals, primes, h) for _ in range(10)]) for h in hbar_vals]
print("hbar vs Osc Amp:", list(zip(hbar_vals, amps)))
print("Semiclassical Growth O(1/ħ):", np.allclose(amps, 1 / hbar_vals, rtol=0.2))
