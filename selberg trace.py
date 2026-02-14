import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid approx for heat

# Mock Selberg: Spectral + Geometric
def selberg_trace(t, evals, primes):
    # Spectral: sum e^{-t λ_j} + 1/(4t)
    spectral = np.sum(np.exp(-t * evals)) + 1/(4*t)
    # Geometric: sum ℓ(γ) / sqrt(4π t) e^{-ℓ^2 / 4t} (primes as lengths)
    geometric = np.sum([np.log(p) / np.sqrt(4*np.pi*t) * np.exp(-(np.log(p))**2 / (4*t)) for p in primes])
    return spectral + geometric

# QFM Evals (from SMRK)
def qfm_evals(N=256):
    log_n = np.log(np.arange(1, N+1))
    H_diag = log_n + np.random.normal(0, 0.1, N)  # Mock λ_j ~ log n
    return np.sort(H_diag)

# Demo: Trace Plot
t_vals = np.logspace(-2, 1, 50)  # t=0.01 to 10
evals = qfm_evals(256)
primes = [p for p in range(2, 100) if sp.isprime(p)]  # 25 primes

traces = [selberg_trace(t, evals, primes) for t in t_vals]

plt.figure(figsize=(10,6))
plt.semilogx(t_vals, traces, 'b-o', label='Selberg Trace Approx')
plt.axhline(y=1/(4*np.mean(t_vals)), color='r', linestyle='--', label='Continuous 1/(4t)')
plt.xlabel('t (Heat Parameter)')
plt.ylabel('Trace')
plt.title('Selberg Trace Formula v QFC: Spectral + Geometric Primes')
plt.legend()
plt.grid(True)
plt.show()

# Test: Asymptotika O(t^ε)
growth = np.log(traces) / np.log(t_vals)
print("Průměrný Růst ε:", np.mean(growth))
print("Lindelöf-like Stable (ε <0.1):", np.mean(growth) < 0.1)
