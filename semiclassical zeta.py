import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma  # Approx log zeta

# Mock Semiclassical Zeta: Product over orbits (primes)
def semiclassical_zeta(s, primes, hbar=0.1):
    log_z = 0.0j
    for p in primes[:20]:
        l_gamma = np.log(p)  # Délka orbit
        S_gamma = l_gamma * np.real(s)  # Action approx
        A_gamma = 1.0 / np.sqrt(p)  # Amplitude
        z_gamma = np.exp(-s * l_gamma + 1j * S_gamma / hbar)
        log_z += np.log(1 - z_gamma)  # -log(1 - z) approx
    return np.exp(log_z)

# QFM Evals (for spectral side)
def qfm_evals(N=256):
    log_n = np.log(np.arange(1, N+1))
    return np.sort(log_n + np.random.normal(0, 0.1, N))

# Demo: Log |Z(s)| vs Re(s)
s_real = np.linspace(0.4, 0.6, 20)  # Around critical line
s = s_real + 1j * 14.0  # Im=14
primes = [p for p in range(2, 100) if sp.isprime(p)]

log_abs_z = [np.log(np.abs(semiclassical_zeta(si, primes, hbar=0.1))) for si in s]

plt.figure(figsize=(10,6))
plt.plot(s_real, log_abs_z, 'b-o', label='Log |Z(s)| (Semiclassical Zeta)')
plt.axvline(x=0.5, color='r', linestyle='--', label='Critical Line Re=0.5')
plt.xlabel('Re(s)')
plt.ylabel('Log |Z(s)|')
plt.title('Semiclassical Zeta v QFC: Orbit Product okolo Kritické Čáry')
plt.legend()
plt.grid(True)
plt.show()

# Test: Růst s 1/ħ
hbar_vals = np.logspace(-2, 0, 5)
growth = [np.mean([np.log(np.abs(semiclassical_zeta(0.5 + 1j*14, primes, h)) for _ in range(3)]) for h in hbar_vals]
print("hbar vs Log |Z| Growth:", list(zip(hbar_vals, growth)))
print("Semiclassical O(1/ħ):", np.allclose(growth, -np.log(hbar_vals), rtol=0.2))
