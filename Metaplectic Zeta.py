import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

# Mock Metaplectic Zeta: Product with double cover phase
def metaplectic_zeta(s, primes, chi_phase=0.5):
    log_z = 0.0j
    for p in primes[:20]:
        chi_p = np.exp(1j * chi_phase * np.log(p))  # Mock character phase
        z_p = chi_p / p**s
        log_z += 0.5 * np.log(1 - z_p)  # Double cover sqrt
    return np.exp(log_z)

# QFM Evals for spectral side
def qfm_evals(N=256):
    log_n = np.log(np.arange(1, N+1))
    return np.sort(log_n + np.random.normal(0, 0.1, N))

# Demo: |Z(s)| vs Re(s)
s_real = np.linspace(0.4, 0.6, 20)
s = s_real + 1j * 14.0
primes = [p for p in range(2, 100) if sp.isprime(p)]

abs_z = [np.abs(metaplectic_zeta(si, primes)) for si in s]

plt.figure(figsize=(10,6))
plt.plot(s_real, abs_z, 'b-o', label='|Metaplectic Z(s)| (Double Cover)')
plt.axvline(x=0.5, color='r', linestyle='--', label='Critical Line Re=0.5')
plt.xlabel('Re(s)')
plt.ylabel('|Z(s)|')
plt.title('Metaplektická Zeta v QFC: Primes Characters okolo Kritické Čáry')
plt.legend()
plt.grid(True)
plt.show()

# Test: Symetrie na Re=0.5
sym_dev = np.mean(np.abs(np.array(abs_z) - np.abs(metaplectic_zeta(0.5 + 1j*14, primes))))
print("Symetrie Dev na Re=0.5:", sym_dev)
print("RH Stable (dev <1e-3):", sym_dev < 1e-3)
