import numpy as np
from scipy.linalg import eig
import sympy as sp
import matplotlib.pyplot as plt

def von_mangoldt(n):
    if n == 1:
        return 0.0
    factors = sp.factorint(n)
    if len(factors) == 1 and list(factors.values())[0] >= 1:
        p = list(factors.keys())[0]
        return np.log(p)
    return 0.0

class QFM_Simulator:
    def __init__(self, max_n=128):
        self.N = max_n
        self.basis = np.arange(1, self.N + 1, dtype=float)
        self.weights = 1.0 / self.basis
        self.primes = [p for p in range(2, self.N+1) if sp.isprime(p)]
    
    def smrk_hamiltonian(self, alpha=1.0, beta=0.5):
        H = np.zeros((self.N, self.N), dtype=complex)
        for p in self.primes[:20]:  # More primes for better osc
            if p > self.N:
                break
            # Backward shift for kinetic (divisibility)
            for i in range(self.N):
                n = i + 1
                if n % p == 0:
                    j = int(n / p) - 1
                    if j >= 0:
                        H[i, j] = 1.0 / np.sqrt(p) * np.sqrt(self.weights[j] / self.weights[i])
        for i in range(self.N):
            n = i + 1
            H[i, i] += alpha * von_mangoldt(n) + beta * np.log(n)
        return H
    
    def spectrum_and_zeros(self, H, tol_re=0.1, tol_im=1e-3):
        evals, _ = eig(H)
        # RH zeros approx: Re~0.5, Im arbitrary but test deviation from 0
        zeros_approx = [e for e in evals if abs(np.real(e) - 0.5) < tol_re]
        im_dev = np.mean([abs(np.imag(z)) for z in zeros_approx]) if zeros_approx else np.inf
        return evals, zeros_approx, im_dev
    
    def trace_formula(self, evals, s=0.5 + 14j, reg=1e-6):
        smooth = np.sum(1.0 / (evals + s)**2)
        osc = sum(np.log(p) * np.exp(-2j * np.pi * np.imag(s) * np.log(p) / np.log(p)) for p in self.primes[:30])
        trace_reg = np.trace(H) + reg * np.sum(np.abs(evals))  # H from init
        return smooth.real, osc.real, trace_reg.real

# Demo: RH Zeros Detection
sim = QFM_Simulator(max_n=128)
H = sim.smrk_hamiltonian(alpha=1.0, beta=0.5)
evals, zeros, im_dev = sim.spectrum_and_zeros(H)
smooth, osc, trace_reg = sim.trace_formula(evals)

print("První 5 Re(evals):", np.sort(np.real(evals))[:5])
print("Aprox Zeros (Re~0.5):", len(zeros), "Im Dev:", im_dev)
print("Trace (s=0.5+14i): Smooth", smooth, "Osc", osc, "Reg", trace_reg)

# Vizu: Zeros Plot
plt.figure(figsize=(8,6))
re_parts = np.real(evals)
im_parts = np.imag(evals)
plt.scatter(re_parts, im_parts, s=5, alpha=0.6, label='Eigenvalues')
plt.axvline(x=0.5, color='r', linestyle='--', label='Critical Line Re=0.5')
plt.scatter([z.real for z in zeros], [z.imag for z in zeros], color='green', s=20, label='Approx Zeros')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')
plt.title('RH Implikace: Spektrum SMRK a Zeros na Kritické Čáře')
plt.legend()
plt.grid(True)
plt.show()
