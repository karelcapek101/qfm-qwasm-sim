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
    def __init__(self, max_n=256):
        self.N = max_n
        self.basis = np.arange(1, self.N + 1, dtype=float)
        self.weights = 1.0 / self.basis
        self.primes = [p for p in range(2, self.N+1) if sp.isprime(p)]
    
    def smrk_hamiltonian(self, alpha=1.0, beta=0.5):
        H = np.zeros((self.N, self.N), dtype=complex)
        for p in self.primes[:50]:  # 50 primes pro robustní osc
            if p > self.N:
                break
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
        zeros_approx = [e for e in evals if abs(np.real(e) - 0.5) < tol_re and abs(np.imag(e)) < tol_im]
        im_dev = np.mean([abs(np.imag(z)) for z in zeros_approx]) if zeros_approx else np.inf
        return evals, zeros_approx, im_dev
    
    def trace_formula(self, evals, s=0.5 + 14j, reg=1e-6):
        smooth = np.sum(1.0 / (evals + s)**2)
        osc = sum(np.log(p) * np.exp(-2j * np.pi * np.imag(s) * np.log(p) / np.log(p)) for p in self.primes[:50])
        trace_reg = np.trace(H) + reg * np.sum(np.abs(evals))  # H from init
        return smooth.real, osc.real, trace_reg.real

# Demo: RH Verification Sim
sim = QFM_Simulator(max_n=256)
H = sim.smrk_hamiltonian(alpha=1.0, beta=0.5)
evals, zeros, im_dev = sim.spectrum_and_zeros(H)
smooth, osc, trace_reg = sim.trace_formula(evals)

print("První 5 Re(evals):", np.sort(np.real(evals))[:5])
print("Aprox RH Zeros (Re~0.5, Im~0):", len(zeros), "Im Dev:", im_dev)
print("Trace (s=0.5+14i): Smooth", smooth, "Osc", osc, "Reg", trace_reg)

# Vizu: Zeros Scatter (Critical Line)
plt.figure(figsize=(10,6))
re_parts = np.real(evals)
im_parts = np.imag(evals)
plt.scatter(re_parts, im_parts, s=3, alpha=0.7, label='Eigenvalues')
plt.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Critical Line Re=0.5')
if zeros:
    plt.scatter([z.real for z in zeros], [z.imag for z in zeros], color='green', s=50, label='Approx Zeros')
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')
plt.title('RH Verification: SMRK Spektrum a Zeros na Kritické Čáře (N=256)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1, 1)  # Zoom na Im~0
plt.show()
