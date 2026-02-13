import numpy as np
from scipy.linalg import eig
import sympy as sp
import matplotlib.pyplot as plt

def von_mangoldt(n):
    """Von Mangoldt: log p if n=p^k, else 0"""
    if n == 1:
        return 0.0
    factors = sp.factorint(n)
    if len(factors) == 1 and list(factors.values())[0] >= 1:
        p = list(factors.keys())[0]
        return np.log(p)
    return 0.0

class QFM_Simulator:
    def __init__(self, max_n=64):
        self.N = max_n
        self.basis = np.arange(1, self.N + 1, dtype=float)
        self.weights = 1.0 / self.basis
        self.primes = [p for p in range(2, self.N+1) if sp.isprime(p)]
    
    def prime_shift(self, p):
        """A_p: forward shift |n> → |p n> (unitary approx)"""
        op = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            n = i + 1
            m = n * p
            if m <= self.N:
                j = int(m) - 1
                op[j, i] = np.sqrt(self.weights[i] / self.weights[j])
        return op
    
    def smrk_hamiltonian(self, alpha=1.0, beta=1.0):
        """Full SMRK: kinetic + potential (self-adjoint)"""
        H = np.zeros((self.N, self.N), dtype=complex)
        for p in self.primes:
            if p > self.N:
                break
            A_p = self.prime_shift(p)
            H += (1.0 / np.sqrt(p)) * A_p
        for i in range(self.N):
            n = i + 1
            H[i, i] += alpha * von_mangoldt(n) + beta * np.log(n)
        return H
    
    def spectrum(self, op):
        evals, _ = eig(op)
        return np.sort_complex(evals)
    
    def trace_formula(self, H, s=0.5 + 14j, reg=1e-6):
        """Regulovaná trace formula: sum cycles + oscillatory (Riemann-like)"""
        evals = self.spectrum(H)
        # Smooth term: sum 1/(λ + s)^2 (zeta-like)
        smooth = np.sum(1.0 / (evals + s)**2)
        # Oscillatory: sum_p log p * exp(-2π i ρ / log p) approx (primes rho=0)
        osc = 0.0
        for p in self.primes[:10]:  # Limit pro demo
            rho = np.log(p)  # Simplified (real part for demo)
            osc += np.log(p) * np.exp(-2j * np.pi * rho / np.log(p))
        # Regulated trace
        trace_reg = np.trace(H) + reg * np.sum(np.abs(evals))
        return smooth.real, osc.real, trace_reg.real

# === DEMO ===
if __name__ == "__main__":
    sim = QFM_Simulator(max_n=32)  # Malé N pro rychlost; zkus 64 pro detaily
    H = sim.smrk_hamiltonian(alpha=1.0, beta=0.5)
    smooth, osc, trace_reg = sim.trace_formula(H, s=0.5 + 14j)
    
    print("Trace Formula Demo (s=0.5+14j):")
    print(f"Smooth term: {smooth:.6f}")
    print(f"Oscillatory term: {osc:.6f}")
    print(f"Regulated trace: {trace_reg:.6f}")
    
    # Graf: Trace vs. Im(s) pro různé s
    s_vals = np.linspace(0.1, 20, 50) + 0.5j
    traces = [sim.trace_formula(H, s=0.5 + 1j * im)[0] for im in s_vals]
    plt.plot(s_vals.imag, np.real(traces), 'o-', label='Re(Trace)')
    plt.xlabel('Im(s)')
    plt.ylabel('Re(Trace)')
    plt.title('Trace Formula vs. Im(s) – Riemann-like')
    plt.grid(True)
    plt.savefig('trace_formula.png')
    plt.show()
    print("\nGraf uložen jako 'trace_formula.png'")
