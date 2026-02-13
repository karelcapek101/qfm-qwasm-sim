import numpy as np
from scipy.linalg import eig
import sympy as sp
import matplotlib.pyplot as plt

def von_mangoldt(n):
    """Von Mangoldt: log p if n=p^k, else 0"""
    if n == 1:
        return 0.0
    factors = sp.factorint(n)
    if len(factors) == 1 and list(factors.values())[0] >= 1:  # p^k pro k>=1
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
        
        # Kinetic: sum_p 1/sqrt(p) A_p (p|n implicit in shift)
        for p in self.primes:
            if p > self.N:
                break
            A_p = self.prime_shift(p)
            H += (1.0 / np.sqrt(p)) * A_p
        
        # Potential: alpha Λ(n) + beta log n (diagonal)
        for i in range(self.N):
            n = i + 1
            H[i, i] += alpha * von_mangoldt(n) + beta * np.log(n)
        
        return H
    
    def spectrum(self, op):
        evals, _ = eig(op)
        return np.sort_complex(evals)
    
    def interfere(self, op, state=None):
        if state is None:
            state = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            state /= np.linalg.norm(state)
        interfered = op @ state
        return state, interfered, np.linalg.norm(interfered)

# === DEMO ===
if __name__ == "__main__":
    sim = QFM_Simulator(max_n=64)  # Zkus větší N pro lepší spektrum
    H = sim.smrk_hamiltonian(alpha=1.0, beta=0.5)  # Tune params jako v QFM
    evals = sim.spectrum(H)
    state, interfered, norm_after = sim.interfere(H)
    
    print("SMRK Hamiltonian (prvních 5x5):")
    print(H[:5, :5])
    print(f"\nSpektrum Re(λ) první 5: {np.real(evals[:5])}")
    print(f"Norma po H: {norm_after:.6f}")
    print(f"Max |λ|: {np.max(np.abs(evals)):.4f}")
    
    # Graf
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(np.real(evals), 'o-', label='Re(λ)')
    plt.xlabel('Index eigenvalue')
    plt.ylabel('Re(λ)')
    plt.title('Re spektrum SMRK Hamiltonian')
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(np.imag(evals), 's-', label='Im(λ)')
    plt.xlabel('Index')
    plt.ylabel('Im(λ)')
    plt.title('Im spektrum (očekávaně ~0)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('smrk_spectrum.png')
    plt.show()
    print("\nGraf uložen jako 'smrk_spectrum.png'")
