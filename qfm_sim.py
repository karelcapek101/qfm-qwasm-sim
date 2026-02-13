import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

class QFM_Simulator:
    def __init__(self, max_n=64):
        self.N = max_n
        self.basis = np.arange(1, self.N + 1, dtype=float)
        self.weights = 1.0 / self.basis
    
    def prime_shift(self, p):
        """A_p: |n> → |p·n> (unitární aproximace)"""
        op = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            j = (i + 1) * p - 1
            if j < self.N:
                op[j, i] = np.sqrt(self.weights[i] / self.weights[j])
        return op
    
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
    sim = QFM_Simulator(max_n=128)
    A2 = sim.prime_shift(2)
    evals = sim.spectrum(A2)
    state, interfered, norm_after = sim.interfere(A2)
    
    print(f"Norma po A₂: {norm_after:.6f} (blízko 1 → dobrá aproximace)")
    print(f"Největší |eigenvalue|: {np.abs(evals[-1]):.4f}")
    
    # Graf spektra (pro zábavu)
    plt.plot(np.abs(evals), 'o-', label='|λ| A₂')
    plt.xlabel('Index eigenvalue')
    plt.ylabel('|λ|')
    plt.title('Spektrum prime-shift operátoru A₂')
    plt.legend()
    plt.grid(True)
    plt.show()
