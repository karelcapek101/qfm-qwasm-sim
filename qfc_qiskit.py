import numpy as np
from scipy.linalg import eig
import sympy as sp
import hashlib

# Mock Qiskit: State vector sim (2 qubits, |00> initial)
def mock_qiskit_circuit(gates=['H0', 'CX01'], shots=1024):
    dim = 4  # 2 qubits
    state = np.array([1.0 + 0j, 0j, 0j, 0j])  # |00>
    
    for gate in gates:
        if gate == 'H0':
            h = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])
            state[:2] = h @ state[:2]  # H on qubit 0
        elif gate == 'CX01':
            # CX: |00>→|00>, |01>→|01>, |10>→|11>, |11>→|10>
            state[2], state[3] = state[3], state[2]  # Swap for control=1
    
    # Measure: Probabilities → counts
    probs = np.abs(state)**2
    counts = {bin(i)[2:].zfill(2): int(p * shots) for i, p in enumerate(probs) if int(p * shots) > 0}
    return state, counts

# QFM Integration: Shift na counts (arithmetic transport)
def qfm_shift_on_counts(counts, p=2):
    shifted = {}
    for state_bin, cnt in counts.items():
        state_int = int(state_bin, 2)
        new_int = (state_int * p) % 4  # Mod 4 for 2 qubits
        new_bin = bin(new_int)[2:].zfill(2)
        shifted[new_bin] = shifted.get(new_bin, 0) + cnt
    return shifted

# SMRK Spectrum na shifted states (mock evals)
def hybrid_spectrum(shifted_counts):
    # Mock states from counts
    states = np.array([np.sqrt(v / sum(shifted_counts.values())) for v in shifted_counts.values()])
    H_mock = np.diag(np.log(np.arange(1, len(states)+1)))  # Approx log n potential
    evals, _ = eig(H_mock)
    return np.sort_complex(evals)

# Trace formula (from prev)
def trace_formula(evals, s=0.5 + 14j, reg=1e-6):
    smooth = np.sum(1.0 / (evals + s)**2)
    primes = [2, 3, 5]  # Mock
    osc = sum(np.log(p) * np.exp(-2j * np.pi * np.log(p) / np.log(p)) for p in primes)
    trace_reg = np.trace(H_mock) + reg * np.sum(np.abs(evals))  # H_mock from above
    return smooth.real, osc.real, trace_reg.real

# === DEMO ===
if __name__ == "__main__":
    state, counts = mock_qiskit_circuit(['H0', 'CX01'])
    print("Qiskit Mock Circuit: Initial |00> → H(0):", state[:2])
    print("After CX(0,1):", state)
    print("Counts (shots=1024):", counts)
    
    shifted = qfm_shift_on_counts(counts, p=2)
    print("\nQFM Integration: Prime-shift p=2 na states")
    print("Shifted Counts:", shifted)
    
    evals = hybrid_spectrum(shifted)
    print("\nHybrid Spektrum: Re(λ)", np.real(evals)[:3])
    
    smooth, osc, trace_reg = trace_formula(evals)
    print("Trace (s=0.5+14j): Smooth", smooth, "Osc", osc, "Reg", trace_reg)
