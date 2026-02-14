import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import math
from scipy.linalg import expm
import hashlib  # Pro EY30 commitment

# [Předchozí funkce: von_mangoldt, primes, diag, H_sparse konstrukce – zkopírováno z minulého]

# Parametry
N = 500
alpha, beta = 1.0, 1.0
t = 0.1

# [Konstrukce H_sparse a H_dense – zkopírováno]

# Výpočet exp(-t H)
exp_minus_tH = expm(-t * H_dense)

# Trace
trace_exp = np.trace(exp_minus_tH)

# Eigenvalues pro summary
evals = eigsh(H_sparse, k=5, which='SM', return_eigenvectors=False)

# EY30-like Export: Spectral Summary jako commitment
spectral_summary = {
    'evals_top5': evals.tolist(),
    'trace_t0.1': float(trace_exp),
    'N': N,
    'params': {'alpha': alpha, 'beta': beta, 't': t}
}
summary_str = str(spectral_summary)
commitment = hashlib.sha256(summary_str.encode()).hexdigest()[:16]  # Short hash pro demo

# Výstup
print(f"Trace of exp(-{t} H): {trace_exp}")
print(f"Shape of exp(-t H): {exp_minus_tH.shape}")
print(f"First 5 eigenvalues of H: {evals}")
print(f"EY30-like Commitment: {commitment}")
