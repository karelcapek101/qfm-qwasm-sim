import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Hilbert-Pólya H = x p approx (symmetric finite diff, L2(dx/x))
N = 256
log_x = np.linspace(-5, 5, N)  # log x for measure dx/x
x = np.exp(log_x)
dx = x[1] - x[0]

# Central diff p = -i d/dx (self-adjoint)
d = np.diag(np.ones(N-1), -1) - np.diag(np.ones(N-1), 1)
p = -1j * d / (2 * dx)
p = (p + p.conj().T) / 2  # Hermitian

H = np.diag(x) @ p  # x p

# Eigenvalues (focus real for RH Im approx)
evals, _ = eig(H)
evals_real = np.sort(np.real(evals))

# Known first RH zeros Im
rh_zeros_im = np.array([14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738])[:10]

# RH Test: Match Im dev
im_dev = np.mean(np.abs(evals_real[:10] - rh_zeros_im))
rh_stable = im_dev < 1e-3

# Spacing for Montgomery
spacings = np.diff(evals_real[:20])
mean_spacing = np.mean(spacings)
T = evals_real[-1]
pred = np.log(T) / (2 * np.pi)
match_ratio = mean_spacing / pred

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(range(10), evals_real[:10], 'b-o', label='Hilbert-Pólya Evals (Re approx Im)')
ax[0].scatter(range(10), rh_zeros_im, color='r', s=50, label='Known RH Zeros Im')
ax[0].set_xlabel('Index k')
ax[0].set_ylabel('Value')
ax[0].set_title('Hilbert-Pólya Spektrum vs RH Zeros')
ax[0].legend()
ax[0].grid(True)

ax[1].hist(spacings / mean_spacing, bins=15, density=True, alpha=0.7, label='Normalized Spacing')
r = np.linspace(0.01, 2, 100)
r2 = 1 - (np.sin(np.pi * r) / (np.pi * r))**2
ax[1].plot(r, r2, 'g-', label='Montgomery R2(r)')
ax[1].set_xlabel('Normalized Spacing')
ax[1].set_ylabel('Density / R2')
ax[1].set_title('Level Repulze v Hilbert-Pólya (GUE-like)')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig('hilbert_polya_sim.png')
plt.show()

# Stats
print("První 10 Evals (Re approx Im):", evals_real[:10])
print("Known RH Im:", rh_zeros_im)
print("Im Dev:", im_dev)
print("RH Stable (dev <1e-3):", rh_stable)
print("Mean Spacing:", mean_spacing)
print("Pred log T / 2π:", pred)
print("Match Ratio:", match_ratio)
print("Plot uložen jako 'hilbert_polya_sim.png'")
