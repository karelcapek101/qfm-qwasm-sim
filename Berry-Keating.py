import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Berry-Keating H = x p approx (symmetric finite diff, x>0)
N = 128
log_x = np.linspace(0, 3, N)  # log x for L2(dx/x)
x = np.exp(log_x)
dx = x[1] - x[0]

# Finite diff p = -i d/dx approx (central, self-adjoint)
d = np.diag(np.ones(N-1), -1) - np.diag(np.ones(N-1), 1)
p = -1j * d / (2 * dx)
p = (p + p.conj().T) / 2  # Hermitian part

H = np.diag(x) @ p  # x p

# Eigenvalues (focus real for RH Im approx)
evals, _ = eig(H)
evals_real = np.sort(np.real(evals))

# Known first RH zeros Im
rh_zeros_im = np.array([14.1347, 21.0220, 25.0109, 30.4249, 32.9351])[:5]

# Spacing
spacings = np.diff(evals_real[:10])

# Montgomery R2 simplified (no full integral)
r = np.linspace(0.01, 2, 100)  # Avoid r=0 div
r2 = 1 - (np.sin(np.pi * r) / (np.pi * r))**2

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(range(len(evals_real)), evals_real, 'b-', label='Berry-Keating Evals (Re approx Im)')
ax[0].scatter(range(5), rh_zeros_im, color='r', s=50, label='Known RH Zeros Im')
ax[0].set_xlabel('Index k')
ax[0].set_ylabel('Value')
ax[0].set_title('Berry-Keating Spektrum vs RH Zeros')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(r, r2, 'g-', label='Montgomery R2(r)')
ax[1].hist(spacings / np.mean(spacings), bins=10, density=True, alpha=0.7, label='Empirical Spacing')
ax[1].set_xlabel('Normalized Spacing')
ax[1].set_ylabel('Density / R2')
ax[1].set_title('Montgomery Korelace v Berry-Keating')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig('berry_keating_sim.png')
plt.show()

# Stats
mean_spacing = np.mean(spacings)
T = evals_real[-1]
pred = np.log(T) / (2 * np.pi)
match_ratio = mean_spacing / pred
print("První 5 Evals (Re approx Im):", evals_real[:5])
print("Known RH Im:", rh_zeros_im)
print("Mean Spacing:", mean_spacing)
print("Pred log T / 2π:", pred)
print("Match Ratio:", match_ratio)
print("Plot uložen jako 'berry_keating_sim.png'")
