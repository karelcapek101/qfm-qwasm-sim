import numpy as np
import matplotlib.pyplot as plt

def montgomery_r2(r):
    """Montgomery R₂(r): Pair correlation"""
    if r == 0:
        return 1.0  # Self-correlation δ(r)
    sin_term = np.sin(np.pi * r) / (np.pi * r)
    integral_approx = np.trapz(np.sin(np.pi * (r - u)) / (np.pi * (r - u)) * np.sin(np.pi * u) / (np.pi * u), u=np.linspace(0, r, 100))
    return 1 - sin_term**2 + integral_approx

def mock_zeros_from_smrk(N=1000):
    """Mock zeros: Im(ρ_k) ≈ (k + 1/2) log(k) / π + noise (RH approx)"""
    k = np.arange(1, N+1)
    zeros_im = (k + 0.5) * np.log(k) / np.pi + np.random.normal(0, 0.1, N)  # GUE noise
    return np.sort(zeros_im)

# Demo: Korelace Plot
zeros = mock_zeros_from_smrk(N=500)
spacings = np.diff(zeros)  # Spacing γ_{k+1} - γ_k
r_vals = np.linspace(0, 2, 100)
r2_vals = [montgomery_r2(r) for r in r_vals]

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.hist(spacings, bins=50, density=True, alpha=0.7, label='Spacing Distribution')
plt.xlabel('Spacing Δγ')
plt.ylabel('Density')
plt.title('Montgomery Spacing (GUE-like Repulze)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(r_vals, r2_vals, 'b-', label='R₂(r)')
plt.scatter(spacings[:50] / np.mean(spacings), np.ones(50)*0.5, alpha=0.5, s=10, label='Empirical Pairs')
plt.xlabel('Normalized r')
plt.ylabel('R₂(r)')
plt.title('Montgomery Pair Korelace (RH na Re=1/2)')
plt.axvline(x=0, color='r', linestyle='--', label='Repulze r→0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# RH Test: Mean Spacing ~ log T / 2π
T = zeros[-1]
mean_spacing = np.mean(spacings)
rh_pred = np.log(T) / (2 * np.pi)
print(f"Empirical Mean Spacing: {mean_spacing:.4f}")
print(f"Montgomery Pred (log T / 2π): {rh_pred:.4f}")
print("Match Ratio:", mean_spacing / rh_pred)
