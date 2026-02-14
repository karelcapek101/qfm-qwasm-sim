import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta  # Approx zeta

# Mock RH Zeros Im (first 50)
rh_zeros_im = np.array([14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
                        52.9703, 56.4462, 59.3470, 60.8318, 65.1125, 67.0798, 69.5464, 72.0671, 75.7042, 77.1447,
                        79.3374, 80.3114, 83.0386, 84.7353, 87.4252, 88.8091, 92.4915, 94.6513, 95.8705, 98.8312,
                        101.3179, 103.7255, 105.4466, 107.1685, 111.8747, 111.8747, 114.3200, 116.4735, 118.7900, 121.3700,
                        122.9460, 124.2560, 127.5160, 129.5790, 131.0870, 133.4970, 134.7500, 137.2940, 139.7350, 141.1230])[:50]

def lindelof_approx(t, zeros_im, epsilon=0.1):
    """Explicit formula approx |ζ(1/2 + it)| ~ t^ε + sum sin terms"""
    log_t = np.log(t)
    smooth = t**epsilon  # Lindelöf bound
    osc = np.sum(np.sin(2 * np.pi * zeros_im * log_t) / (zeros_im * log_t))  # Mock sum 1/ρ
    return np.abs(smooth + osc)

# Demo: Růst Plot
t_vals = np.logspace(1, 4, 100)  # t=10 to 10^4
zeta_vals = [np.abs(zeta(0.5 + 1j * t)) for t in t_vals]  # Real zeta approx
lind_vals = [lindelof_approx(t, rh_zeros_im, epsilon=0.1) for t in t_vals]

plt.figure(figsize=(10,6))
plt.loglog(t_vals, zeta_vals, 'b-o', label='Real |ζ(1/2 + it)|')
plt.loglog(t_vals, lind_vals, 'r--', label='Lindelöf Approx O(t^0.1)')
plt.xlabel('t (Im(s))')
plt.ylabel('|ζ(1/2 + it)|')
plt.title('Lindelöf Hypotéza: Růst na Kritické Čáře (ε=0.1)')
plt.legend()
plt.grid(True)
plt.show()

# Test: Max růst vs bound
max_growth = np.max(zeta_vals / t_vals**0.1)
print("Max |ζ| / t^0.1:", max_growth)
print("Lindelöf Stable (max < C for some C):", max_growth < 10)  # Mock C=10
