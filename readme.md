# QFC Simulator: Quansistor Field Computing (2026 Prior Art)

Quantum-like computing framework inspired by QFM/QVM stack (Books I–V, 101research.group). Deterministic operator-based execution on classical HW.

## Features
- **Iterace 1**: Prime-shift operátory (A_p) + interference.
- **Iterace 2**: Full SMRK Hamiltonian (self-adjoint, aritmetický spektrum).
- Hilbert space: ℓ²(ℕ, 1/n) s konformní váhou.
- Quantum-like: Interference bez qubitů, replay-ready.

## Instalace
```bash
pip install numpy scipy sympy matplotlib  # (už v REPL env)
python qfm_smrk.py  # Spusť demo
