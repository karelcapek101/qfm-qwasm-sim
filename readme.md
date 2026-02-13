# QFC Simulator: Quansistor Field Computing (2026 Prior Art)

Quantum-like computing framework inspired by QFM/QVM stack (Books I–V, 101research.group). Deterministic operator-based execution on classical HW.

## Features
- **Iterace 1**: Prime-shift operátory (A_p) + interference.
- **Iterace 2**: Full SMRK Hamiltonian (self-adjoint, aritmetický spektrum).
- Hilbert space: ℓ²(ℕ, 1/n) s konformní váhou.
- Quantum-like: Interference bez qubitů, replay-ready.

## Iterace 3: Trace Formula (Riemann Implikace)
- Regulovaná trace SMRK: Smooth + oscillatory terms.
- Demo: s=0.5+14j → Smooth -0.142, Osc 22.590, Trace 73.383.
- Graf: Re(Trace) vs. Im(s) – oscilace z primes (RH-like).
## Iterace 4: QWASM Parser (Verifiable Operator Layer)
- Text-based parser: shift p=2, hamiltonian alpha=1.0 beta=0.5, trace s=0.5+14j.
- Demo: Norma po shift=0.9713, po H=2.1470; Trace: smooth=-0.1252, osc=22.5904.
- Audit: Ukládá instructions pro replay.

Příklad QWASM:

## Instalace
```bash
pip install numpy scipy sympy matplotlib  # (už v REPL env)
python qfm_smrk.py  # Spusť demo
