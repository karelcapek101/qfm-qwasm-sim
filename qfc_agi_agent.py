import numpy as np
from scipy.linalg import eig
import sympy as sp
import random

# QFM Core (from prev iterations – simplified)
class QFM_Simulator:
    def __init__(self, max_n=64):
        self.N = max_n
        self.state = np.random.randn(self.N) + 1j * np.random.randn(self.N)
        self.state /= np.linalg.norm(self.state)
    
    def prime_shift(self, p):
        # Mock shift: Multiply amplitudes by sqrt(1/p)
        self.state *= np.sqrt(1.0 / p)
        return np.linalg.norm(self.state)
    
    def hamiltonian_step(self, alpha=1.0):
        # Mock H: Add potential
        potential = alpha * np.log(np.arange(1, self.N+1))
        self.state *= potential
        return np.linalg.norm(self.state)
    
    def trace_reward(self, target_osc=22.59):
        # Mock osc from primes
        primes = [2,3,5,7,11][:5]
        osc = sum(np.log(p) for p in primes)
        return -abs(osc - target_osc)  # RH stability reward

# AGI Agent: Operator-first decisions
class QFC_AGI_Agent:
    def __init__(self, qfm_sim, max_steps=10):
        self.qfm = qfm_sim
        self.max_steps = max_steps
        self.trajectory = []
        self.invariants = {'dynamical_var': 0.0}
    
    def decide_action(self):
        # Quantum-like: Random shift p or hamiltonian α (interference choice)
        if random.random() < 0.5:
            p = random.choice([2,3,5])
            reward = self.qfm.prime_shift(p)
            action = f"shift p={p}"
        else:
            alpha = np.random.uniform(0.5, 1.0)
            reward = self.qfm.hamiltonian_step(alpha)
            action = f"hamiltonian α={alpha:.2f}"
        osc_reward = self.qfm.trace_reward()
        self.trajectory.append({'step': len(self.trajectory), 'action': action, 'reward': osc_reward})
        # Mock var for invariant
        self.invariants['dynamical_var'] = np.var(np.abs(self.qfm.state)**2)
        return action, osc_reward
    
    def run_episode(self):
        for step in range(self.max_steps):
            action, reward = self.decide_action()
            print(f"Step {step+1}: {action} → Osc Reward {reward:.4f}, Var {self.invariants['dynamical_var']:.4f}")
            if self.invariants['dynamical_var'] > 0.5:  # Governance kill
                print("Invariant violated: Cooldown!")
                break
        print("Agent Trajectory:", self.trajectory)
        return self.trajectory

# === DEMO ===
if __name__ == "__main__":
    sim = QFM_Simulator(max_n=64)
    agent = QFC_AGI_Agent(sim, max_steps=10)
    traj = agent.run_episode()
    print("Converged Osc:", traj[-1]['reward'] if traj else "Halted")
