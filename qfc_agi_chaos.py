import numpy as np
from scipy.linalg import eig
import sympy as sp
import random
import matplotlib.pyplot as plt

# [QFM_Simulator class z předchozích – vlož s smrk_hamiltonian a trace_reward]

# Berry-Keating Mock (finite diff H = x p)
def berry_keating_evals(N=128):
    log_x = np.linspace(0, 3, N)
    x = np.exp(log_x)
    dx = x[1] - x[0]
    # Central diff p = -i d/dx (symmetric)
    d = np.diag(np.ones(N-1), -1) - np.diag(np.ones(N-1), 1)
    p = -1j * d / (2 * dx)
    p = (p + p.conj().T) / 2  # Hermitian
    H = np.diag(x) @ p
    evals, _ = eig(H)
    return np.sort(np.real(evals))  # RH Im approx

# Montgomery Match (spacing vs pred)
def montgomery_match(evals):
    spacings = np.diff(evals[:10])  # First 10
    mean_sp = np.mean(spacings)
    T = evals[-1]
    pred = np.log(T) / (2 * np.pi)
    match_ratio = mean_sp / pred if pred != 0 else np.inf
    return match_ratio, mean_sp, pred

# Extended Chaos AGI Agent
class QFC_AGI_Chaos_Agent:
    def __init__(self, qfm_sim, max_steps=20):
        self.qfm = qfm_sim
        self.max_steps = max_steps
        self.trajectory = []
        self.invariants = {'dynamical_var': 0.0}
        self.berry_evals = berry_keating_evals()  # Chaos field init
    
    def decide_action(self):
        actions = ['shift', 'hamiltonian', 'chaos_eval']
        action = random.choice(actions)
        if action == 'shift':
            p = random.choice([2,3,5,7,11])
            reward = self.qfm.prime_shift(p)
            chaos_reward = self.chaos_match_reward()
            act_desc = f"shift p={p}"
        elif action == 'hamiltonian':
            alpha = np.random.uniform(0.5, 1.2)
            beta = np.random.uniform(0.3, 0.7)
            reward = self.qfm.hamiltonian_step(alpha, beta)
            chaos_reward = self.chaos_match_reward()
            act_desc = f"hamiltonian α={alpha:.2f} β={beta:.2f}"
        else:  # chaos_eval
            chaos_reward = self.chaos_match_reward()
            act_desc = "chaos_eval (Berry-Keating match)"
            reward = chaos_reward
        self.trajectory.append({'step': len(self.trajectory), 'action': act_desc, 'chaos_reward': chaos_reward, 'norma': reward})
        self.invariants['dynamical_var'] = np.var(np.abs(self.qfm.state)**2)
        return act_desc, chaos_reward
    
    def chaos_match_reward(self):
        match, _, _ = montgomery_match(self.berry_evals)
        rh_bonus = 0.5 if abs(self.qfm.trace_reward() - 22.59) < 1.0 else 0.0  # RH osc stable
        return -abs(match - 1.0) + rh_bonus  # Chaos match + RH bonus
    
    def run_episode(self):
        for step in range(self.max_steps):
            act, chaos_r = self.decide_action()
            print(f"Step {step+1}: {act} → Chaos Reward {chaos_r:.4f}, Var {self.invariants['dynamical_var']:.4f}")
            if self.invariants['dynamical_var'] > 0.5:
                print("Invariant violated: Cooldown!")
                break
        return self.trajectory
    
    def plot_trajectory(self, traj):
        steps = [t['step'] for t in traj]
        chaos_rewards = [t['chaos_reward'] for t in traj]
        plt.figure(figsize=(10,6))
        plt.plot(steps, chaos_rewards, 'o-', label='Chaos Match Reward')
        plt.axhline(y=0, color='r', linestyle='--', label='Ideal Match=1.0 (RH Stable)')
        plt.xlabel('Step')
        plt.ylabel('Chaos Reward')
        plt.title('AGI Chaos Trajectory: Konvergence k Berry-Keating RH')
        plt.legend()
        plt.grid(True)
        plt.show()

# Demo: Chaos Episode
sim = QFM_Simulator(max_n=128)
agent = QFC_AGI_Chaos_Agent(sim, max_steps=20)
traj = agent.run_episode()
agent.plot_trajectory(traj)
print("Final Chaos Reward:", traj[-1]['chaos_reward'] if traj else "Halted")
print("Berry Evals Sample:", agent.berry_evals[:5])
