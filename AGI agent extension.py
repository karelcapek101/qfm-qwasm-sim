import numpy as np
from scipy.linalg import eig
import sympy as sp
import random
import matplotlib.pyplot as plt

# [QFM_Simulator class z předchozího příkladu – vlož celou]

# Extended AGI Agent
class QFC_AGI_Agent:
    def __init__(self, qfm_sim, max_steps=20):
        self.qfm = qfm_sim
        self.max_steps = max_steps
        self.trajectory = []
        self.invariants = {'dynamical_var': 0.0}
    
    def decide_action(self):
        actions = ['shift', 'hamiltonian', 'trace_eval']
        action = random.choice(actions)
        if action == 'shift':
            p = random.choice([2,3,5,7,11])
            reward = self.qfm.prime_shift(p)
            rh_reward = self.qfm.trace_reward()
            act_desc = f"shift p={p}"
        elif action == 'hamiltonian':
            alpha = np.random.uniform(0.5, 1.2)
            beta = np.random.uniform(0.3, 0.7)
            reward = self.qfm.hamiltonian_step(alpha, beta)
            rh_reward = self.qfm.trace_reward()
            act_desc = f"hamiltonian α={alpha:.2f} β={beta:.2f}"
        else:  # trace_eval
            rh_reward = self.qfm.trace_reward()
            act_desc = "trace_eval (RH check)"
            reward = rh_reward
        self.trajectory.append({'step': len(self.trajectory), 'action': act_desc, 'rh_reward': rh_reward, 'norma': reward})
        self.invariants['dynamical_var'] = np.var(np.abs(self.qfm.state)**2)
        return act_desc, rh_reward
    
    def run_episode(self):
        for step in range(self.max_steps):
            act, rh_r = self.decide_action()
            print(f"Step {step+1}: {act} → RH Reward {rh_r:.4f}, Var {self.invariants['dynamical_var']:.4f}")
            if self.invariants['dynamical_var'] > 0.5:
                print("Invariant violated: Cooldown!")
                break
        return self.trajectory
    
    def plot_trajectory(self, traj):
        steps = [t['step'] for t in traj]
        rh_rewards = [t['rh_reward'] for t in traj]
        plt.figure(figsize=(10,6))
        plt.plot(steps, rh_rewards, 'o-', label='RH Reward (-Im Dev)')
        plt.axhline(y=0, color='r', linestyle='--', label='RH Ideal (Im=0)')
        plt.xlabel('Step')
        plt.ylabel('RH Reward')
        plt.title('AGI Agent Trajectory: Konvergence k RH Stabilitě')
        plt.legend()
        plt.grid(True)
        plt.show()

# Demo: RH-Focused Episode
sim = QFM_Simulator(max_n=128)
agent = QFC_AGI_Agent(sim, max_steps=20)
traj = agent.run_episode()
agent.plot_trajectory(traj)
print("Final RH Reward:", traj[-1]['rh_reward'] if traj else "Halted")
print("Converged Im Dev:", -traj[-1]['rh_reward'])
