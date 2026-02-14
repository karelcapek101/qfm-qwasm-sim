import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap

# Montgomery Match (from prev)
def montgomery_match(evals):
    spacings = np.diff(evals[:10])
    mean_sp = np.mean(spacings)
    T = evals[-1]
    pred = np.log(T) / (2 * np.pi)
    match_ratio = mean_sp / pred if pred != 0 else np.inf
    return match_ratio, mean_sp, pred

# Berry-Keating Evals (from prev)
def berry_keating_evals(N=64):
    log_x = np.linspace(0, 3, N)
    x = np.exp(log_x)
    dx = x[1] - x[0]
    d = np.diag(np.ones(N-1), -1) - np.diag(np.ones(N-1), 1)
    p = -1j * d / (2 * dx)
    p = (p + p.conj().T) / 2
    H = np.diag(x) @ p
    evals, _ = eig(H)
    return np.sort(np.real(evals))

# Expanded Grok-4 Spectral RL
class Grok4_Spectral_RL:
    def __init__(self, N=64, num_actions=5, learning_rate=0.01, discount=0.95):
        self.N = N
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount
        self.value_func = np.random.rand(N)
        self.policy = np.ones(N) / N
        self.q_table = np.random.rand(N, num_actions)
        self.primes = [2, 3, 5, 7, 11][:num_actions]
        self.berry_evals = berry_keating_evals(N)  # Chaos field
        self.target_osc = 22.59  # RH osc stable
    
    def qfm_shift_action(self, state_idx, action_idx):
        p = self.primes[action_idx]
        log_scale = np.log(np.arange(1, self.num_actions + 1))
        self.q_table[state_idx, :] *= log_scale * np.sqrt(1.0 / p)
        return np.linalg.norm(self.q_table[state_idx])
    
    def spectral_update(self, state, action, reward, next_state):
        evals = self.berry_evals  # Full N chaos evals
        q_current = self.q_table[state, action]
        q_next = np.max(self.q_table[next_state])
        target = reward + self.gamma * q_next * np.mean(evals)
        self.q_table[state, action] += self.lr * (target - q_current)
        self.value_func[state] = np.max(self.q_table[state])
        rh_stable = np.mean(np.abs(np.imag(evals))) < 1e-3
        return rh_stable
    
    def rlhf_update(self, reward=0.92):
        evals = self.berry_evals
        self.policy = np.exp(-evals) / np.sum(np.exp(-evals))
        self.value_func += reward * evals
        return np.mean(self.policy)
    
    def chaos_match_reward(self, reward):
        match, _, _ = montgomery_match(self.berry_evals)
        rh_bonus = 0.5 if abs(reward - self.target_osc) < 1.0 else 0.0
        return -abs(match - 1.0) + rh_bonus  # Chaos + RH
    
    def run_episode(self, steps=10, env_reward_fn=lambda s: np.random.uniform(0.9, 0.95)):
        trajectory = []
        state = 0
        for step in range(steps):
            action = np.argmax(self.q_table[state])
            next_state = (state + action) % self.N
            base_reward = env_reward_fn(state)
            chaos_reward = self.chaos_match_reward(base_reward)
            norma = self.qfm_shift_action(state, action)
            rh_stable = self.spectral_update(state, action, chaos_reward, next_state)
            policy_mean = self.rlhf_update(chaos_reward)
            trajectory.append({'step': step, 'state': state, 'action': action, 'base_reward': base_reward, 'chaos_reward': chaos_reward, 'rh_stable': rh_stable, 'norma': norma})
            state = next_state
            print(f"Step {step+1}: State {state}, Action {action}, Base Reward {base_reward:.4f}, Chaos Reward {chaos_reward:.4f}, RH Stable {rh_stable}, Norma {norma:.4f}")
        return trajectory
    
    def plot_q_table(self, traj):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(self.q_table, cmap='viridis', annot=False)
        plt.title('Spectral Q-Table Heatmap')
        plt.xlabel('Action')
        plt.ylabel('State')
        
        plt.subplot(1, 2, 2)
        steps = [t['step'] for t in traj]
        chaos_rewards = [t['chaos_reward'] for t in traj]
        plt.plot(steps, chaos_rewards, 'o-', label='Chaos Rewards')
        plt.axhline(y=0, color='r', linestyle='--', label='Ideal Match=1.0')
        plt.xlabel('Step')
        plt.ylabel('Chaos Reward')
        plt.title('Episode Chaos Trajectory')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Demo: Chaos Spectral RL Episode
agent = Grok4_Spectral_RL(N=64, num_actions=5, learning_rate=0.01)
traj = agent.run_episode(steps=10)
agent.plot_q_table(traj)
print("Final Policy Mean:", np.mean(agent.policy))
print("Chaos Episodes:", len(traj))
print("Avg Chaos Reward:", np.mean([t['chaos_reward'] for t in traj]))
