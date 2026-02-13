import numpy as np
from scipy.linalg import eig
import sympy as sp
import re
from collections import defaultdict
import json
import hashlib  # Pro activation ID

def von_mangoldt(n):
    """Von Mangoldt: log p if n=p^k, else 0"""
    if n == 1:
        return 0.0
    factors = sp.factorint(n)
    if len(factors) == 1 and list(factors.values())[0] >= 1:
        p = list(factors.keys())[0]
        return np.log(p)
    return 0.0

class QFM_Simulator:
    def __init__(self, max_n=64):
        self.N = max_n
        self.basis = np.arange(1, self.N + 1, dtype=float)
        self.weights = 1.0 / self.basis
        self.primes = [p for p in range(2, self.N+1) if sp.isprime(p)]
        self.state = np.random.randn(self.N) + 1j * np.random.randn(self.N)
        self.state /= np.linalg.norm(self.state)
    
    def prime_shift(self, p):
        """A_p: forward shift |n> → |p n> (unitary approx)"""
        op = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            n = i + 1
            m = n * p
            if m <= self.N:
                j = int(m) - 1
                op[j, i] = np.sqrt(self.weights[i] / self.weights[j])
        return op
    
    def smrk_hamiltonian(self, alpha=1.0, beta=1.0):
        """Full SMRK: kinetic + potential (self-adjoint)"""
        H = np.zeros((self.N, self.N), dtype=complex)
        for p in self.primes:
            if p > self.N:
                break
            A_p = self.prime_shift(p)
            H += (1.0 / np.sqrt(p)) * A_p
        for i in range(self.N):
            n = i + 1
            H[i, i] += alpha * von_mangoldt(n) + beta * np.log(n)
        return H
    
    def apply_op(self, op_name, params=None):
        if op_name == 'shift':
            p = int(params['p']) if params and 'p' in params else 2
            op = self.prime_shift(p)
            self.state = op @ self.state
            return f"Applied A_{p}: norma = {np.linalg.norm(self.state):.4f}"
        elif op_name == 'hamiltonian':
            alpha = float(params['alpha']) if params and 'alpha' in params else 1.0
            beta = float(params['beta']) if params and 'beta' in params else 1.0
            H = self.smrk_hamiltonian(alpha, beta)
            self.state = H @ self.state
            return f"Applied H (α={alpha}, β={beta}): norma = {np.linalg.norm(self.state):.4f}"
        return "Unknown op"

class QWASM_Parser:
    def __init__(self, simulator):
        self.sim = simulator
        self.instructions = []
    
    def parse(self, code):
        lines = code.strip().split('\n')
        result = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.match(r'(shift|hamiltonian|trace)\s*(.*)', line.lower())
            if match:
                op = match.group(1)
                params_str = match.group(2).strip()
                params = {}
                if op == 'shift':
                    if 'p=' in params_str:
                        params['p'] = params_str.split('p=')[1].split()[0]
                elif op == 'hamiltonian':
                    if 'alpha=' in params_str:
                        params['alpha'] = params_str.split('alpha=')[1].split()[0]
                    if 'beta=' in params_str:
                        params['beta'] = params_str.split('beta=')[1].split()[0]
                elif op == 'trace':
                    if 's=' in params_str:
                        s_part = params_str.split('s=')[1].split()[0]
                        if '+' in s_part:
                            real, imag = s_part.split('+')
                            params['s_real'] = real
                            params['s_imag'] = imag.replace('j', '')
                self.instructions.append((op, params))
                output = self.sim.apply_op(op, params)
                result.append(f"Parsed '{line}': {output}")
            else:
                result.append(f"Parse error: {line}")
        return result

class ChainSimulator:
    def __init__(self):
        self.chains = {
            'ICP': {'role': 'control', 'state': {}, 'traces': []},
            'NEAR': {'role': 'execution', 'state': {}, 'traces': []},
            'Celestia': {'role': 'availability', 'state': {}, 'traces': []},
            'Kaspa': {'role': 'temporal', 'state': {}, 'traces': []},
            'Juno': {'role': 'governance', 'state': {}, 'traces': []}
        }
        self.global_trace = []
        self.juno = JunoDAOSimulator()  # Juno DAO
    
    def execute_on_chain(self, chain, instruction):
        if chain not in self.chains:
            return f"Unknown chain: {chain}"
        ch = self.chains[chain]
        if chain == 'NEAR':
            sim = QFM_Simulator(max_n=32)
            parser = QWASM_Parser(sim)
            outputs = parser.parse(instruction)
            ch['state']['output'] = outputs
            ch['traces'].append({'step': len(ch['traces']), 'output': outputs[-1]})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"NEAR exec: {outputs[-1]}"
        elif chain == 'ICP':
            quorum = np.random.choice([True, False], p=[0.9, 0.1])
            ch['state']['decision'] = bool(quorum)
            ch['traces'].append({'step': len(ch['traces']), 'quorum': bool(quorum)})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"ICP quorum: {'Accepted' if quorum else 'Rejected'}"
        elif chain == 'Celestia':
            trace_id = f"trace_{len(ch['traces'])}"
            ch['state'][trace_id] = instruction
            ch['traces'].append({'step': len(ch['traces']), 'id': trace_id})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"Celestia stored: {trace_id}"
        elif chain == 'Kaspa':
            if 'trace' in instruction:
                trace_id = instruction['trace']['id']
                commitment = hashlib.sha256(str(instruction).encode()).hexdigest()[:8]
                anchor_result = self.kaspa.anchor_commitment(trace_id, commitment)
                ch['state']['anchor'] = anchor_result
                ch['traces'].append({'step': len(ch['traces']), 'anchor_hash': self.kaspa.anchors.get(trace_id, 'none')})
                self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
                return anchor_result
            return "No trace for anchor"
        elif chain == 'Juno':
            if 'policy' in instruction:
                policy = instruction['policy']
                activation_result = self.juno.activate_policy(policy)
                ch['state']['activation'] = activation_result
                ch['traces'].append({'step': len(ch['traces']), 'activation_id': self.juno.activations.get(policy, 'none')})
                self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
                return activation_result
            return "No policy for activation"
        return "No action"
    
    def replay(self):
        verified = []
        for trace in self.global_trace:
            chain = trace['chain']
            t = trace['trace']
            if chain == 'NEAR':
                if isinstance(t['output'], str) and 'norma' in t['output']:
                    norma = float(re.search(r'norma = ([\d.]+)', t['output']).group(1))
                    verified.append(f"Replay NEAR: Norma match {norma:.4f}")
                else:
                    verified.append(f"Replay NEAR: Trace {t}")
            elif chain == 'ICP':
                verified.append(f"Replay ICP: Quorum {t['quorum']}")
            elif chain == 'Celestia':
                verified.append(f"Replay Celestia: Retrieved {t['id']}")
            elif chain == 'Kaspa':
                verified.append(f"Replay Kaspa: Verified {t.get('anchor_hash', 'none')}")
            elif chain == 'Juno':
                verified.append(f"Replay Juno: Activation {t.get('activation_id', 'none')}")
        return verified

# Juno DAO Simulator (new for iter 9)
class JunoDAOSimulator:
    def __init__(self):
        self.activations = {}  # {policy: activation_id}
    
    def activate_policy(self, policy):
        # Fixed approve for demo (RNG=1.0)
        vote_approve = True  # np.random.choice([True, False], p=[0.7, 0.3]) for real
        if vote_approve:
            activation_id = hashlib.sha256(policy.encode()).hexdigest()[:8]
            self.activations[policy] = activation_id
            return f"Juno DAO: Activated policy '{policy}' → ID {activation_id}"
        return f"Juno DAO: Policy '{policy}' rejected (DAO vote failed)"

# Kaspa Simulator (from previous)
class KaspaSimulator:
    def __init__(self):
        self.dag = {}
        self.roots = []
        self.anchors = {}
    
    def anchor_commitment(self, trace_id, commitment):
        parent = np.random.choice(self.roots + list(self.dag.keys())) if self.dag else 'genesis'
        new_hash = hashlib.sha256((trace_id + commitment).encode()).hexdigest()[:8]
        self.dag.setdefault(parent, []).append(new_hash)
        if parent == 'genesis':
            self.roots.append(new_hash)
        self.anchors[trace_id] = new_hash
        return f"Anchored {trace_id} to {new_hash} (parent: {parent})"
    
    def verify_anchor(self, trace_id):
        if trace_id in self.anchors:
            return f"Verified: {trace_id} → {self.anchors[trace_id]}"
        return f"No anchor for {trace_id}"

# === DEMO ===
cs = ChainSimulator()
policy = "throttle_metabolic=50"

print("Juno DAO Demo:")
print(cs.execute_on_chain('Juno', {'policy': policy}))

print("\nICP Record Activation:")
print(cs.execute_on_chain('ICP', "record_activation"))  # Mock sync

print("\nReplay:")
for res in cs.replay()[-2:]:
    print(res)

print("\nGlobal Trace JSON:")
def json_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError
print(json.dumps(cs.global_trace, indent=2, default=json_serializable))
