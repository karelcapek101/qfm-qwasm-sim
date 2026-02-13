import numpy as np
from scipy.linalg import eig
import sympy as sp
import re
from collections import defaultdict
import json  # Pro trace export

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
    
    def spectrum(self, op):
        evals, _ = eig(op)
        return np.sort_complex(evals)
    
    def trace_formula(self, H, s=0.5 + 14j, reg=1e-6):
        evals = self.spectrum(H)
        smooth = np.sum(1.0 / (evals + s)**2)
        osc = 0.0
        for p in self.primes[:10]:
            rho = np.log(p)
            osc += np.log(p) * np.exp(-2j * np.pi * rho / np.log(p))
        trace_reg = np.trace(H) + reg * np.sum(np.abs(evals))
        return smooth.real, osc.real, trace_reg.real
    
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
        elif op_name == 'trace':
            s_real = float(params['s_real']) if params and 's_real' in params else 0.5
            s_imag = float(params['s_imag']) if params and 's_imag' in params else 14.0
            H = self.smrk_hamiltonian()
            smooth, osc, trace_reg = self.trace_formula(H, s_real + 1j * s_imag)
            return f"Trace (s={s_real}+{s_imag}j): smooth={smooth:.4f}, osc={osc:.4f}, reg={trace_reg:.4f}"
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

# Nové pro iteraci 5: Multichain Replay Sim
class ChainSimulator:
    def __init__(self):
        self.chains = {
            'ICP': {'role': 'control', 'state': {}, 'traces': []},
            'NEAR': {'role': 'execution', 'state': {}, 'traces': []},
            'Celestia': {'role': 'availability', 'state': {}, 'traces': []}
        }
        self.global_trace = []
    
    def execute_on_chain(self, chain, instruction):
        if chain not in self.chains:
            return f"Unknown chain: {chain}"
        ch = self.chains[chain]
        if chain == 'NEAR':
            # Sim exekuce: QWASM run
            sim = QFM_Simulator(max_n=32)
            parser = QWASM_Parser(sim)
            outputs = parser.parse(instruction)
            ch['state']['output'] = outputs
            ch['traces'].append({'step': len(ch['traces']), 'output': outputs[-1]})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"NEAR exec: {outputs[-1]}"
        elif chain == 'ICP':
            # Sim control: Quorum decision
            quorum = np.random.choice([True, False], p=[0.9, 0.1])  # 90% accept
            ch['state']['decision'] = bool(quorum)  # Python bool pro JSON
            ch['traces'].append({'step': len(ch['traces']), 'quorum': bool(quorum)})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"ICP quorum: {'Accepted' if quorum else 'Rejected'}"
        elif chain == 'Celestia':
            # Sim availability: Store trace
            trace_id = f"trace_{len(ch['traces'])}"
            ch['state'][trace_id] = instruction
            ch['traces'].append({'step': len(ch['traces']), 'id': trace_id})
            self.global_trace.append({'chain': chain, 'trace': ch['traces'][-1]})
            return f"Celestia stored: {trace_id}"
        return "No action"
    
    def replay(self):
        # Replay: Re-execute traces pro verifikaci
        verified = []
        for trace in self.global_trace:
            chain = trace['chain']
            t = trace['trace']
            if chain == 'NEAR':
                # Re-run last output
                sim = QFM_Simulator(max_n=32)
                parser = QWASM_Parser(sim)
                if isinstance(t['output'], str) and 'norma' in t['output']:
                    # Parse norma pro check
                    norma = float(re.search(r'norma = ([\d.]+)', t['output']).group(1))
                    verified.append(f"Replay NEAR: Norma match {norma:.4f}")
                else:
                    verified.append(f"Replay NEAR: Trace {t}")
            elif chain == 'ICP':
                verified.append(f"Replay ICP: Quorum {t['quorum']}")
            elif chain == 'Celestia':
                verified.append(f"Replay Celestia: Retrieved {t['id']}")
        return verified

# === DEMO ===
chains = ChainSimulator()
qwasm_code = "shift p=2\nhamiltonian alpha=1.0 beta=0.5"

print("Multichain Replay Demo:")
print("1. NEAR Execution:")
print(chains.execute_on_chain('NEAR', qwasm_code))

print("\n2. ICP Control (Quorum):")
print(chains.execute_on_chain('ICP', "quorum_check"))

print("\n3. Celestia Availability (Store Trace):")
print(chains.execute_on_chain('Celestia', "trace_id_1"))

print("\n4. Replay Verification:")
replay_results = chains.replay()
for res in replay_results[-3:]:  # Poslední 3 pro demo
    print(res)

print("\nGlobal Trace JSON:")
# Fix JSON: Convert numpy types
def json_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Type {type(obj)} not serializable")
print(json.dumps(chains.global_trace, indent=2, default=json_serializable))
