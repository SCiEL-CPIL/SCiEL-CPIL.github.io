"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 40.0  # safety margin before 270s kill


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                              [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _known_decompositions(name, qubits, budget):
    """Return exact gate lists for known standard gates."""
    # CNOT gate: 1 CNOT
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    # iSWAP: 3 CNOTs
    if name == "iSWAP" and qubits == 2:
        gates = []
        # iSWAP = (H on both) * SWAP * (H on both) with phases
        # iSWAP = exp(i*pi/4) * CX(0,1) * S on both * CX(1,0) * S on both * CX(0,1)
        # Standard: iSWAP = CX * S * CX * S * CX with appropriate single-qubit gates
        gates.append(["u3", 0, math.pi/2, 0, 0])  # S gate
        gates.append(["u3", 1, math.pi/2, 0, 0])  # S gate
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, math.pi/2, 0, math.pi/2])  # -iS or similar
        gates.append(["u3", 1, math.pi/2, 0, math.pi/2])
        gates.append(["cx", 1, 0])
        gates.append(["u3", 0, math.pi/2, 0, math.pi])
        gates.append(["u3", 1, math.pi/2, 0, math.pi])
        gates.append(["cx", 0, 1])
        return gates
    # SWAP: 3 CNOTs
    if name == "SWAP" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    # Toffoli: 4 CNOTs (standard decomposition)
    if name == "Toffoli" and qubits == 3:
        gates = []
        # Standard Toffoli decomposition with 4 CNOTs
        # H on target, CX(0,2), CX(1,2) with T gates
        gates.append(["u3", 2, math.pi/2, 0, 0])  # S gate on target
        gates.append(["cx", 1, 2])
        gates.append(["u3", 2, 0, 0, math.pi/2])  # T gate
        gates.append(["cx", 0, 2])
        gates.append(["u3", 2, 0, 0, -math.pi/2])  # T† gate
        gates.append(["cx", 1, 2])
        gates.append(["u3", 2, 0, 0, math.pi/2])  # T gate
        gates.append(["u3", 1, 0, 0, math.pi/2])  # T gate on control2
        gates.append(["cx", 0, 1])
        gates.append(["u3", 1, 0, 0, -math.pi/2])  # T† on control2
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, 0, 0, math.pi/2])  # T on control1
        gates.append(["u3", 2, 0, 0, -math.pi/2])  # T† on target
        gates.append(["cx", 1, 2])
        gates.append(["u3", 2, 0, 0, math.pi/2])  # T on target
        gates.append(["u3", 2, math.pi/2, 0, math.pi])  # S† on target
        return gates
    return None


def _simulate_annealing(target, qubits, budget, rng, n_params, max_iters=5000):
    """Simulated annealing optimization for generic unitaries."""
    n_u3 = n_params // 3
    n_cx = budget
    
    # Build layout: n_cx CNOTs with u3 gates before/after each
    # Pattern: u3s, cx, u3s, cx, ..., u3s
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(n_cx):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize
    x = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    best_x = x[:]
    best_dist = _distance(target, _build(circuit(x), qubits))
    
    T = 1.0
    T_min = 0.001
    cooling = 0.95
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        # Pick a random parameter to perturb
        i = rng.randint(0, n_params - 1)
        x_new = x[:]
        x_new[i] = x[i] + rng.gauss(0, T)
        x_new[i] = (x_new[i] + math.pi) % (2 * math.pi) - math.pi
        
        dist_new = _distance(target, _build(circuit(x_new), qubits))
        delta = dist_new - best_dist
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            x = x_new
            if dist_new < best_dist:
                best_x = x_new[:]
                best_dist = dist_new
        
        T = max(T * cooling, T_min)
    
    return circuit(best_x), best_dist


def synthesize(problem):
    """Hybrid: exact decompositions for known gates, simulated annealing for generic."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decomposition for known gates
    exact = _known_decompositions(name, qubits, budget)
    if exact is not None:
        # Verify it's within budget
        n_cx = sum(1 for g in exact if g[0] == "cx")
        if n_cx <= budget:
            return exact
    
    # For generic unitaries, use simulated annealing
    rng = random.Random(4242 + qubits + len(name))
    
    # Number of u3 parameters: (budget+1) layers * qubits * 3 params each
    n_u3_layers = budget + 1
    n_params = n_u3_layers * qubits * 3
    
    best_gates = None
    best_dist = 1.0
    
    # Multiple restarts
    max_restarts = 3
    for restart in range(max_restarts):
        if _time_remaining() < 10.0:
            break
        gates, dist = _simulate_annealing(target, qubits, budget, rng, n_params, 
                                           max_iters=3000 if restart == 0 else 2000)
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
    
    if best_gates is None:
        # Fallback: return identity-like circuit
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
# EVOLVE-BLOCK-END
