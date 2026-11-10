"""Quantum circuit synthesis: exact known gates + coordinate descent for generics."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list, optimized for repeated calls."""
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


def _known_decompositions(name, qubits, budget, target):
    """Return exact gate lists for known standard gates."""
    if name == "CNOT" and qubits == 2:
        if budget >= 1:
            return [["cx", 0, 1]]
    if name == "SWAP" and qubits == 2:
        if budget >= 3:
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    if name == "iSWAP" and qubits == 2:
        if budget >= 3:
            # iSWAP = exp(iπ/4) * [[0,1],[1,0]] with phases
            # Standard decomposition: 3 CNOTs
            gates = [
                ["u3", 0, math.pi / 2, 0, 0],
                ["u3", 1, math.pi / 2, 0, 0],
                ["cx", 0, 1],
                ["u3", 0, math.pi / 2, 0, math.pi / 2],
                ["u3", 1, math.pi / 2, 0, math.pi / 2],
                ["cx", 1, 0],
                ["u3", 0, math.pi / 2, 0, math.pi],
                ["u3", 1, math.pi / 2, 0, math.pi],
                ["cx", 0, 1],
            ]
            # Verify this is close to the target
            built = _build_fast(gates, qubits)
            dist = _distance(target, built)
            if dist < 1e-6:
                return gates
            return None
    if name == "Toffoli" and qubits == 3:
        if budget >= 4:
            # Standard 4-CNOT Toffoli decomposition
            gates = [
                ["u3", 2, math.pi / 2, 0, 0],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, math.pi / 2],
                ["cx", 0, 2],
                ["u3", 2, 0, 0, -math.pi / 2],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, math.pi / 2],
                ["u3", 1, 0, 0, math.pi / 2],
                ["cx", 0, 1],
                ["u3", 1, 0, 0, -math.pi / 2],
                ["cx", 0, 1],
                ["u3", 0, 0, 0, math.pi / 2],
                ["u3", 2, 0, 0, -math.pi / 2],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, math.pi / 2],
                ["u3", 2, math.pi / 2, 0, math.pi],
            ]
            # Verify this is close to the target
            built = _build_fast(gates, qubits)
            dist = _distance(target, built)
            if dist < 1e-6:
                return gates
            return None
    return None


def _coordinate_descent(target, qubits, budget, rng, time_limit=20.0):
    """Coordinate descent on fixed CNOT ladder."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    # Build layout
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = n_u3 * 3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    start_time = time.time()
    sweep = 0
    while time.time() - start_time < time_limit:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if time.time() - start_time >= time_limit:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved and sweep > 5:
            break
        sweep += 1
    
    return circuit(best), best_distance


def synthesize(problem):
    """Hybrid: exact decompositions for known gates, coordinate descent for generic."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decomposition for known gates
    exact = _known_decompositions(name, qubits, budget, target)
    if exact is not None:
        n_cx = sum(1 for g in exact if g[0] == "cx")
        n_total = len(exact)
        if n_cx <= budget and n_total <= 400:
            return exact
    
    # For generic unitaries, use coordinate descent with time management
    rng = random.Random(4242 + qubits + len(name))
    
    # Allocate time per generic target
    remaining = _time_remaining()
    time_per_target = max(5.0, min(30.0, remaining / 4.0))
    
    best_gates = None
    best_dist = 1.0
    
    # Multiple restarts if time allows
    n_restarts = 2 if time_per_target > 15 else 1
    for restart in range(n_restarts):
        if _time_remaining() < 5.0:
            break
        t_alloc = time_per_target / (n_restarts - restart) if restart < n_restarts - 1 else time_per_target
        gates, dist = _coordinate_descent(target, qubits, budget, rng, time_limit=t_alloc)
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
    
    if best_gates is None:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
# EVOLVE-BLOCK-END
