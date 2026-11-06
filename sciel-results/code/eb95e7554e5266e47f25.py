"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _build_fast(gates, qubits):
    n = 2 ** qubits
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            # CNOT: flip target bit where control bit is 1
            matrix = matrix[np.where(bit == 1, index ^ (1 << (qubits - 1 - target)), index), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _try_exact_cnot(problem):
    """Try to find exact decomposition with known CNOT counts."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # For 2 qubits, any unitary can be decomposed with at most 3 CNOTs
    # For 3 qubits, Toffoli needs 6 CNOTs (without ancilla)
    # We'll use a simple approach: try to match the target exactly
    
    # For 2-qubit gates, we can use the standard decomposition
    if qubits == 2:
        # Standard 2-qubit decomposition: U = (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F) CNOT (G ⊗ H)
        # This is complex to implement from scratch, so we'll rely on optimization
        pass
    
    return None  # Fall through to optimization


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact methods first for known gates
    exact_result = _try_exact_cnot(problem)
    if exact_result is not None:
        return exact_result
    
    # Build layout: alternating layers of u3 gates and CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with adaptive steps
    for sweep in range(20):
        if _check_deadline():
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _check_deadline():
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    step = max(step * 0.5, 0.01)
    
    return circuit(best)
# EVOLVE-BLOCK-END
