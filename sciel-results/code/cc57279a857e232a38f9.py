"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds safety margin before 270s kill


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


def _known_decompositions(problem):
    """Return exact gate lists for known gates, or None."""
    name = problem["name"]
    qubits = problem["qubits"]
    target = _target(problem)
    
    # CNOT gate
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    
    # iSWAP gate
    if name == "iSWAP" and qubits == 2:
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
            ["cx", 0, 1],
            ["u3", 0, 0.0, 0.0, -math.pi / 2],  # S† on q0
            ["u3", 1, 0.0, 0.0, math.pi / 2],   # S on q1
            ["cx", 1, 0],
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
        ]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    # SWAP gate
    if name == "SWAP" and qubits == 2:
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    # Toffoli gate
    if name == "Toffoli" and qubits == 3:
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, math.pi / 4],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
        ]
        built = _build(gates, 3)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    return None


def _optimize_generic(problem, max_time=25.0):
    """Optimize a generic unitary using gradient-based optimization."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build CNOT layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_ctrls = [pairs[i % len(pairs)][0] for i in range(budget)]
    cnot_tgts = [pairs[i % len(pairs)][1] for i in range(budget)]
    
    # U3 slots: before each CNOT layer (all qubits) and after last layer
    u3_qubits = []
    for layer in range(budget):
        for q in range(qubits):
            u3_qubits.append(q)
    for q in range(qubits):
        u3_qubits.append(q)
    
    n_u3 = len(u3_qubits)
    n_params = 3 * n_u3
    
    def build_from_params(params):
        mat = np.eye(n, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        
        cnot_perms = []
        for ctrl, tgt in zip(cnot_ctrls, cnot_tgts):
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - ctrl)) & 1
            perm = idx ^ (bit << (qubits - 1 - tgt))
            cnot_perms.append(perm)
        
        gate_idx = 0
        cnot_idx = 0
        u3_idx = 0
        
        for layer in range(budget):
            for q in range(qubits):
                theta, phi, lam = params[3*u3_idx], params[3*u3_idx+1], params[3*u3_idx+2]
                c, s = math.cos(theta/2), math.sin(theta/2)
                single = np.array([[c, -np.exp(1j*lam)*s],
                                   [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]])
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if k == q else eye2)
                mat = op @ mat
                u3_idx += 1
            mat = mat[cnot_perms[cnot_idx], :]
            cnot_idx += 1
        
        for q in range(qubits):
            theta, phi, lam = params[3*u3_idx], params[3*u3_idx+1], params[3*u3_idx+2]
            c, s = math.cos(theta/2), math.sin(theta/2)
            single = np.array([[c, -np.exp(1j*lam)*s],
                               [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]])
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            mat = op @ mat
            u3_idx += 1
            
        return mat

    def objective(params):
        mat = build_from_params(params)
        return _distance(target, mat)

    # Initial guess
    best_params = np.zeros(n_params)
    best_dist = objective(best_params)
    
    start = time.time()
    
    # Try a few random initializations
    for _ in range(5):
        if time.time() - start > max_time or _time_remaining() < 5.0:
            break
        params = np.random.uniform(-math.pi, math.pi, n_params)
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100})
        if res.fun < best_dist:
            best_dist = res.fun
            best_params = res.x
    
    # Convert to gate list
    gates = []
    k = 0
    for layer in range(budget):
        for q in range(qubits):
            gates.append(["u3", q, best_params[k], best_params[k+1], best_params[k+2]])
            k += 3
        gates.append(["cx", cnot_ctrls[layer], cnot_tgts[layer]])
    for q in range(qubits):
        gates.append(["u3", q, best_params[k], best_params[k+1], best_params[k+2]])
        k += 3
    
    return gates


def synthesize(problem):
    """Synthesize a quantum circuit within the CNOT budget."""
    # First try known exact decompositions
    known = _known_decompositions(problem)
    if known is not None:
        return known
    
    # For generic unitaries, use optimization
    return _optimize_generic(problem)
# EVOLVE-BLOCK-END
