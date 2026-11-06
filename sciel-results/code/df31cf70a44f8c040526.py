"""Gradient-based synthesis using direct matrix multiplication for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 215.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list using efficient kron approach."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
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
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - control)) & 1
            perm = idx ^ (bit << (qubits - 1 - target))
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    return 1.0 - min(1.0, val)


def _cnot_gates():
    return [_cx(0, 1)]


def _swap_gates():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _iswap_gates():
    gates = []
    gates.append(_u3(0, math.pi, 0, math.pi))
    gates.append(_u3(1, math.pi, 0, math.pi))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, 0, 0, math.pi / 2))
    gates.append(_u3(1, 0, 0, math.pi / 2))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, math.pi, 0, math.pi))
    gates.append(_u3(1, math.pi, 0, math.pi))
    return gates


def _toffoli_gates():
    gates = []
    gates.append(_u3(2, math.pi, 0, math.pi))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, 0, 0, math.pi / 4))
    gates.append(_cx(1, 2))
    gates.append(_u3(1, 0, 0, -math.pi / 4))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, 0, 0, math.pi / 4))
    gates.append(_u3(2, 0, 0, math.pi / 4))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, math.pi, 0, math.pi))
    return gates


def _build_circuit_from_params(params, qubits, budget):
    """Build circuit matrix from parameter vector. Returns (matrix, gate_list)."""
    pairs_2q = [(0, 1)]
    pairs_3q = [(0, 1), (1, 2), (0, 2)]
    pairs = pairs_3q if qubits == 3 else pairs_2q
    
    num_layers = budget + 1
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    gate_list = []
    
    pi = 0
    for layer in range(num_layers):
        for q in range(qubits):
            theta, phi, lam = params[pi], params[pi + 1], params[pi + 2]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
            gate_list.append(_u3(q, theta, phi, lam))
            pi += 3
        if layer < budget:
            c_idx, t_idx = pairs[layer % len(pairs)]
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - c_idx)) & 1
            perm = idx ^ (bit << (qubits - 1 - t_idx))
            matrix = matrix[perm, :]
            gate_list.append(_cx(c_idx, t_idx))
    
    return matrix, gate_list


def _objective_and_grad(params, target, qubits, budget):
    """Compute objective and gradient using finite differences on a subset of params."""
    dim = 2 ** qubits
    matrix, _ = _build_circuit_from_params(params, qubits, budget)
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    obj = 1.0 - min(1.0, val)
    
    # Compute gradient via finite differences (central difference)
    # This is expensive but we limit the number of parameters we differentiate
    grad = np.zeros_like(params)
    n_params = len(params)
    
    # For efficiency, only compute gradient for a subset of parameters
    # Use forward difference with small step
    h = 1e-5
    # Limit the number of gradient evaluations
    max_grad_evals = 200
    
    if n_params <= max_grad_evals:
        indices = range(n_params)
    else:
        # Sample a subset of parameters
        indices = np.random.choice(n_params, size=max_grad_evals, replace=False)
    
    for i in indices:
        params_plus = params.copy()
        params_plus[i] += h
        mat_plus, _ = _build_circuit_from_params(params_plus, qubits, budget)
        val_plus = abs(np.trace(target.conj().T @ mat_plus)) / dim
        obj_plus = 1.0 - min(1.0, val_plus)
        grad[i] = (obj_plus - obj) / h
    
    return obj, grad


def _gradient_descent(target, qubits, budget, max_iter=500, lr=0.1):
    """Gradient descent optimization."""
    pairs_2q = [(0, 1)]
    pairs_3q = [(0, 1), (1, 2), (0, 2)]
    pairs = pairs_3q if qubits == 3 else pairs_2q
    
    num_layers = budget + 1
    total_params = num_layers * qubits * 3
    
    # Initialize with small random values
    np.random.seed(42)
    x = np.random.uniform(-0.1, 0.1, size=total_params)
    
    best_obj = float('inf')
    best_x = x.copy()
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 10:
            break
        
        obj, grad = _objective_and_grad(x, target, qubits, budget)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        
        # Gradient step
        x = x - lr * grad
        
        # Clamp parameters to reasonable range
        x = np.clip(x, -2 * math.pi, 2 * math.pi)
        
        if obj < 1e-10:
            break
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.8
    
    matrix, gate_list = _build_circuit_from_params(best_x, qubits, budget)
    return gate_list, best_obj


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations for a generic unitary with given CNOT budget using gradient descent."""
    # Try multiple random initializations
    best_gates = []
    best_dist = float('inf')
    
    num_restarts = 3
    for restart in range(num_restarts):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Set different random seed for each restart
        np.random.seed(42 + restart)
        gates, dist = _gradient_descent(target, qubits, budget, max_iter=300, lr=0.1)
        
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
    
    return best_gates, best_dist


def synthesize(problem):
    """Synthesize quantum circuit for given target unitary within CNOT budget."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check deadline
    if _elapsed() > _DEADLINE - 5:
        return []

    # Try known decompositions
    if name == "CNOT" or (qubits == 2 and budget == 1):
        gates = _cnot_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "SWAP" or (qubits == 2 and budget == 3):
        gates = _swap_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "iSWAP" or (qubits == 2 and budget == 2):
        gates = _iswap_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "Toffoli" or (qubits == 3 and budget == 6):
        gates = _toffoli_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    # For generic unitaries, use gradient-based optimization
    if _elapsed() > _DEADLINE - 20:
        return []

    best_gates, best_dist = _optimize_generic(target, qubits, budget)

    # Verify gate count
    cnot_count = sum(1 for g in best_gates if g[0] == "cx")
    if cnot_count > budget or len(best_gates) > 400:
        pass

    return best_gates
# EVOLVE-BLOCK-END
