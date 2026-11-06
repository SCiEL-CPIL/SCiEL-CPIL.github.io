"""Quantum circuit synthesis with analytical decompositions and gradient-based optimization."""
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = 255.0  # safety margin before 270s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance."""
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0, math.pi)


def _s(q):
    return _u3(q, 0, 0, math.pi / 2)


def _t(q):
    return _u3(q, 0, 0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0, 0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0, 0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]

    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates

    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates

    return None


def _apply_u3_to_matrix(matrix, q, theta, phi, lam, qubits):
    """Apply u3 gate to a matrix efficiently."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                      [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == q else eye)
    return operator @ matrix


def _apply_cx_to_matrix(matrix, control, target, qubits):
    """Apply CNOT gate to a matrix efficiently."""
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _gradient_optimize(target, qubits, budget, rng_seed=42, max_iter=2000):
    """
    Gradient-based optimization using parameterized circuit.
    Uses a layout of CNOTs with single-qubit rotations between them.
    """
    if _time_left() < 10:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    rng = random.Random(rng_seed)
    
    # Build circuit layout: alternating CNOT patterns with u3 gates
    # For 2 qubits: CNOT(0,1) pattern
    # For 3 qubits: cycle through (0,1), (1,2), (0,2)
    
    if qubits == 2:
        cx_pattern = [(0, 1)]
    else:
        cx_pattern = [(0, 1), (1, 2), (0, 2)]
    
    # Number of CNOT layers to use (capped by budget)
    n_cnot_layers = min(budget, 12)
    
    # Build layout: u3 for each qubit, then CNOT, repeat
    # Parameters: for each u3 gate, 3 parameters (theta, phi, lam)
    layout = []  # list of ('u3', qubit) or ('cx', control, target)
    param_indices = {}  # maps (layer, qubit) -> param index
    param_count = 0
    
    # Initial layer of u3
    for q in range(qubits):
        layout.append(('u3', q))
        param_indices[(0, q)] = param_count
        param_count += 3
    
    # CNOT layers with u3 between
    for layer in range(n_cnot_layers):
        c, t = cx_pattern[layer % len(cx_pattern)]
        layout.append(('cx', c, t))
        
        if layer < n_cnot_layers - 1:
            for q in range(qubits):
                layout.append(('u3', q))
                param_indices[(layer + 1, q)] = param_count
                param_count += 3
    
    # Final layer of u3
    for q in range(qubits):
        layout.append(('u3', q))
        param_indices[(n_cnot_layers, q)] = param_count
        param_count += 3
    
    # Initialize parameters
    params = np.array([rng.uniform(-math.pi, math.pi) for _ in range(param_count)])
    
    def circuit_unitary(p):
        """Compute unitary from parameters."""
        matrix = np.eye(2 ** qubits, dtype=complex)
        idx = 0
        for item in layout:
            if item[0] == 'u3':
                q = item[1]
                theta, phi, lam = p[idx], p[idx + 1], p[idx + 2]
                matrix = _apply_u3_to_matrix(matrix, q, theta, phi, lam, qubits)
                idx += 3
            else:
                c, t = item[1], item[2]
                matrix = _apply_cx_to_matrix(matrix, c, t, qubits)
        return matrix
    
    def fidelity(p):
        """Compute phase-insensitive fidelity."""
        U = circuit_unitary(p)
        return abs(np.trace(target.conj().T @ U)) / target.shape[0]
    
    # Adam optimizer
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros(param_count)
    v = np.zeros(param_count)
    t = 0
    
    best_params = params.copy()
    best_fid = fidelity(params)
    
    # Compute gradient via finite differences or analytical
    # For speed, use finite differences with small step
    step_size = 1e-4
    
    iterations_done = 0
    for it in range(max_iter):
        if _time_left() < 15:
            break
        
        # Compute gradient numerically
        grad = np.zeros(param_count)
        for i in range(param_count):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += step_size
            p_minus[i] -= step_size
            grad[i] = (fidelity(p_plus) - fidelity(p_minus)) / (2 * step_size)
        
        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params + lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Clamp parameters to reasonable range
        params = np.clip(params, -2 * math.pi, 2 * math.pi)
        
        current_fid = fidelity(params)
        if current_fid > best_fid:
            best_fid = current_fid
            best_params = params.copy()
        
        if best_fid > 1.0 - 1e-10:
            break
        
        iterations_done += 1
    
    # Convert best params to gate list
    gates = []
    idx = 0
    for item in layout:
        if item[0] == 'u3':
            q = item[1]
            theta, phi, lam = best_params[idx], best_params[idx + 1], best_params[idx + 2]
            gates.append(_u3(q, theta, phi, lam))
            idx += 3
        else:
            c, t = item[1], item[2]
            gates.append(_cx(c, t))
    
    return gates


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using gradient-based optimization with multiple restarts."""
    if _time_left() < 10:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    
    # Try multiple restarts with different seeds and layouts
    best_gates = None
    best_distance = 1.0
    
    # Number of restarts based on time available
    time_per_restart = 30
    n_restarts = max(1, int((_time_left() - 20) // time_per_restart))
    n_restarts = min(n_restarts, 5)  # Cap restarts
    
    for restart in range(n_restarts):
        if _time_left() < 20:
            break
        
        seed = rng_seed + restart * 1000
        # Vary the number of CNOT layers
        if qubits == 2:
            max_layers = min(budget, 10)
        else:
            max_layers = min(budget, 8)
        
        # Try different numbers of layers
        layer_choices = [max_layers, max(2, max_layers - 2), max(1, max_layers - 4)]
        layer_choice = layer_choices[restart % len(layer_choices)]
        
        gates = _gradient_optimize(target, qubits, layer_choice, rng_seed=seed, max_iter=1000)
        
        if gates is not None:
            dist = _distance(target, _build(gates, qubits))
            if dist < best_distance:
                best_distance = dist
                best_gates = gates
                if best_distance < 1e-8:
                    break
    
    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    if _time_left() < 2:
        # Emergency fallback: return identity or something minimal
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # Fallback to optimization if known circuit failed or doesn't exist
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
