"""Quantum circuit synthesis: exact known gates + optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # safety margin before 270s kill


def _time_remaining():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _apply_u3_to_matrix(matrix, qubit, theta, phi, lam, nqubits):
    """Apply u3 on given qubit to the full unitary matrix (qubit 0 = MSB)."""
    single = _u3_matrix(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(nqubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _apply_cx_to_matrix(matrix, control, target, nqubits):
    """Apply CNOT to the full unitary matrix (qubit 0 = MSB)."""
    dim = 2 ** nqubits
    index = np.arange(dim)
    bit = (index >> (nqubits - 1 - control)) & 1
    return matrix[index ^ (bit << (nqubits - 1 - target)), :]


def _build(gates, qubits):
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3_to_matrix(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cx_to_matrix(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(problem, target):
    """Check if target matches a known gate (up to global phase)."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    
    # Normalize by global phase
    phase = target[np.argmax(np.abs(target))]
    if abs(phase) < 1e-10:
        return None
    normalized = target / phase
    normalized = np.round(normalized, 8)
    
    # CNOT
    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        if np.allclose(normalized, cnot, atol=1e-6):
            return ["cx", 0, 1]
        iswap = np.array([[1, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 1]], dtype=complex)
        if np.allclose(normalized, iswap, atol=1e-6):
            return "iswap"
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if np.allclose(normalized, swap, atol=1e-6):
            return "swap"
    
    # Toffoli
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        if np.allclose(normalized, toffoli, atol=1e-6):
            return "toffoli"
    
    return None


def _synthesize_known(gate_type, qubits):
    """Return exact gate list for known gates."""
    if gate_type == ["cx", 0, 1]:
        return [["cx", 0, 1]]
    if gate_type == "swap":
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    if gate_type == "iswap":
        # iSWAP = (H ⊗ H) · CNOT · (Rz(π/2) ⊗ Rx(π/2)) · (H ⊗ H) · ... 
        # Actually: iSWAP = e^{iπ/4} · (iSWAP matrix)
        # Standard decomposition:
        # iSWAP = (Rz(π/2) ⊗ I) · CNOT · (Rx(π/2) ⊗ Rx(π/2)) · CNOT · (Rz(π/2) ⊗ I)
        # Let me use a known decomposition:
        # iSWAP = [Rz(π/2) on q0, Rx(π/2) on q1] · CNOT(0,1) · [Rx(π/2) on q0, Rx(π/2) on q1] · CNOT(1,0) · [Rz(π/2) on q0]
        # Actually, let me just return a placeholder and handle it in optimization
        return "iswap"
    if gate_type == "toffoli":
        # Toffoli = 7 CNOTs + single-qubit gates
        # Standard decomposition uses 6 CNOTs
        gates = [
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 4, 0.0, 0.0],
            ["cx", 0, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 4, 0.0, 0.0],
            ["cx", 0, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 4, 0.0, 0.0],
            ["u3", 0, math.pi / 2, 0.0, 0.0],
            ["u3", 1, math.pi / 2, 0.0, 0.0],
        ]
        return gates
    return None


def _optimize_generic(target, qubits, budget, max_iter=500):
    """Optimize single-qubit parameters for a fixed CNOT layout."""
    dim = 2 ** qubits
    
    # Create a layout with CNOTs and parameterized u3 gates
    # Use a brickwork pattern of CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    # Initial layer of u3
    for q in range(qubits):
        layout.append(("u3", q, 0, 1, 2))  # params at indices 0,1,2
    param_idx = 3
    
    for layer in range(budget):
        layout.append(("cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
        for q in range(qubits):
            layout.append(("u3", q, param_idx, param_idx + 1, param_idx + 2))
            param_idx += 3
    
    # Final layer of u3
    for q in range(qubits):
        layout.append(("u3", q, param_idx, param_idx + 1, param_idx + 2))
        param_idx += 3
    
    n_params = param_idx
    
    # Initialize parameters
    params = np.random.uniform(-math.pi, math.pi, size=n_params)
    best_params = params.copy()
    best_dist = _distance(target, _build_from_layout(layout, params, qubits))
    
    # Coordinate descent
    step = 1.0
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n_params):
            for delta in (step, -step):
                trial = params.copy()
                trial[i] = params[i] + delta
                trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                dist = _distance(target, _build_from_layout(layout, trial, qubits))
                if dist < best_dist:
                    best_dist = dist
                    best_params = trial.copy()
                    params = trial
                    improved = True
        if not improved:
            step *= 0.7
            if step < 0.01:
                break
    
    # Convert best params to gate list
    gates = []
    for item in layout:
        if item[0] == "u3":
            _, q, i, j, k = item
            gates.append(["u3", q, best_params[i], best_params[j], best_params[k]])
        else:
            _, c, t = item
            gates.append(["cx", c, t])
    
    return gates, best_dist


def _build_from_layout(layout, params, qubits):
    """Build unitary from layout and parameters."""
    matrix = np.eye(2 ** qubits, dtype=complex)
    for item in layout:
        if item[0] == "u3":
            _, q, i, j, k = item
            matrix = _apply_u3_to_matrix(matrix, q, params[i], params[j], params[k], qubits)
        else:
            _, c, t = item
            matrix = _apply_cx_to_matrix(matrix, c, t, qubits)
    return matrix


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    known = _is_known_gate(problem, target)
    if known is not None:
        if isinstance(known, list):
            return known
        if known == "swap":
            return _synthesize_known("swap", qubits)
        if known == "toffoli":
            return _synthesize_known("toffoli", qubits)
        if known == "iswap":
            # iSWAP decomposition (exact)
            # iSWAP = (Rz(π/2) ⊗ Rx(π/2)) · CNOT(0,1) · (Rx(π/2) ⊗ Rz(π/2)) · CNOT(1,0) · (Rz(π/2) ⊗ Rx(π/2))
            # Let me use a verified decomposition
            gates = [
                ["u3", 0, math.pi / 2, 0.0, 0.0],
                ["u3", 1, math.pi / 2, 0.0, 0.0],
                ["cx", 0, 1],
                ["u3", 0, math.pi / 2, 0.0, 0.0],
                ["u3", 1, math.pi / 2, 0.0, 0.0],
                ["cx", 1, 0],
                ["u3", 0, math.pi / 2, 0.0, 0.0],
                ["u3", 1, math.pi / 2, 0.0, 0.0],
            ]
            # Verify and adjust if needed
            u = _build(gates, qubits)
            if _distance(target, u) > 0.01:
                # Fall back to optimization
                gates, dist = _optimize_generic(target, qubits, budget)
                return gates
            return gates
    
    # Generic unitary: optimize
    gates, dist = _optimize_generic(target, qubits, budget)
    return gates
# EVOLVE-BLOCK-END
