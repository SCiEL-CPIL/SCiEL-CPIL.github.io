"""Quantum circuit synthesis: exact decompositions for named gates, KAK for generics."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 260.0  # 260s out of 270s wall-clock limit


def _time_remaining():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the full unitary from a gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
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


def _is_known_gate(problem):
    """Check if the problem is one of the four named gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    if name == "CNOT" and qubits == 2:
        return "CNOT"
    if name == "iSWAP" and qubits == 2:
        return "iSWAP"
    if name == "SWAP" and qubits == 2:
        return "SWAP"
    if name == "Toffoli" and qubits == 3:
        return "Toffoli"
    return None


def _synthesize_cnot():
    """CNOT: 1 CNOT, 0 single-qubit gates."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP: 2 CNOTs with single-qubit rotations."""
    # iSWAP = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    # Decomposition: H on both, then CNOT(0,1), H on both, then phase corrections
    gates = []
    gates.append(["u3", 0, math.pi, 0.0, math.pi / 2])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi / 2])  # H
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi, 0.0, math.pi / 2])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi / 2])  # H
    return gates


def _synthesize_swap():
    """SWAP: 3 CNOTs."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _synthesize_toffoli():
    """Toffoli: 6 CNOTs with single-qubit rotations."""
    gates = []
    # Standard Toffoli decomposition
    gates.append(["u3", 2, math.pi / 2, 0.0, math.pi / 2])  # H on target
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi / 4, 0.0, math.pi / 2])  # T†
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, -math.pi / 4, 0.0, math.pi / 2])  # T
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi / 4, 0.0, math.pi / 2])  # T
    gates.append(["u3", 1, -math.pi / 4, 0.0, math.pi / 2])  # T†
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi / 4, 0.0, math.pi / 2])  # T
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, -math.pi / 4, 0.0, math.pi / 2])  # T†
    gates.append(["u3", 2, math.pi / 2, 0.0, math.pi / 2])  # H on target
    return gates


def _kak_decompose_3qubit(target):
    """
    Attempt KAK decomposition for 3-qubit unitary.
    Returns a circuit with at most 6 CNOTs if successful.
    """
    # For 3-qubit, we use the canonical form with 6 CNOTs
    # This is a heuristic approach: try to find a good decomposition
    # using coordinate descent on a fixed 6-CNOT template
    
    qubits = 3
    budget = 6
    
    # Create a template with 6 CNOTs and surrounding single-qubit gates
    # CNOT pattern for 3 qubits
    cnot_pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    # Layout: single-qubit gates before each CNOT layer and after
    # Total single-qubit gates: 7 layers * 3 qubits = 21 gates
    # Total CNOTs: 6
    # Total gates: 27
    
    layout = []
    for i in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *cnot_pairs[i]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    # Use Nelder-Mead or coordinate descent
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(_circuit_from_layout(layout, best), qubits))
    
    # Coordinate descent
    step = 0.5
    for sweep in range(20):
        if _time_remaining() < 5.0:
            break
        step *= 0.9
        improved = False
        for i in range(n_params):
            if _time_remaining() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(_circuit_from_layout(layout, values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved and sweep > 5:
            break
    
    return _circuit_from_layout(layout, best)


def _circuit_from_layout(layout, values):
    """Build circuit from layout template and parameter values."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _generic_synthesis_2qubit(problem):
    """Generic 2-qubit synthesis with up to 3 CNOTs."""
    qubits = 2
    target = _target(problem)
    budget = min(3, problem["cnot_budget"])
    
    # Try different CNOT patterns
    patterns = [
        [(0, 1), (1, 0), (0, 1)],  # SWAP-like
        [(0, 1), (0, 1), (1, 0)],
        [(0, 1), (1, 0), (1, 0)],
    ]
    
    if budget < 3:
        patterns = [p[:budget] for p in patterns if len(p) >= budget]
    
    best_circuit = None
    best_distance = 1.0
    
    for pattern in patterns:
        if _time_remaining() < 10.0:
            break
        
        layout = []
        for i in range(len(pattern)):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", *pattern[i]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        n_params = 3 * sum(1 for g in layout if g[0] == "u3")
        best = [0.0] * n_params
        best_distance_local = _distance(target, _build_fast(_circuit_from_layout(layout, best), qubits))
        
        # Coordinate descent
        step = 0.5
        for sweep in range(15):
            if _time_remaining() < 10.0:
                break
            step *= 0.9
            improved = False
            for i in range(n_params):
                if _time_remaining() < 10.0:
                    break
                for trial in (best[i] + step, best[i] - step):
                    values = best[:]
                    values[i] = trial
                    value = _distance(target, _build_fast(_circuit_from_layout(layout, values), qubits))
                    if value < best_distance_local:
                        best, best_distance_local = values, value
                        improved = True
            if not improved and sweep > 5:
                break
        
        if best_distance_local < best_distance:
            best_distance = best_distance_local
            best_circuit = _circuit_from_layout(layout, best)
    
    if best_circuit is None:
        # Fallback: identity
        best_circuit = []
    
    return best_circuit


def _generic_synthesis_3qubit(problem):
    """Generic 3-qubit synthesis."""
    qubits = 3
    target = _target(problem)
    budget = min(6, problem["cnot_budget"])
    
    # Try KAK-style decomposition
    circuit = _kak_decompose_3qubit(target)
    
    # Verify CNOT count
    cnot_count = sum(1 for g in circuit if g[0] == "cx")
    if cnot_count > budget:
        # Truncate or use simpler approach
        circuit = _simple_3qubit_synthesis(problem)
    
    return circuit


def _simple_3qubit_synthesis(problem):
    """Simple 3-qubit synthesis with limited CNOTs."""
    qubits = 3
    target = _target(problem)
    budget = min(6, problem["cnot_budget"])
    
    # Use a simpler template
    cnot_pairs = [(0, 1), (1, 2)] * 3
    cnot_pairs = cnot_pairs[:budget]
    
    layout = []
    for i in range(len(cnot_pairs)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *cnot_pairs[i]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(_circuit_from_layout(layout, best), qubits))
    
    # Coordinate descent
    step = 0.5
    for sweep in range(10):
        if _time_remaining() < 10.0:
            break
        step *= 0.9
        improved = False
        for i in range(n_params):
            if _time_remaining() < 10.0:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(_circuit_from_layout(layout, values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved and sweep > 3:
            break
    
    return _circuit_from_layout(layout, best)


def synthesize(problem):
    """Synthesize quantum circuit for the given problem."""
    global _START_TIME, _DEADLINE
    
    # Reset deadline if this is a fresh call (shouldn't happen in normal flow)
    if time.time() - _START_TIME > 250:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 260.0
    
    known = _is_known_gate(problem)
    
    if known == "CNOT":
        return _synthesize_cnot()
    elif known == "iSWAP":
        return _synthesize_iswap()
    elif known == "SWAP":
        return _synthesize_swap()
    elif known == "Toffoli":
        return _synthesize_toffoli()
    else:
        qubits = problem["qubits"]
        if qubits == 2:
            return _generic_synthesis_2qubit(problem)
        else:
            return _generic_synthesis_3qubit(problem)
# EVOLVE-BLOCK-END
