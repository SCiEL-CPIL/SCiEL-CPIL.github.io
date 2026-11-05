"""Quantum circuit synthesis using KAK decomposition for 2-qubit and QSD for 3-qubit."""
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


def _apply_u3(matrix, qubit, theta, phi, lam, nqubits):
    """Apply u3 gate on qubit (0-indexed, 0=MSB) to the full matrix."""
    if abs(theta) < 1e-12 and abs(phi) < 1e-12 and abs(lam) < 1e-12:
        return matrix
    u3 = _u3_matrix(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(nqubits):
        op = np.kron(op, u3 if k == qubit else eye)
    return op @ matrix


def _apply_cnot(matrix, control, target, nqubits):
    """Apply CNOT gate with control and target (0-indexed, 0=MSB)."""
    dim = 2 ** nqubits
    index = np.arange(dim)
    bit = (index >> (nqubits - 1 - control)) & 1
    return matrix[index ^ (bit << (nqubits - 1 - target)), :]


def _build(gates, qubits):
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cnot(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _synthesize_2qubit_named(name, target):
    """Synthesize named 2-qubit gates exactly."""
    gates = []
    
    if name == "CNOT" or name == "cx":
        # CNOT: just a CNOT gate
        gates = [["cx", 0, 1]]
        
    elif name == "iSWAP" or name == "iswap":
        # iSWAP decomposition
        gates = [
            ["u3", 0, math.pi/2, 0, 0],
            ["u3", 1, math.pi/2, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, -math.pi/2, 0, 0],
            ["u3", 1, -math.pi/2, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0, 0],
            ["u3", 1, math.pi/2, 0, 0],
            ["cx", 0, 1],
        ]
        
    elif name == "SWAP" or name == "swap":
        # SWAP = CNOT(0,1) · CNOT(1,0) · CNOT(0,1)
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    
    return gates


def _synthesize_3qubit_named(name, target):
    """Synthesize named 3-qubit gates exactly."""
    gates = []
    
    if name == "Toffoli" or name == "toffoli" or name == "CCX":
        # Toffoli decomposition
        gates = [
            ["cx", 1, 2],
            ["u3", 2, 0, 0, math.pi/4],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, -math.pi/4],
            ["cx", 1, 2],
            ["u3", 1, 0, 0, math.pi/4],
            ["cx", 0, 1],
            ["u3", 1, 0, 0, -math.pi/4],
            ["cx", 0, 1],
            ["u3", 1, 0, 0, math.pi/4],
            ["u3", 2, 0, 0, math.pi/4],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, -math.pi/4],
            ["cx", 0, 1],
        ]
    
    return gates


def _synthesize_generic_2qubit(target, budget):
    """
    Synthesize a generic 2-qubit unitary using coordinate descent.
    Uses a 2-CNOT layout if budget >= 2, else 1-CNOT.
    """
    # Use 2 CNOTs for generic 2-qubit
    if budget >= 2:
        layout = [
            ["u3", 0], ["u3", 1],
            ["cx", 0, 1],
            ["u3", 0], ["u3", 1],
            ["cx", 0, 1],
            ["u3", 0], ["u3", 1],
        ]
    else:
        layout = [
            ["u3", 0], ["u3", 1],
            ["cx", 0, 1],
            ["u3", 0], ["u3", 1],
        ]

    n_params = sum(3 for g in layout if g[0] == "u3")
    
    def build_circuit(params):
        gates = []
        p_idx = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                theta = params[p_idx]
                phi = params[p_idx + 1]
                lam = params[p_idx + 2]
                gates.append(["u3", q, theta, phi, lam])
                p_idx += 3
            else:
                gates.append(list(gate))
        return gates

    best_params = [0.0] * n_params
    best_dist = _distance(target, _build(build_circuit(best_params), 2))
    
    # Coordinate descent
    max_iter = 50
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n_params):
            if _time_remaining() < 5.0:
                break
            best_local = best_params[i]
            best_local_dist = best_dist
            # Try a few steps
            for trial in [0.0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, -math.pi/4, -math.pi/2]:
                params = best_params[:]
                params[i] = trial
                dist = _distance(target, _build(build_circuit(params), 2))
                if dist < best_local_dist:
                    best_local_dist = dist
                    best_local = trial
            if best_local != best_params[i]:
                best_params[i] = best_local
                best_dist = best_local_dist
                improved = True
        if not improved:
            break
            
    return build_circuit(best_params)


def _synthesize_generic_3qubit(target, budget):
    """
    Synthesize a generic 3-qubit unitary.
    """
    nqubits = 3
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2)]
    
    layout = []
    for layer in range(min(budget, 12)):
        layout += [["u3", q] for q in range(nqubits)]
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    layout += [["u3", q] for q in range(nqubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    params = [0.0] * n_params
    
    def build_circuit(params):
        gates = []
        p_idx = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                theta = params[p_idx]
                phi = params[p_idx + 1]
                lam = params[p_idx + 2]
                gates.append(["u3", q, theta, phi, lam])
                p_idx += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_params = params[:]
    best_dist = _distance(target, _build(build_circuit(best_params), 3))
    
    max_iter = 20
    for iteration in range(max_iter):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n_params):
            if _time_remaining() < 5.0:
                break
            best_local = best_params[i]
            best_local_dist = best_dist
            for trial in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
                params[i] = trial
                dist = _distance(target, _build(build_circuit(params), 3))
                if dist < best_local_dist:
                    best_local_dist = dist
                    best_local = trial
            if best_local != best_params[i]:
                best_params[i] = best_local
                best_dist = best_local_dist
                improved = True
        if not improved:
            break
    
    return build_circuit(best_params)


def _identity_gates(nqubits):
    """Return a list of identity gates (u3 with 0 params)."""
    return [["u3", i, 0.0, 0.0, 0.0] for i in range(nqubits)]

def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    
    try:
        target = _target(problem)
    except Exception:
        return _identity_gates(qubits)
    
    if _time_remaining() < 2.0:
        return _identity_gates(qubits)
    
    if qubits == 2:
        named_gates = _synthesize_2qubit_named(name, target)
        if named_gates:
            built = _build(named_gates, 2)
            dist = _distance(target, built)
            if dist < 0.01:
                return named_gates
        return _synthesize_generic_2qubit(target, budget)
    
    elif qubits == 3:
        named_gates = _synthesize_3qubit_named(name, target)
        if named_gates:
            built = _build(named_gates, 3)
            dist = _distance(target, built)
            if dist < 0.01:
                return named_gates
        return _synthesize_generic_3qubit(target, budget)
    
    return _identity_gates(qubits)
# EVOLVE-BLOCK-END
