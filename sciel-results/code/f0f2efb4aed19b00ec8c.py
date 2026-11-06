"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _apply_u3(matrix, q, theta, phi, lam, qubits):
    """Apply u3 gate to a specific qubit in-place on the matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    bit_pos = qubits - 1 - q
    n = 2 ** qubits
    for base in range(n):
        if base & (1 << bit_pos):
            continue
        i0 = base
        i1 = base | (1 << bit_pos)
        m00 = matrix[i0]
        m01 = matrix[i1]
        matrix[i0] = c * m00 - e1 * s * m01
        matrix[i1] = e2 * s * m00 + e2 * e1 * c * m01


def _apply_cx(matrix, control, target, qubits):
    """Apply CNOT gate in-place on the matrix."""
    bit_pos_t = qubits - 1 - target
    c_bit_pos = qubits - 1 - control
    n = 2 ** qubits
    for i in range(n):
        if i & (1 << c_bit_pos):
            j = i ^ (1 << bit_pos_t)
            tmp = matrix[i].copy()
            matrix[i] = matrix[j]
            matrix[j] = tmp


def _build_fast(gates, qubits):
    """Build unitary using in-place operations for speed."""
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            _apply_u3(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            _apply_cx(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _known_decomposition(name, qubits, budget):
    """Return known exact decompositions for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    
    if name == "iSWAP" and qubits == 2:
        # iSWAP matrix: [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
        # Decomposition using 1 CNOT
        # iSWAP = exp(i*pi/4*(XX+YY)/2)
        # One decomposition: S on both, H on both, CNOT, H on both, S-dag on both
        # Let's use a known 1-CNOT decomposition
        gates = [
            ["u3", 0, math.pi/2, 0, math.pi/2],  # S on q0
            ["u3", 1, math.pi/2, 0, math.pi/2],  # S on q1
            ["u3", 0, math.pi/2, 0, 0],          # H-like (S gate with theta=pi/2, phi=0, lam=0)
            ["u3", 1, math.pi/2, 0, 0],          # H-like
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0, 0],
            ["u3", 1, math.pi/2, 0, 0],
            ["u3", 0, math.pi/2, 0, -math.pi/2], # S-dag on q0
            ["u3", 1, math.pi/2, 0, -math.pi/2], # S-dag on q1
        ]
        return gates
    
    if name == "SWAP" and qubits == 2:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    
    if name == "Toffoli" and qubits == 3:
        # Toffoli decomposition with 6 CNOTs
        # Standard decomposition from Nielsen & Chuang
        gates = [
            ["u3", 2, math.pi/2, 0, 0],       # H on q2
            ["cx", 1, 2],
            ["u3", 2, math.pi/4, 0, 0],       # T on q2
            ["cx", 0, 2],
            ["u3", 2, -math.pi/4, 0, 0],      # T-dag on q2
            ["cx", 1, 2],
            ["u3", 2, math.pi/4, 0, 0],       # T on q2
            ["cx", 0, 2],
            ["u3", 2, -math.pi/4, 0, 0],      # T-dag on q2
            ["u3", 1, math.pi/2, 0, 0],       # H on q1
            ["cx", 1, 2],
            ["u3", 2, math.pi/2, 0, 0],       # H on q2
        ]
        return gates
    
    return None


def _coordinate_descent(objective, x0, n_iter=1000, tol=1e-12, timeout=None):
    """Simple coordinate descent optimizer."""
    n = len(x0)
    x = list(x0)
    fx = objective(x)
    
    step_sizes = [0.5] * n
    min_steps = [1e-10] * n
    
    for it in range(n_iter):
        if timeout is not None and time.time() > timeout:
            break
        
        improved = False
        for i in range(n):
            # Try positive step
            x[i] = x[i] + step_sizes[i]
            fx_new = objective(x)
            if fx_new < fx:
                fx = fx_new
                improved = True
                step_sizes[i] = min(step_sizes[i] * 1.5, 1.0)
            else:
                x[i] = x[i] - step_sizes[i]
                fx_new = objective(x)
                if fx_new < fx:
                    fx = fx_new
                    improved = True
                    step_sizes[i] = min(step_sizes[i] * 1.5, 1.0)
                else:
                    x[i] = x[i] - step_sizes[i]
                    step_sizes[i] = max(step_sizes[i] * 0.5, min_steps[i])
        
        if not improved:
            break
        if fx < tol:
            break
    
    return x, fx


def synthesize(problem):
    """Synthesize quantum circuit with known decompositions and optimization for generic gates."""
    name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decomposition first
    known = _known_decomposition(name, qubits, budget)
    if known is not None:
        if len(known) <= 400:
            cnot_count = sum(1 for g in known if g[0] == "cx")
            if cnot_count <= budget:
                dist = _distance(target, _build_fast(known, qubits))
                if dist < 1e-10:
                    return known
    
    # For generic unitaries, use coordinate descent optimization
    # Build a layout with the given CNOT budget
    if qubits == 2:
        pairs = [(0, 1)]
    else:
        # Use a balanced CNOT pattern
        pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    # Create layout: layers of CNOTs with u3 gates around them
    layout = []
    n_u3_per_layer = qubits
    
    for layer in range(budget):
        # Add u3 gates for all qubits before CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        # Add CNOT
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    
    # Final u3 gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Count parameters
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initial values
    best = [0.0] * n_params
    
    def circuit_from_values(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(values):
        gates = circuit_from_values(values)
        if len(gates) > 400:
            return 1.0
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count > budget:
            return 1.0
        matrix = _build_fast(gates, qubits)
        return _distance(target, matrix)
    
    # Coordinate descent optimization
    best_values, best_dist = _coordinate_descent(objective, best, n_iter=500, tol=1e-14, timeout=_DEADLINE)
    
    return circuit_from_values(best_values)
# EVOLVE-BLOCK-END
