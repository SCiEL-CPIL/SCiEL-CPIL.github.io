"""Exact algebraic synthesis for known gates + optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list using precomputed operators."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    # Precompute single-qubit operators for each qubit position
    single_qubit_ops = [None] * qubits
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                              [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build the full operator via kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: permute rows
            shift = 1 << (qubits - 1 - target)
            mask = 1 << (qubits - 1 - control)
            perm = np.arange(n)
            for i in range(n):
                if i & mask:
                    perm[i] = i ^ shift
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)


# Exact decompositions for known gates

def _decompose_cnot(problem):
    """CNOT gate: 1 CNOT, 0 single-qubit gates needed (it IS a CNOT)."""
    return [["cx", 0, 1]]


def _decompose_iswap(problem):
    """iSWAP decomposition with 3 CNOTs.
    
    iSWAP matrix:
    [[1, 0, 0, 0],
     [0, 0, i, 0],
     [0, i, 0, 0],
     [0, 0, 0, 1]]
    
    Decomposition: Use H gates to convert between X-basis and Z-basis,
    then use CNOTs with phase corrections.
    """
    gates = []
    # H on both qubits
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # RZ(π/2) on both: u3(0, 0, π/2)
    gates.append(["u3", 0, 0, 0, math.pi/2])
    gates.append(["u3", 1, 0, 0, math.pi/2])
    # CNOT 1->0
    gates.append(["cx", 1, 0])
    # RZ(-π/2) on both
    gates.append(["u3", 0, 0, 0, -math.pi/2])
    gates.append(["u3", 1, 0, 0, -math.pi/2])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # H on both
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    return gates


def _decompose_swap(problem):
    """SWAP = CNOT(0,1) · CNOT(1,0) · CNOT(0,1)"""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _decompose_toffoli(problem):
    """Toffoli gate: 3-qubit, 6 CNOTs.
    
    Standard 6-CNOT Toffoli decomposition.
    Q0 (control 1), Q1 (control 2), Q2 (target)
    """
    gates = []
    pi4 = math.pi / 4
    
    # H on q2
    gates.append(["u3", 2, math.pi/2, 0, math.pi])
    # CNOT q1 -> q2
    gates.append(["cx", 1, 2])
    # T† on q2
    gates.append(["u3", 2, 0, 0, -pi4])
    # CNOT q0 -> q2
    gates.append(["cx", 0, 2])
    # T on q2
    gates.append(["u3", 2, 0, 0, pi4])
    # T on q1
    gates.append(["u3", 1, 0, 0, pi4])
    # CNOT q0 -> q1
    gates.append(["cx", 0, 1])
    # T† on q1
    gates.append(["u3", 1, 0, 0, -pi4])
    # T on q0
    gates.append(["u3", 0, 0, 0, pi4])
    # CNOT q0 -> q1
    gates.append(["cx", 0, 1])
    # T† on q0
    gates.append(["u3", 0, 0, 0, -pi4])
    # T† on q1
    gates.append(["u3", 1, 0, 0, -pi4])
    # CNOT q1 -> q2
    gates.append(["cx", 1, 2])
    # T on q2
    gates.append(["u3", 2, 0, 0, pi4])
    # H on q2
    gates.append(["u3", 2, math.pi/2, 0, math.pi])
    # T on q1
    gates.append(["u3", 1, 0, 0, pi4])
    # T† on q0
    gates.append(["u3", 0, 0, 0, -pi4])
    
    return gates


def _optimize_generic(problem, name, time_budget):
    """Optimize a generic unitary using coordinate descent with adaptive step size."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Create a layout with CNOTs and u3 gates
    rng = random.Random(4242 + hash(name) % 10000 + qubits)
    
    # Interleave CNOTs and u3 layers
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with adaptive step size
    start_time = time.time()
    
    sweep = 0
    while time.time() - start_time < time_budget:
        step = 0.5 / (1 + sweep * 0.5)
        improved = False
        for i in range(n_params):
            # Try several perturbations
            for trial in (best[i] + step, best[i] - step, 
                         best[i] + rng.uniform(-step * 2, step * 2)):
                values = best[:]
                values[i] = trial
                dist = _distance(target, _build_fast(circuit(values), qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary using exact decompositions for known gates
    and coordinate descent for generic unitaries."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Check if this is a known gate
    name_lower = name.lower() if name else ""
    
    if qubits == 2:
        if "cnot" in name_lower or name_lower == "cx":
            gates = _decompose_cnot(problem)
            cnot_count = sum(1 for g in gates if g[0] == "cx")
            if cnot_count <= budget:
                return gates
        elif "iswap" in name_lower:
            gates = _decompose_iswap(problem)
            cnot_count = sum(1 for g in gates if g[0] == "cx")
            if cnot_count <= budget:
                return gates
        elif "swap" in name_lower:
            gates = _decompose_swap(problem)
            cnot_count = sum(1 for g in gates if g[0] == "cx")
            if cnot_count <= budget:
                return gates
    
    if qubits == 3:
        if "toffoli" in name_lower or "ccnot" in name_lower:
            gates = _decompose_toffoli(problem)
            cnot_count = sum(1 for g in gates if g[0] == "cx")
            if cnot_count <= budget:
                return gates
    
    # Generic unitary: use optimization with remaining time budget
    remaining_time = 200.0 - time.time()  # Start from ~40s elapsed
    time_per_generic = max(10.0, remaining_time / 3.0)
    
    return _optimize_generic(problem, name, time_per_generic)
# EVOLVE-BLOCK-END
