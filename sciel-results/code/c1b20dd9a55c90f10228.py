"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _build_from_params(qubits, params, cnots):
    """Build unitary from single-qubit rotation params and CNOT list.
    params: list of (theta, phi, lam) per single-qubit slot, in order of appearance.
    cnots: list of (control, target) CNOTs, interleaved with single-qubit slots.
    The layout is: for each CNOT in cnots, we have a layer of single-qubit gates before it,
    and a final layer after all CNOTs.
    """
    n_sq = len(params)
    n_cnots = len(cnots)
    # Layout: n_sq single-qubit slots and n_cnots CNOTs
    # The structure is: SQ layer, CNOT, SQ layer, CNOT, ..., SQ layer
    # So there are n_cnots + 1 SQ layers, each with 'qubits' gates
    # Total SQ slots = (n_cnots + 1) * qubits
    # But n_sq should equal (n_cnots + 1) * qubits
    
    # Build gate list
    gates = []
    sq_idx = 0
    for i in range(n_cnots):
        for q in range(qubits):
            theta, phi, lam = params[sq_idx]
            gates.append(['u3', q, theta, phi, lam])
            sq_idx += 1
        gates.append(['cx', cnots[i][0], cnots[i][1]])
    for q in range(qubits):
        theta, phi, lam = params[sq_idx]
        gates.append(['u3', q, theta, phi, lam])
        sq_idx += 1
    return gates


def _decompose_two_qubit(target):
    """Decompose a 2-qubit unitary into single-qubit gates and CNOTs using KAK decomposition.
    Returns a list of gates with 3 CNOTs.
    """
    from scipy.linalg import schur, eig
    
    # KAK decomposition for 2-qubit gates
    # U = A0 (CNOT) A1 (CNOT) A2 (CNOT) A3
    # where Ai are single-qubit gates on both qubits
    
    # Use a simpler approach: try to fit with 3 CNOTs
    # This is a known decomposition but implementing full KAK is complex.
    # Instead, use a numerical optimization approach.
    
    qubits = 2
    budget = 3
    
    # Layout: SQ layer, CNOT, SQ layer, CNOT, SQ layer, CNOT, SQ layer
    # 4 layers of 2 single-qubit gates = 8 single-qubit gates = 24 parameters
    # 3 CNOTs
    
    cnots = [(0, 1), (1, 0), (0, 1)]
    n_params = 4 * qubits * 3  # 4 layers * 2 qubits * 3 params
    
    rng = random.Random(12345)
    best_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    def eval_params(params):
        # Reshape params into (theta, phi, lam) tuples
        tuples = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(len(params)//3)]
        gates = _build_from_params(qubits, tuples, cnots)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    best_dist = eval_params(best_params)
    
    # Coordinate descent
    for sweep in range(20):
        step = 0.5 / (1 + sweep * 0.5)
        improved = False
        for i in range(len(best_params)):
            for trial in (best_params[i] + step, best_params[i] - step):
                params = best_params[:]
                params[i] = trial
                dist = eval_params(params)
                if dist < best_dist:
                    best_params = params
                    best_dist = dist
                    improved = True
        if not improved and sweep > 5:
            break
    
    # Extract gates
    tuples = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(len(best_params)//3)]
    return _build_from_params(qubits, tuples, cnots)


def _decompose_three_qubit_toffoli():
    """Decompose Toffoli gate into 7 CNOTs and single-qubit gates.
    Known decomposition uses 7 CNOTs.
    """
    qubits = 3
    # Toffoli: |110> -> |111>, |111> -> |110>
    # Standard decomposition:
    # H on target, then sequence of CNOTs and single-qubit gates
    
    # Use a known 7-CNOT decomposition
    # This is the standard decomposition from Barenco et al.
    gates = []
    
    # Single-qubit gates
    h = math.pi / 2
    # T gate: pi/8
    t = math.pi / 8
    
    # The standard 7-CNOT Toffoli decomposition:
    # Q0: H, T, T, T, T, T, T, T, H
    # Q1: T, T, T, T, T, T, T, T
    # Q2: (various)
    
    # Let me use a different approach: numerical optimization
    # with 7 CNOTs
    
    # Layout with 7 CNOTs
    cnots = [(0, 2), (1, 2), (0, 2), (1, 2), (0, 2), (1, 2), (0, 2)]
    n_params = 8 * qubits * 3  # 8 layers * 3 qubits * 3 params
    
    rng = random.Random(99999)
    best_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    def eval_params(params):
        tuples = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(len(params)//3)]
        gates = _build_from_params(qubits, tuples, cnots)
        matrix = _build(gates, qubits)
        return _distance(target_3q, matrix)
    
    global target_3q
    # We'll set this in synthesize
    
    best_dist = eval_params(best_params)
    
    for sweep in range(15):
        step = 0.5 / (1 + sweep * 0.5)
        improved = False
        for i in range(len(best_params)):
            for trial in (best_params[i] + step, best_params[i] - step):
                params = best_params[:]
                params[i] = trial
                dist = eval_params(params)
                if dist < best_dist:
                    best_params = params
                    best_dist = dist
                    improved = True
        if not improved and sweep > 5:
            break
    
    tuples = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(len(best_params)//3)]
    return _build_from_params(qubits, tuples, cnots)


def _generic_decompose(target, qubits, budget):
    """Generic decomposition using numerical optimization."""
    # Use budget CNOTs
    # For 2 qubits: use up to 3 CNOTs
    # For 3 qubits: use up to budget CNOTs
    
    if qubits == 2:
        cnots = [(0, 1), (1, 0), (0, 1)][:budget]
    else:
        # For 3 qubits, use a reasonable CNOT pattern
        cnots = []
        pairs = [(0, 1), (1, 2), (0, 2)]
        for i in range(budget):
            cnots.append(pairs[i % len(pairs)])
    
    n_layers = len(cnots) + 1
    n_params = n_layers * qubits * 3
    
    rng = random.Random(77777 + qubits * 100 + budget)
    best_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    def eval_params(params):
        tuples = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(len(params)//3)]
        gates = _build_from_params(qubits, tuples, cnots)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    best_dist = eval_params(best_params)
    
    # Coordinate descent with multiple sweeps
    for sweep in range(25):
        step = 0.8 / (1 + sweep * 0.3)
        improved = False
        for i in range(len(best_params)):
            for trial in (best_params[i] + step, best_params[i] - step, 
                         best_params[i] + rng.uniform(-math.pi, math.pi) * 0.1):
                params = best_params[:]
                params[i] = trial
                dist = eval_params(params)
                if dist < best_dist:
                    best_params = params
                    best_dist = dist
                    improved = True
        if not improved and sweep > 10:
            break
    
    tuples = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(len(best_params)//3)]
    return _build_from_params(qubits, tuples, cnots)


target_3q = None

def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    global target_3q
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    if qubits == 2:
        # For 2-qubit gates, use up to 3 CNOTs
        actual_budget = min(budget, 3)
        if actual_budget == 3:
            return _decompose_two_qubit(target)
        else:
            return _generic_decompose(target, qubits, actual_budget)
    else:
        # For 3-qubit gates
        target_3q = target
        return _generic_decompose(target, qubits, budget)
# EVOLVE-BLOCK-END
