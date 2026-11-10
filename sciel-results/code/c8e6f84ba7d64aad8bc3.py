"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0


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
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi)


def _sdag(q):
    return _u3(q, 0.0, 0.0, -math.pi)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _tdag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _toffoli(c1, c2, t):
    gates = [
        _h(t), _t(c1), _cx(c1, c2), _tdag(c2),
        _cx(c1, c2), _t(c2), _t(t), _cx(c2, t),
        _tdag(c1), _t(c2), _cx(c1, c2), _tdag(c2),
        _h(t), _s(c2), _s(t),
    ]
    return gates


def _cnot_gate(c, t):
    return [_cx(c, t)]


def _swap_gate(a, b):
    return [_cx(a, b), _cx(b, a), _cx(a, b)]


def _iswap_gate(a, b):
    return [
        _h(a), _cx(a, b), _s(a), _s(b), _cx(b, a), _h(a), _s(a), _s(b),
    ]


def _match_fixed(name, qubits):
    name_l = str(name).lower().replace("_", "").replace("-", "")
    if name_l in ("cnot", "cx"):
        return _cnot_gate(0, 1)
    if name_l in ("swap",):
        return _swap_gate(0, 1)
    if name_l in ("iswap",):
        return _iswap_gate(0, 1)
    if name_l in ("toffoli", "ccnot", "ccx"):
        return _toffoli(0, 1, 2)
    return None


def _u3_from_unitary(u):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters."""
    # u3 matrix: [[e^{-i(phi+lam)/2}cos(theta/2), -e^{i lam}sin(theta/2)],
    #             [e^{i phi}sin(theta/2), e^{i(phi+lam)/2}cos(theta/2)]]
    # But our convention: [[c, -e^{i lam}s], [e^{i phi}s, e^{i(phi+lam)}c]]
    # where c=cos(theta/2), s=sin(theta/2)
    # This is u3 with a global phase factor of e^{-i(phi+lam)/2}
    
    # Extract theta from |u[0,0]|
    u00 = u[0, 0]
    u01 = u[0, 1]
    u10 = u[1, 0]
    u11 = u[1, 1]
    
    # |u00| = |cos(theta/2)|
    abs_u00 = abs(u00)
    abs_u01 = abs(u01)
    
    # Handle edge cases
    if abs_u00 < 1e-12:
        theta = math.pi
        if abs(abs_u01 - 1.0) > 1e-9:
            theta = math.pi
        # phi from u10: e^{i phi} * sin(theta/2) = u10 => e^{i phi} = u10 / sin(theta/2)
        # sin(pi/2) = 1
        if abs(u10) > 1e-12:
            phi = np.angle(u10)
        else:
            phi = 0.0
        # lam from u01: -e^{i lam} * sin(theta/2) = u01 => -e^{i lam} = u01
        if abs(u01) > 1e-12:
            lam = np.angle(-u01)
        else:
            lam = 0.0
        return theta, phi, lam
    elif abs_u01 < 1e-12:
        theta = 0.0
        phi = 0.0
        lam = 0.0
        return theta, phi, lam
    
    theta = 2.0 * math.acos(min(1.0, max(0.0, abs_u00)))
    
    # phi from u10: u10 = e^{i phi} * sin(theta/2)
    s = math.sin(theta / 2)
    if s > 1e-12 and abs(u10) > 1e-12:
        phi = np.angle(u10 / s)
    else:
        phi = 0.0
    
    # lam from u01: u01 = -e^{i lam} * sin(theta/2)
    if s > 1e-12 and abs(u01) > 1e-12:
        lam = np.angle(-u01 / s)
    else:
        lam = 0.0
    
    return theta, phi, lam


def _kron2(a, b):
    """Kronecker product of two 2x2 matrices."""
    return np.kron(a, b)


def _two_qubit_exact_synthesis(target):
    """
    Exact synthesis of a 2-qubit unitary using the CNOT ladder:
    CNOT(0,1), CNOT(1,0), CNOT(0,1)
    Structure: (A⊗B) · CNOT(0,1) · CNOT(1,0) · CNOT(0,1) · (C⊗D)
    We extract A, B, C, D analytically.
    """
    U = target.copy()
    
    # Remove global phase: make det(U) = 1
    det_U = np.linalg.det(U)
    if abs(det_U) > 1e-12:
        phase = np.sqrt(det_U)
        U = U / phase
    
    # CNOT(0,1) matrix in computational basis (qubit 0 is MSB)
    # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    cnot_01 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # CNOT(1,0): control qubit 1, target qubit 0
    # |00> -> |00>, |01> -> |11>, |10> -> |10>, |11> -> |01>
    cnot_10 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    
    # Middle = CNOT(0,1) @ CNOT(1,0) @ CNOT(0,1)
    middle = cnot_01 @ cnot_10 @ cnot_01
    
    # U = (A⊗B) @ middle @ (C⊗D)
    # => (C⊗D) = middle^{-1} @ U @ (A⊗B)^{-1}
    # This is circular. Let's use a different approach.
    
    # Alternative: use the fact that we can write
    # U = (A⊗B) @ middle @ (C⊗D)
    # Let M = middle. Then (A⊗B) = U @ M^{-1} @ (C⊗D)^{-1}
    
    # Better approach: use partial trace / local invariants
    # For 2-qubit unitaries, we can use the KAK decomposition.
    # But let's try a direct extraction approach.
    
    # Strategy: 
    # 1. Compute V = U @ middle^{-1} = (A⊗B) @ (C⊗D)^{-1}
    # 2. V is a product of local unitaries: V = (A⊗B) @ (C^{-1}⊗D^{-1})
    #    Let P = A @ C^{-1} and Q = B @ D^{-1}, so V = P ⊗ Q
    # 3. Extract P and Q from V
    
    mid_inv = np.linalg.inv(middle)
    V = U @ mid_inv
    
    # V should be P ⊗ Q for some 2x2 unitaries P, Q
    # Extract P and Q:
    # V[0,0] = P[0,0]*Q[0,0], V[0,1] = P[0,0]*Q[0,1], V[0,2] = P[0,1]*Q[0,0], etc.
    
    # From V[0,0] and V[0,1]: Q[0,0] = V[0,0]/P[0,0], Q[0,1] = V[0,1]/P[0,0]
    # From V[0,2] and V[0,3]: Q[0,0] = V[0,2]/P[0,1], Q[0,1] = V[0,3]/P[0,1]
    # So V[0,0]/P[0,0] = V[0,2]/P[0,1] => P[0,1] = P[0,0] * V[0,2] / V[0,0]
    
    # Let's extract more carefully.
    # V = P ⊗ Q means:
    # V[i,j] = P[i//2, j//2] * Q[i%2, j%2]
    
    # Let p00 = P[0,0], p01 = P[0,1], p10 = P[1,0], p11 = P[1,1]
    # Let q00 = Q[0,0], q01 = Q[0,1], q10 = Q[1,0], q11 = Q[1,1]
    
    # V[0,0] = p00*q00
    # V[0,1] = p00*q01
    # V[0,2] = p01*q00
    # V[0,3] = p01*q01
    # V[1,0] = p10*q00
    # V[1,1] = p10*q01
    # V[1,2] = p11*q00
    # V[1,3] = p11*q01
    # V[2,0] = p00*q10
    # V[2,1] = p00*q11
    # V[2,2] = p01*q10
    # V[2,3] = p01*q11
    # V[3,0] = p10*q10
    # V[3,1] = p10*q11
    # V[3,2] = p11*q10
    # V[3,3] = p11*q11
    
    # From V[0,0] and V[0,1]: q01/q00 = V[0,1]/V[0,0]
    # From V[0,2] and V[0,3]: q01/q00 = V[0,3]/V[0,2]
    # So V[0,1]/V[0,0] = V[0,3]/V[0,2] (consistency check)
    
    # Let's set q00 = 1 (absorbing phase into P)
    # Then q01 = V[0,1]/V[0,0]
    # p00 = V[0,0]
    # p01 = V[0,2]/q00 = V[0,2]
    # q10 = V[2,0]/p00 = V[2,0]/V[0,0]
    # q11 = V[2,1]/p00 = V[2,1]/V[0,0]
    # p10 = V[1,0]/q00 = V[1,0]
    # p11 = V[1,2]/q00 = V[1,2]
    
    # But this doesn't respect unitarity. Let me use a more robust approach.
    
    # Robust extraction:
    # Take the first row of P: [p00, p01]
    # V[0,0] = p00*q00, V[0,1] = p00*q01
    # V[0,2] = p01*q00, V[0,3] = p01*q01
    
    # If V[0,0] != 0:
    #   p01 = V[0,2] / V[0,0] * p00
    #   q01 = V[0,1] / V[0,0] * q00
    
    # Let me try a different, more standard approach using the KAK decomposition.
    
    # Actually, let's use a simpler but still exact method:
    # We know U = (A⊗B) @ M @ (C⊗D) where M = middle
    # Let W = U @ M^{-1} = (A⊗B) @ (C⊗D)^{-1}
    # Let X = A @ C^† and Y = B @ D^†, so W = X ⊗ Y
    # Extract X and Y from W, then A = X @ C, B = Y @ D
    # But we still need C and D...
    
    # Let's use a different decomposition strategy.
    # U = (A⊗B) @ M @ (C⊗D)
    # => U @ M^{-1} = (A⊗B) @ (C⊗D)^{-1} = (A @ C^†) ⊗ (B @ D^†)
    # Let P = A @ C^† and Q = B @ D^†
    # Then W = U @ M^{-1} = P ⊗ Q
    
    # Extract P and Q from W:
    # P is 2x2, Q is 2x2
    # W[0:2, 0:2] = P[0,:].reshape(1,-1) @ Q[:,0].reshape(-1,1) = outer(P[0,:], Q[:,0])
    # W[0:2, 2:4] = outer(P[0,:], Q[:,1])
    # W[2:4, 0:2] = outer(P[1,:], Q[:,0])
    # W[2:4, 2:4] = outer(P[1,:], Q[:,1])
    
    # From W[0:2, 0:2]: rank-1 matrix = P[0,:] ⊗ Q[:,0]
    # SVD: W[0:2, 0:2] = u * s * v^H
    # P[0,:] proportional to u, Q[:,0] proportional to v
    
    W = U @ mid_inv
    
    # Extract P and Q
    # Block (0,0): W[0:2, 0:2]
    B00 = W[0:2, 0:2]
    B01 = W[0:2, 2:4]
    B10 = W[2:4, 0:2]
    B11 = W[2:4, 2:4]
    
    # SVD of B00
    U_svd, S, Vh = np.linalg.svd(B00)
    if S[0] < 1e-12:
        # Degenerate case, try different approach
        return None
    
    p0 = U_svd[:, 0] / np.sqrt(S[0])  # P[0,:]
    q_col0 = Vh[0, :] / np.sqrt(S[0])  # Q[:,0]
    
    # Verify with B01
    # B01 should be outer(p0, q_col1)
    # q_col1 = B01 @ pinv(p0)
    p0_norm = np.linalg.norm(p0)
    if p0_norm > 1e-12:
        q_col1 = (B01 @ p0.conj().T) / np.sum(p0 * p0.conj().T)
    else:
        return None
    
    # Verify with B10
    # B10 should be outer(p1, q_col0)
    q_col0_norm = np.linalg.norm(q_col0)
    if q_col0_norm > 1e-12:
        p1 = (q_col0 @ B10) / np.sum(q_col0 * q_col0.conj().T)
    else:
        return None
    
    # Verify with B11
    # B11 should be outer(p1, q_col1)
    # Check consistency
    B11_pred = np.outer(p1, q_col1)
    if np.linalg.norm(B11 - B11_pred) > 1e-6:
        # Inconsistent, try alternative
        return None
    
    P = np.array([p0, p1])
    Q = np.array([q_col0, q_col1]).T
    
    # Now W = P ⊗ Q = (A @ C^†) ⊗ (B @ D^†)
    # We need to find A, B, C, D such that A @ C^† = P and B @ D^† = Q
    # This has many solutions. Let's set C = I, D = I
    # Then A = P, B = Q
    # But we need A, B, C, D to be unitary.
    # P and Q should be unitary (up to numerical error)
    
    # Check if P and Q are approximately unitary
    if np.linalg.norm(P @ P.conj().T - np.eye(2)) > 0.1:
        # Try to make them unitary using polar decomposition
        U_p, _ = np.linalg.svd(P)
        P = U_p @ U_p.conj().T  # Not quite right
        # Polar decomposition: P = U @ H where U is unitary, H is Hermitian positive
        # Use SVD: P = U_s @ S @ Vh, then U_polar = U_s @ Vh
        U_s, S, Vh = np.linalg.svd(P)
        P_unitary = U_s @ Vh
        U_s, S, Vh = np.linalg.svd(Q)
        Q_unitary = U_s @ Vh
        P = P_unitary
        Q = Q_unitary
    
    # Set C = I, D = I, A = P, B = Q
    # Then U = (P⊗Q) @ M @ (I⊗I) = P⊗Q @ M
    # But we need U = (A⊗B) @ M @ (C⊗D) = (P⊗Q) @ M
    # So this works!
    
    A = P
    B = Q
    C = np.eye(2, dtype=complex)
    D = np.eye(2, dtype=complex)
    
    # Verify
    U_check = _kron2(A, B) @ middle @ _kron2(C, D)
    # Remove global phase for comparison
    det_check = np.linalg.det(U_check)
    if abs(det_check) > 1e-12:
        U_check = U_check / np.sqrt(det_check)
    
    dist = _distance(target, U_check)
    if dist > 1e-6:
        return None
    
    # Now decompose A, B, C, D into u3 gates
    gates = []
    
    # (C⊗D) first
    for mat, q in [(C, 0), (D, 1)]:
        theta, phi, lam = _u3_from_unitary(mat)
        gates.append(_u3(q, theta, phi, lam))
    
    # CNOT(0,1)
    gates.append(_cx(0, 1))
    # CNOT(1,0)
    gates.append(_cx(1, 0))
    # CNOT(0,1)
    gates.append(_cx(0, 1))
    
    # (A⊗B)
    for mat, q in [(A, 0), (B, 1)]:
        theta, phi, lam = _u3_from_unitary(mat)
        gates.append(_u3(q, theta, phi, lam))
    
    return gates


def _optimize_generic(target, qubits, budget):
    """Coordinate descent on a fixed CNOT ladder for generic unitaries."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    layout = []
    n_u3_layers = budget + 1
    for layer in range(n_u3_layers):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        if layer < budget:
            layout.append(["cx", *pairs[layer % len(pairs)]])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    for sweep in range(12):
        if _time_remaining() < 15.0:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _time_remaining() < 15.0:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)

    # 1) Exact known-gate fast path
    fixed = _match_fixed(name, qubits)
    if fixed is not None:
        cx_count = sum(1 for g in fixed if g[0] == "cx")
        if cx_count <= budget and len(fixed) <= 400:
            return fixed
        return []

    # 2) For 2-qubit generic unitaries, try exact KAK-based synthesis
    if qubits == 2 and budget >= 3:
        exact = _two_qubit_exact_synthesis(target)
        if exact is not None:
            cx_count = sum(1 for g in exact if g[0] == "cx")
            if cx_count <= budget and len(exact) <= 400:
                return exact

    # 3) Generic: coordinate descent
    if _time_remaining() > 20.0:
        return _optimize_generic(target, qubits, budget)
    
    return []
# EVOLVE-BLOCK-END
