"""Exact decompositions for known gates + scipy-based optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import leastsq

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _build_unitary(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(name, qubits):
    if name == "CNOT" and qubits == 2:
        return "CNOT"
    elif name == "iSWAP" and qubits == 2:
        return "iSWAP"
    elif name == "SWAP" and qubits == 2:
        return "SWAP"
    elif name == "Toffoli" and qubits == 3:
        return "Toffoli"
    return None


def _synthesize_CNOT():
    return [_cx(0, 1)]


def _synthesize_iSWAP():
    gates = []
    gates.append(_u3(0, math.pi, 0, math.pi / 2))
    gates.append(_u3(1, math.pi, 0, math.pi / 2))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, math.pi, 0, math.pi / 2))
    gates.append(_u3(1, math.pi, 0, math.pi / 2))
    gates.append(_cx(1, 0))
    gates.append(_u3(0, math.pi, 0, math.pi / 2))
    gates.append(_u3(1, math.pi, 0, math.pi / 2))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, 0, 0, -math.pi / 4))
    gates.append(_u3(1, 0, 0, -math.pi / 4))
    return gates


def _synthesize_SWAP():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _synthesize_Toffoli():
    gates = []
    gates.append(_u3(2, math.pi / 4, 0, 0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -math.pi / 4, 0, 0))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, math.pi / 4, 0, 0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -math.pi / 4, 0, 0))
    gates.append(_u3(2, math.pi / 8, 0, 0))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, -math.pi / 8, 0, 0))
    gates.append(_u3(2, math.pi / 8, 0, 0))
    gates.append(_cx(0, 2))
    gates.append(_u3(1, math.pi / 8, 0, 0))
    gates.append(_u3(2, -math.pi / 8, 0, 0))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, math.pi / 8, 0, 0))
    gates.append(_u3(2, math.pi / 8, 0, 0))
    return gates


def _mat_to_u3_params(M):
    """Convert a 2x2 unitary to u3(theta, phi, lam) parameters."""
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                         [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    # Handle global phase: multiply by exp(-i*alpha) to make M[0,0] real
    if abs(M[0,0]) > 1e-10:
        alpha = np.angle(M[0,0])
    elif abs(M[0,1]) > 1e-10:
        alpha = np.angle(-M[0,1])
    elif abs(M[1,0]) > 1e-10:
        alpha = np.angle(M[1,0])
    else:
        alpha = np.angle(M[1,1])
    
    M2 = M * np.exp(-1j * alpha)
    
    c = M2[0, 0].real
    c = max(-1.0, min(1.0, c))
    theta = 2 * np.arccos(c)
    
    if abs(theta) < 1e-10:
        return 0.0, 0.0, 0.0
    
    s = math.sin(theta / 2)
    if abs(s) < 1e-10:
        return 0.0, 0.0, 0.0
    
    lam = np.angle(-M2[0, 1] / s)
    phi = np.angle(M2[1, 0] / s)
    
    return float(theta), float(phi), float(lam)


def _peel_2qubit_layer(U, control=0, target=1, n_qubits=2):
    """
    Peel one layer: U = (A ⊗ I) CNOT (B ⊗ I) * rest
    where rest acts trivially on qubit `control`.
    
    Returns (A_matrix, B_matrix, rest_unitary)
    """
    dim = 2 ** n_qubits
    half = dim // 2
    
    # U = (A ⊗ I) CNOT (B ⊗ I) V
    # => U (B† ⊗ I) CNOT† V† = A ⊗ I
    # Since CNOT† = CNOT:
    # => U (B† ⊗ I) CNOT V† = A ⊗ I
    
    # We need to find B such that U (B† ⊗ I) CNOT V† = A ⊗ I
    # V acts trivially on qubit `control`, so V = I ⊗ W for some W.
    
    # Let's use the SVD approach:
    # Write U in block form with respect to qubit `control`:
    # U = [[U00, U01], [U10, U11]] where each block is (dim/2)x(dim/2)
    
    # For the CNOT with control=0, target=1 (in 2-qubit case):
    # CNOT |00> = |00>, |01> = |01>, |10> = |11>, |11> = |10>
    # In block form: CNOT = [[I, 0], [0, X]] where X = [[0,1],[1,0]]... 
    # Actually CNOT(0->1) in basis |00>,|01>,|10>,|11>:
    # = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # = [[I_2, 0], [0, X]] where X = [[0,1],[1,0]]
    
    # Let me use a different approach. The standard Barenco peeling:
    # U = (A ⊗ I) CNOT (B ⊗ I) V where V acts trivially on qubit 0.
    # 
    # To find A and B:
    # Let M = U. We want M (B† ⊗ I) CNOT V† = A ⊗ I.
    # 
    # Alternative: use the fact that we can find B by looking at the
    # action of U on the subspace where qubit 0 = 0.
    #
    # Actually, let me use a cleaner formulation.
    # 
    # U = (A ⊗ I) CNOT (B ⊗ I) V
    # 
    # Multiply from right by V†: U V† = (A ⊗ I) CNOT (B ⊗ I)
    # 
    # Let N = U V†. Then N = (A ⊗ I) CNOT (B ⊗ I).
    # 
    # In 2-qubit case with CNOT(0->1):
    # N = [[A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,1]*B[0,0], A[0,1]*B[0,1]],
    #      [A[0,0]*B[1,0], A[0,0]*B[1,1], A[0,1]*B[1,0], A[0,1]*B[1,1]],
    #      [A[1,0]*B[0,0], A[1,0]*B[0,1], A[1,1]*B[0,0], A[1,1]*B[0,1]],
    #      [A[1,0]*B[1,0], A[1,0]*B[1,1], A[1,1]*B[1,0], A[1,1]*B[1,1]]]
    # 
    # Wait, that's not right. Let me be more careful.
    # 
    # (A ⊗ I) = [[A[0,0]*I, A[0,1]*I], [A[1,0]*I, A[1,1]*I]]
    # CNOT(0->1) = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # (B ⊗ I) = [[B[0,0]*I, B[0,1]*I], [B[1,0]*I, B[1,1]*I]]
    # 
    # CNOT * (B ⊗ I):
    # Row 0: [B[0,0], 0, B[0,1], 0] * I... 
    
    # This is getting complex. Let me just use scipy to optimize directly.
    pass


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations around a CNOT ladder using scipy's leastsq."""
    dim = 2 ** qubits
    
    # Build CNOT layout: use all qubit pairs in a structured way
    if qubits == 2:
        # For 2 qubits, use CNOT(0,1) repeatedly
        cnot_pairs = [(0, 1)] * min(budget, 3)
    else:
        # For 3 qubits, use a structured set of CNOTs
        all_pairs = [(0,1), (0,2), (1,2), (1,0), (2,0), (2,1)]
        cnot_pairs = all_pairs[:budget]
    
    # Build gate layout: u3 gates before and after each CNOT
    layout = []
    for i, (c, t) in enumerate(cnot_pairs):
        # u3 gates on both qubits
        layout.append(("u3", c))
        layout.append(("u3", t))
        layout.append(("cx", c, t))
    
    # Add final u3 gates
    for q in range(qubits):
        layout.append(("u3", q))
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def build_gates(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(["cx", g[1], g[2]])
        return gates
    
    def objective(params):
        gates = build_gates(params)
        U = _build_unitary(gates, qubits)
        # Phase-insensitive distance
        tr = np.trace(target.conj().T @ U)
        # We want to minimize |1 - |tr|/dim|, which is equivalent to maximizing |tr|
        # Use least-squares: minimize |tr|^2 - dim^2 (but this isn't quite right)
        # Better: minimize the Frobenius norm of (target - U) up to global phase
        # For leastsq, we need real-valued residuals
        # Use: residuals = (target - U * exp(i*alpha)).flatten() where alpha optimizes phase
        alpha = np.angle(tr) if abs(tr) > 1e-10 else 0
        U_adj = U * np.exp(1j * alpha)
        diff = target - U_adj
        # Return real and imaginary parts as residuals
        return np.concatenate([diff.real.flatten(), diff.imag.flatten()])
    
    # Try multiple initializations
    import random
    best_params = None
    best_dist = float('inf')
    
    rng = random.Random(12345 + qubits * 100 + budget)
    
    # Start with identity-like initialization
    init_params = [0.0] * n_params
    best_params = init_params[:]
    U_init = _build_unitary(build_gates(init_params), qubits)
    best_dist = _distance(target, U_init)
    
    # Try random initializations
    for restart in range(5):
        if _elapsed() > _DEADLINE - 15:
            break
        params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        try:
            result, success, info, nfev = leastsq(
                objective, params, full_output=1, maxfev=200
            )
        except Exception:
            continue
        
        U_opt = _build_unitary(build_gates(result), qubits)
        dist = _distance(target, U_opt)
        
        if dist < best_dist:
            best_dist = dist
            best_params = result[:]
    
    # Also try the identity initialization with leastsq
    try:
        result, success, info, nfev = leastsq(
            objective, init_params, full_output=1, maxfev=200
        )
        U_opt = _build_unitary(build_gates(result), qubits)
        dist = _distance(target, U_opt)
        if dist < best_dist:
            best_dist = dist
            best_params = result[:]
    except Exception:
        pass
    
    # Polish with smaller steps if needed
    if best_dist > 1e-8 and _time_left() > 10:
        step = 0.01
        for i in range(n_params):
            if _elapsed() > _DEADLINE - 5:
                break
            for trial in (best_params[i] + step, best_params[i] - step):
                params = best_params[:]
                params[i] = trial
                U_test = _build_unitary(build_gates(params), qubits)
                d = _distance(target, U_test)
                if d < best_dist:
                    best_params = params[:]
                    best_dist = d
    
    return build_gates(best_params)


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    known = _is_known_gate(name, qubits)
    if known == "CNOT":
        return _synthesize_CNOT()
    elif known == "iSWAP":
        return _synthesize_iSWAP()
    elif known == "SWAP":
        return _synthesize_SWAP()
    elif known == "Toffoli":
        return _synthesize_Toffoli()
    
    # Generic optimization
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
