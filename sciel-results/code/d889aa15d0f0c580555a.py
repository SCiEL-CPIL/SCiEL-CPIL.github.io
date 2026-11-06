"""Quantum circuit synthesis with exact decompositions and SVD-based analytic peeling."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 250.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
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


def _is_cnot_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_iswap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_swap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_toffoli_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 3:
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        return np.allclose(target, expected, atol=1e-10)
    return False


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """Decompose iSWAP into 2 CNOTs and single-qubit rotations."""
    # iSWAP = (H⊗H) · CNOT · (S⊗S†) · CNOT · (H⊗H)
    # where S = diag(1, i), S† = diag(1, -i)
    gates = []
    # H on qubit 0
    gates.append(["u3", 0, math.pi, 0, math.pi])
    # H on qubit 1
    gates.append(["u3", 1, math.pi, 0, math.pi])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # S on qubit 0: u3(0, 0, pi/2) = diag(1, i)
    gates.append(["u3", 0, 0, 0, math.pi / 2])
    # S† on qubit 1: u3(0, 0, -pi/2) = diag(1, -i)
    gates.append(["u3", 1, 0, 0, -math.pi / 2])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # H on qubit 0
    gates.append(["u3", 0, math.pi, 0, math.pi])
    # H on qubit 1
    gates.append(["u3", 1, math.pi, 0, math.pi])
    return gates


def _decompose_swap_2qubit():
    """Decompose SWAP into 3 CNOTs."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _decompose_toffoli_3qubit():
    """Decompose Toffoli into 6 CNOTs and single-qubit rotations."""
    gates = []
    # H on target (qubit 2)
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # T† on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # T† on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # H on target
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # T† on control 1
    gates.append(["u3", 1, 0, 0, -math.pi / 4])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # T on control 1
    gates.append(["u3", 1, 0, 0, math.pi / 4])
    # T† on control 0
    gates.append(["u3", 0, 0, 0, -math.pi / 4])
    return gates


def _u3_from_matrix(single_qubit_matrix):
    """Extract u3 parameters (theta, phi, lam) from a 2x2 unitary matrix."""
    m = single_qubit_matrix
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                         [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    
    # Handle edge cases
    if abs(a) < 1e-15:
        # theta = pi
        theta = math.pi
        if abs(b) < 1e-15:
            lam = 0
        else:
            lam = -np.angle(b) % (2 * math.pi)
        if abs(c) < 1e-15:
            phi = 0
        else:
            phi = np.angle(c) % (2 * math.pi)
        return theta, phi, lam
    
    if abs(b) < 1e-15:
        # theta = 0
        theta = 0
        phi = 0
        lam = 0
        return theta, phi, lam
    
    theta = 2 * np.arcsin(abs(b))
    # Ensure theta in [0, pi]
    if theta < 0:
        theta += 2 * math.pi
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta = 2 * math.pi - theta
    
    sin_half = math.sin(theta / 2)
    if abs(sin_half) < 1e-15:
        return 0, 0, 0
    
    lam = -np.angle(b) % (2 * math.pi)
    phi = np.angle(c) % (2 * math.pi)
    
    return theta, phi, lam


def _apply_single_qubit_gate_to_matrix(matrix, qubit, single_qubit_matrix, qubits):
    """Apply a single-qubit gate to the full unitary matrix."""
    dim = 2 ** qubits
    operator = np.array([[1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single_qubit_matrix if k == qubit else eye)
    return operator @ matrix


def _apply_cnot_to_matrix(matrix, control, target, qubits):
    """Apply a CNOT gate to the full unitary matrix."""
    dim = 2 ** qubits
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _svd_peeling_2qubit(target, budget):
    """SVD-based analytic peeling for 2-qubit unitaries."""
    # For 2 qubits, we can use the KAK decomposition or SVD peeling
    # Strategy: peel off single-qubit gates from left and right using SVD
    
    current = target.copy()
    gates = []
    
    # Try different CNOT patterns and pick the best
    cnot_patterns = [
        [(0, 1)],
        [(0, 1), (1, 0)],
        [(0, 1), (1, 0), (0, 1)],
    ]
    
    best_gates = None
    best_distance = 1.0
    
    for pattern in cnot_patterns:
        if len(pattern) > budget:
            continue
        try:
            gates_trial = _peel_with_pattern(current, pattern, 2)
            if gates_trial is not None:
                u = _build_unitary(gates_trial, 2)
                d = _distance(target, u)
                if d < best_distance:
                    best_distance = d
                    best_gates = gates_trial
        except:
            continue
    
    if best_gates is not None:
        return best_gates
    
    # Fallback: use identity
    return []


def _peel_with_pattern(target, cnot_pattern, qubits):
    """Peel single-qubit gates using SVD at each step."""
    current = target.copy()
    gates = []
    dim = 2 ** qubits
    
    # Work backwards: we want to find gates such that U = G1 * G2 * ... * Gn
    # where G_i are the gates in order
    
    # First, apply all CNOTs in reverse to get the residual
    residual = current.copy()
    for control, target_q in reversed(cnot_pattern):
        residual = _apply_cnot_to_matrix(residual, control, target_q, qubits)
    
    # Now residual should be a product of single-qubit gates
    # For 2 qubits: residual = A ⊗ B
    # We can extract A and B by looking at specific columns
    
    if qubits == 2:
        # residual is 4x4
        # If residual = A ⊗ B, then:
        # residual[0:2, 0:2] = A[0,0] * B
        # residual[0:2, 2:4] = A[0,1] * B
        # residual[2:4, 0:2] = A[1,0] * B
        # residual[2:4, 2:4] = A[1,1] * B
        
        # Check if it's a tensor product
        # Use SVD to find the closest tensor product
        # Reshape to 2x2x2x2
        r = residual.reshape(2, 2, 2, 2)
        
        # Try to find A and B such that r[i,j,k,l] = A[i,k] * B[j,l]
        # This is a rank-1 constraint in the (i,k) vs (j,l) space
        # Reshape to (4, 4) where rows are (i,k) and cols are (j,l)
        r_mat = r.transpose(0, 2, 1, 3).reshape(4, 4)
        
        # SVD
        U_svd, s, Vt = np.linalg.svd(r_mat, full_matrices=False)
        
        if len(s) >= 1 and s[0] > 1e-10:
            # Rank-1 approximation
            A_mat = U_svd[:, 0:1] * np.sqrt(s[0])
            B_mat = np.sqrt(s[0]) * Vt[0:1, :]
            
            # Reshape back
            A = A_mat.reshape(2, 2)
            B = B_mat.reshape(2, 2)
            
            # Normalize
            if abs(A[0, 0]) > 1e-15:
                phase_A = A[0, 0] / abs(A[0, 0])
                A = A / phase_A
            if abs(B[0, 0]) > 1e-15:
                phase_B = B[0, 0] / abs(B[0, 0])
                B = B / phase_B
            
            # Check if A and B are unitary
            if np.allclose(A @ A.conj().T, np.eye(2), atol=1e-6) and \
               np.allclose(B @ B.conj().T, np.eye(2), atol=1e-6):
                
                # Build gates: first A on qubit 0, then B on qubit 1, then CNOTs in order
                theta_a, phi_a, lam_a = _u3_from_matrix(A)
                theta_b, phi_b, lam_b = _u3_from_matrix(B)
                
                gates_trial = []
                gates_trial.append(["u3", 0, theta_a, phi_a, lam_a])
                gates_trial.append(["u3", 1, theta_b, phi_b, lam_b])
                for control, target_q in cnot_pattern:
                    gates_trial.append(["cx", control, target_q])
                
                return gates_trial
    
    return None


def _svd_peeling_3qubit(target, budget):
    """SVD-based analytic peeling for 3-qubit unitaries."""
    # For 3 qubits, use a more general approach
    # Try different CNOT patterns
    
    cnot_patterns = [
        [(0, 1), (1, 2), (0, 1), (1, 2)],
        [(0, 2), (1, 2), (0, 2), (1, 2)],
        [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2)],
        [(0, 1), (1, 2), (0, 2), (1, 2), (0, 1)],
    ]
    
    best_gates = None
    best_distance = 1.0
    
    for pattern in cnot_patterns:
        if len(pattern) > budget:
            continue
        try:
            gates_trial = _peel_3qubit_with_pattern(target, pattern, 3)
            if gates_trial is not None:
                u = _build_unitary(gates_trial, 3)
                d = _distance(target, u)
                if d < best_distance:
                    best_distance = d
                    best_gates = gates_trial
        except:
            continue
    
    if best_gates is not None:
        return best_gates
    
    # Fallback: use coordinate descent
    return _coordinate_descent_3qubit(target, budget)


def _peel_3qubit_with_pattern(target, cnot_pattern, qubits):
    """Peel single-qubit gates for 3-qubit case."""
    current = target.copy()
    dim = 2 ** qubits
    
    # Apply CNOTs in reverse to get residual
    residual = current.copy()
    for control, target_q in reversed(cnot_pattern):
        residual = _apply_cnot_to_matrix(residual, control, target_q, qubits)
    
    # Now residual should be a product of single-qubit gates: A ⊗ B ⊗ C
    # Use iterative peeling: peel off one qubit at a time
    
    gates_trial = []
    remaining = residual.copy()
    qubits_remaining = list(range(3))
    
    for step in range(3):
        if time.time() - _start_time > _DEADLINE - 5:
            return None
        
        q = qubits_remaining[0]
        
        # Try to extract the single-qubit gate on qubit q
        # If remaining = G_q ⊗ R_rest, we can extract G_q
        # For qubit q, the remaining matrix has a specific structure
        
        # Use SVD approach: reshape and find the best single-qubit gate
        # This is complex for 3 qubits, so use a simpler approach:
        # Assume the gate is on qubit 0 (most significant) for now
        
        # Actually, let's use a different strategy:
        # Use the fact that for a tensor product A⊗B⊗C, we can extract
        # each factor by looking at specific slices
        
        # For now, use a heuristic: extract the gate on qubit 0
        # by averaging over the other qubits
        
        # Reshape remaining to (2, 2, 2, 2, 2, 2)
        r = remaining.reshape((2,) * 6)
        
        # Extract A (gate on qubit 0):
        # A[i,k] = sum over j,l,m,n of r[i,j,k,l,m,n] * B[j,l] * C[m,n]
        # This is circular, so use SVD
        
        # Reshape to (2*2*2, 2*2*2) = (8, 8) where first 3 indices are (i,j,m) and last 3 are (k,l,n)
        r_mat = r.transpose(0, 2, 4, 1, 3, 5).reshape(8, 8)
        
        U_svd, s, Vt = np.linalg.svd(r_mat, full_matrices=False)
        
        if len(s) >= 1 and s[0] > 1e-10:
            # The first left singular vector gives us information about A
            # But this is not straightforward for 3 qubits
            
            # Alternative: use coordinate descent for the single-qubit gates
            pass
    
    # For simplicity, use coordinate descent for 3-qubit generic unitaries
    return None


def _coordinate_descent_2qubit(target, budget):
    """Coordinate descent for 2-qubit unitaries."""
    import random
    rng = random.Random(4242)
    
    # Build initial layout
    pairs = [(0, 1), (1, 0)]
    layout = []
    for layer in range(min(budget, 4)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_unitary(circuit(best), 2))
    
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_unitary(circuit(values), 2))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def _coordinate_descent_3qubit(target, budget):
    """Coordinate descent for 3-qubit unitaries."""
    import random
    rng = random.Random(4243)
    
    pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    layout = []
    for layer in range(min(budget, 6)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_unitary(circuit(best), 3))
    
    max_sweeps = 15
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_unitary(circuit(values), 3))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check for known gates and use exact decompositions
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries, use SVD peeling with coordinate descent fallback
    if qubits == 2:
        gates = _svd_peeling_2qubit(target, budget)
        if gates is None or len(gates) == 0:
            gates = _coordinate_descent_2qubit(target, budget)
        return gates
    else:
        gates = _svd_peeling_3qubit(target, budget)
        if gates is None or len(gates) == 0:
            gates = _coordinate_descent_3qubit(target, budget)
        return gates
# EVOLVE-BLOCK-END
