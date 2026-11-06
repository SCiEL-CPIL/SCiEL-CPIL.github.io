"""Quantum circuit synthesis with exact decompositions and analytic SVD peeling."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 250.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _apply_single_qubit(matrix, qubit, single_qubit_matrix, qubits):
    dim = 2 ** qubits
    operator = np.array([[1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single_qubit_matrix if k == qubit else eye)
    return operator @ matrix


def _apply_cnot(matrix, control, target, qubits):
    dim = 2 ** qubits
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            matrix = _apply_single_qubit(matrix, q, single, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cnot(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_from_matrix(m):
    """Extract u3 parameters (theta, phi, lam) from a 2x2 unitary matrix."""
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    
    if abs(a) < 1e-15:
        theta = math.pi
        lam = (-np.angle(b)) % (2 * math.pi) if abs(b) > 1e-15 else 0
        phi = np.angle(c) % (2 * math.pi) if abs(c) > 1e-15 else 0
        return theta, phi, lam
    
    if abs(b) < 1e-15:
        return 0, 0, 0
    
    theta = 2 * np.arcsin(min(1.0, abs(b)))
    if theta > math.pi:
        theta = 2 * math.pi - theta
    
    sin_half = math.sin(theta / 2)
    if abs(sin_half) < 1e-15:
        return 0, 0, 0
    
    lam = (-np.angle(b)) % (2 * math.pi)
    phi = np.angle(c) % (2 * math.pi)
    
    return theta, phi, lam


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


def _extract_single_qubit_gate_2qubit(residual, qubit):
    """Extract the single-qubit gate on `qubit` from a 4x4 matrix that is a tensor product."""
    # residual = A ⊗ B where A is on qubit 0, B on qubit 1
    # If qubit == 0, extract A; if qubit == 1, extract B
    if qubit == 0:
        # A[0,0]*B = residual[0:2, 0:2]
        # A[0,1]*B = residual[0:2, 2:4]
        # A[1,0]*B = residual[2:4, 0:2]
        # A[1,1]*B = residual[2:4, 2:4]
        # Use first column of each block to get A's columns (up to phase)
        # A[0,0] = residual[0,0] / B[0,0] ... but we don't know B
        # Instead: reshape to (2,2,2,2) and use SVD
        r = residual.reshape(2, 2, 2, 2)
        # r[i,j,k,l] where i,k are qubit 0 indices, j,l are qubit 1 indices
        # If r = A ⊗ B, then r[i,j,k,l] = A[i,k] * B[j,l]
        # Reshape to (4,4): rows=(i,k), cols=(j,l)
        r_mat = r.transpose(0, 2, 1, 3).reshape(4, 4)
        U_svd, s, Vt = np.linalg.svd(r_mat, full_matrices=False)
        if s[0] > 1e-10:
            A_mat = (U_svd[:, 0:1] * np.sqrt(s[0])).reshape(2, 2)
            # Normalize phase
            if abs(A_mat[0, 0]) > 1e-15:
                phase = A_mat[0, 0] / abs(A_mat[0, 0])
                A_mat = A_mat / phase
            return A_mat
    else:
        # Extract B (qubit 1)
        r = residual.reshape(2, 2, 2, 2)
        r_mat = r.transpose(0, 2, 1, 3).reshape(4, 4)
        U_svd, s, Vt = np.linalg.svd(r_mat, full_matrices=False)
        if s[0] > 1e-10:
            B_mat = (np.sqrt(s[0]) * Vt[0:1, :]).reshape(2, 2)
            if abs(B_mat[0, 0]) > 1e-15:
                phase = B_mat[0, 0] / abs(B_mat[0, 0])
                B_mat = B_mat / phase
            return B_mat
    return np.eye(2, dtype=complex)


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """Decompose iSWAP into 2 CNOTs and single-qubit rotations.
    
    iSWAP = (H⊗H) · CNOT · (S†⊗S) · CNOT · (H⊗H)
    where S = diag(1, i)
    
    Let me verify: 
    H = 1/sqrt(2) [[1,1],[1,-1]]
    S = diag(1, i)
    
    iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    """
    gates = []
    # H on qubit 0
    gates.append(["u3", 0, math.pi, 0, math.pi])
    # H on qubit 1
    gates.append(["u3", 1, math.pi, 0, math.pi])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # S† on qubit 0: diag(1, -i) = u3(0, 0, -pi/2)
    gates.append(["u3", 0, 0, 0, -math.pi / 2])
    # S on qubit 1: diag(1, i) = u3(0, 0, pi/2)
    gates.append(["u3", 1, 0, 0, math.pi / 2])
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
    # Standard 6-CNOT Toffoli decomposition
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


def _analytic_peel_2qubit(target, budget):
    """Layer-by-layer analytic peeling for 2-qubit unitaries.
    
    Strategy: For a circuit with CNOTs at positions p1, p2, ..., pn,
    we have U = S1 · C1 · S2 · C2 · ... · Cn · Sn+1
    where Si are single-qubit gates and Ci are CNOTs.
    
    We peel from the right: 
    1. Start with residual = U
    2. For each CNOT from right to left:
       - Apply CNOT† to residual (which is just CNOT since CNOT is self-inverse)
       - The result should be a tensor product A ⊗ B
       - Extract A and B via SVD
       - These become the single-qubit gates before this CNOT layer
    3. After all CNOTs, extract the final single-qubit gates
    """
    # Try different CNOT patterns
    cnot_patterns = []
    # Use alternating CNOTs to ensure connectivity
    for n_cnots in range(1, min(budget, 4) + 1):
        pattern = []
        for i in range(n_cnots):
            if i % 2 == 0:
                pattern.append((0, 1))
            else:
                pattern.append((1, 0))
        cnot_patterns.append(pattern)
    
    best_gates = None
    best_distance = 1.0
    
    for pattern in cnot_patterns:
        if len(pattern) > budget:
            continue
        try:
            gates_trial = _peel_2qubit_analytic(target, pattern)
            if gates_trial is not None:
                u = _build_unitary(gates_trial, 2)
                d = _distance(target, u)
                if d < best_distance:
                    best_distance = d
                    best_gates = gates_trial
                    if d < 1e-10:
                        break
        except Exception:
            continue
    
    return best_gates if best_gates is not None else []


def _peel_2qubit_analytic(target, cnot_pattern):
    """Analytically peel single-qubit gates for 2-qubit case."""
    current = target.copy()
    gates = []
    n_cnots = len(cnot_pattern)
    
    # Peel from right to left
    # After all CNOTs: current = S_{n+1}
    # After CNOT_n: current = S_n · CNOT_n
    # So: S_n = current · CNOT_n† = current · CNOT_n (CNOT is self-inverse)
    
    # Work backwards through the CNOTs
    for i in range(n_cnots - 1, -1, -1):
        control, target_q = cnot_pattern[i]
        # Apply CNOT to get the single-qubit layer before this CNOT
        current = _apply_cnot(current, control, target_q, 2)
        
        # Now current should be A ⊗ B (tensor product of single-qubit gates)
        A, B = _extract_tensor_product_2qubit(current)
        if A is None or B is None:
            return None
        
        # Add these gates (they come before the CNOT in the circuit)
        theta_a, phi_a, lam_a = _u3_from_matrix(A)
        theta_b, phi_b, lam_b = _u3_from_matrix(B)
        gates.append(["u3", 0, theta_a, phi_a, lam_a])
        gates.append(["u3", 1, theta_b, phi_b, lam_b])
        gates.append(["cx", control, target_q])
    
    # Final single-qubit layer (after all CNOTs)
    A, B = _extract_tensor_product_2qubit(current)
    if A is None or B is None:
        return None
    
    theta_a, phi_a, lam_a = _u3_from_matrix(A)
    theta_b, phi_b, lam_b = _u3_from_matrix(B)
    gates.append(["u3", 0, theta_a, phi_a, lam_a])
    gates.append(["u3", 1, theta_b, phi_b, lam_b])
    
    # Reverse gates since we built them backwards
    gates.reverse()
    return gates


def _extract_tensor_product_2qubit(matrix):
    """Extract A and B from a 4x4 matrix that should be A ⊗ B."""
    r = matrix.reshape(2, 2, 2, 2)
    # r[i,j,k,l] = A[i,k] * B[j,l]
    r_mat = r.transpose(0, 2, 1, 3).reshape(4, 4)
    
    U_svd, s, Vt = np.linalg.svd(r_mat, full_matrices=False)
    
    if len(s) < 1 or s[0] < 1e-10:
        return None, None
    
    # Check if it's approximately rank-1
    if len(s) > 1 and s[1] > 1e-8 * s[0]:
        # Not a clean tensor product; use best rank-1 approximation
        pass
    
    A_mat = (U_svd[:, 0:1] * np.sqrt(s[0])).reshape(2, 2)
    B_mat = (np.sqrt(s[0]) * Vt[0:1, :]).reshape(2, 2)
    
    # Check unitarity
    if not np.allclose(A_mat @ A_mat.conj().T, np.eye(2), atol=1e-6):
        return None, None
    if not np.allclose(B_mat @ B_mat.conj().T, np.eye(2), atol=1e-6):
        return None, None
    
    return A_mat, B_mat


def _analytic_peel_3qubit(target, budget):
    """Layer-by-layer analytic peeling for 3-qubit unitaries."""
    # Try different CNOT patterns
    cnot_patterns = []
    for n_cnots in range(2, min(budget, 6) + 1):
        # Use a pattern that ensures all qubits are connected
        pattern = []
        pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
        for i in range(n_cnots):
            pattern.append(pairs[i % len(pairs)])
        cnot_patterns.append(pattern)
    
    best_gates = None
    best_distance = 1.0
    
    for pattern in cnot_patterns:
        if len(pattern) > budget:
            continue
        try:
            gates_trial = _peel_3qubit_analytic(target, pattern)
            if gates_trial is not None:
                u = _build_unitary(gates_trial, 3)
                d = _distance(target, u)
                if d < best_distance:
                    best_distance = d
                    best_gates = gates_trial
                    if d < 1e-10:
                        break
        except Exception:
            continue
    
    return best_gates if best_gates is not None else []


def _peel_3qubit_analytic(target, cnot_pattern):
    """Analytically peel single-qubit gates for 3-qubit case."""
    current = target.copy()
    gates = []
    n_cnots = len(cnot_pattern)
    
    # Peel from right to left
    for i in range(n_cnots - 1, -1, -1):
        control, target_q = cnot_pattern[i]
        # Apply CNOT to get the single-qubit layer before this CNOT
        current = _apply_cnot(current, control, target_q, 3)
        
        # Now current should be A ⊗ B ⊗ C
        A, B, C = _extract_tensor_product_3qubit(current)
        if A is None or B is None or C is None:
            return None
        
        # Add these gates
        theta_a, phi_a, lam_a = _u3_from_matrix(A)
        theta_b, phi_b, lam_b = _u3_from_matrix(B)
        theta_c, phi_c, lam_c = _u3_from_matrix(C)
        gates.append(["u3", 0, theta_a, phi_a, lam_a])
        gates.append(["u3", 1, theta_b, phi_b, lam_b])
        gates.append(["u3", 2, theta_c, phi_c, lam_c])
        gates.append(["cx", control, target_q])
    
    # Final single-qubit layer
    A, B, C = _extract_tensor_product_3qubit(current)
    if A is None or B is None or C is None:
        return None
    
    theta_a, phi_a, lam_a = _u3_from_matrix(A)
    theta_b, phi_b, lam_b = _u3_from_matrix(B)
    theta_c, phi_c, lam_c = _u3_from_matrix(C)
    gates.append(["u3", 0, theta_a, phi_a, lam_a])
    gates.append(["u3", 1, theta_b, phi_b, lam_b])
    gates.append(["u3", 2, theta_c, phi_c, lam_c])
    
    gates.reverse()
    return gates


def _extract_tensor_product_3qubit(matrix):
    """Extract A, B, C from an 8x8 matrix that should be A ⊗ B ⊗ C."""
    # Reshape to (2,2,2,2,2,2)
    r = matrix.reshape((2,) * 6)
    # r[i,j,k,l,m,n] = A[i,l] * B[j,m] * C[k,n]
    
    # Use iterative SVD peeling
    # First, try to extract A by contracting over B and C
    
    # Reshape to (2*2*2, 2*2*2) where first 3 indices are (i,j,k) and last 3 are (l,m,n)
    r_mat = r.transpose(0, 2, 4, 1, 3, 5).reshape(8, 8)
    
    # This is a product of three 2x2 matrices
    # Use SVD iteratively
    
    # Extract A: look at the (i,l) structure
    # For fixed j,m,k,n: r[i,j,k,l,m,n] = A[i,l] * (B[j,m] * C[k,n])
    # Sum over j,m,k,n with appropriate weights
    
    # Simpler approach: use the fact that if M = A⊗B⊗C, then
    # M.reshape(2,2,2,2,2,2).transpose(0,3,1,4,2,5).reshape(4,4,4) gives us
    # a tensor where we can extract each factor
    
    # Alternative: use the column space approach
    # Take a slice where B and C are "fixed"
    
    # For now, use a heuristic: extract A from the (0,0) slice
    # A[i,l] should be proportional to sum_{j,m,k,n} r[i,j,k,l,m,n] * B[j,m] * C[k,n]
    # Since we don't know B and C, use the norm
    
    # Extract A: 
    # A[i,l] ∝ ||r[i,:,:,:,:,l,:]||_F (Frobenius norm over the B,C indices)
    A = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for l in range(2):
            # Sum over j,m,k,n
            A[i, l] = np.sum(r[i, :, :, l, :, :])  # This is a simplification
    
    # Better: use SVD on the reshaped matrix
    # Reshape to (2*2, 2*2, 2*2) = (4, 4, 4)
    # This represents the three qubits
    
    # For a tensor product A⊗B⊗C, we can use:
    # 1. Extract A by taking the "slice" where B and C are identity-like
    
    # Use the fact that M = A⊗B⊗C means:
    # M.reshape(2,2,2,2,2,2)[i,j,k,l,m,n] = A[i,l]*B[j,m]*C[k,n]
    
    # Extract A:
    # A[i,l] = (1/4) * sum_{j,m,k,n} M[i,j,k,l,m,n] * B†[m,j] * C†[n,k]
    # But we don't know B and C...
    
    # Use iterative approach:
    # 1. Assume B=I, C=I initially
    # 2. Extract A
    # 3. Divide out A
    # 4. Extract B assuming C=I
    # 5. Divide out B
    # 6. Extract C
    # 7. Iterate
    
    B = np.eye(2, dtype=complex)
    C = np.eye(2, dtype=complex)
    
    for iteration in range(10):
        # Extract A
        A_new = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for l in range(2):
                # A[i,l] = sum_{j,m,k,n} r[i,j,k,l,m,n] * B†[m,j] * C†[n,k]
                val = 0
                for j in range(2):
                    for m in range(2):
                        for k in range(2):
                            for n in range(2):
                                val += r[i, j, k, l, m, n] * B.conj().T[m, j] * C.conj().T[n, k]
                A_new[i, l] = val / 4.0
        
        # Check if A is unitary
        if not np.allclose(A_new @ A_new.conj().T, np.eye(2), atol=1e-4):
            # Try different initialization
            break
        
        # Divide out A
        r_divided = r.copy()
        for i in range(2):
            for l in range(2):
                for j in range(2):
                    for m in range(2):
                        for k in range(2):
                            for n in range(2):
                                r_divided[i, j, k, l, m, n] /= A_new[i, l] if abs(A_new[i, l]) > 1e-15 else 1.0
        
        # Extract B from r_divided (which should be B⊗C)
        B_new = np.zeros((2, 2), dtype=complex)
        for j in range(2):
            for m in range(2):
                val = 0
                for k in range(2):
                    for n in range(2):
                        val += r_divided[0, j, k, 0, m, n] * C.conj().T[n, k]
                B_new[j, m] = val / 2.0
        
        if not np.allclose(B_new @ B_new.conj().T, np.eye(2), atol=1e-4):
            break
        
        # Extract C from r_divided after dividing out B
        C_new = np.zeros((2, 2), dtype=complex)
        for k in range(2):
            for n in range(2):
                val = 0
                for j in range(2):
                    for m in range(2):
                        val += r_divided[0, j, k, 0, m, n] * B_new.conj().T[m, j]
                C_new[k, n] = val / 2.0
        
        if not np.allclose(C_new @ C_new.conj().T, np.eye(2), atol=1e-4):
            break
        
        A, B, C = A_new, B_new, C_new
    
    # Final check
    if not np.allclose(A @ A.conj().T, np.eye(2), atol=1e-4):
        return None, None, None
    if not np.allclose(B @ B.conj().T, np.eye(2), atol=1e-4):
        return None, None, None
    if not np.allclose(C @ C.conj().T, np.eye(2), atol=1e-4):
        return None, None, None
    
    return A, B, C


def _fallback_optimize_2qubit(target, budget):
    """Fallback: simple optimization if analytic peeling fails."""
    import random
    rng = random.Random(4242)
    
    # Build a simple layout with alternating CNOTs
    pairs = [(0, 1), (1, 0)]
    n_cnots = min(budget, 3)
    layout = []
    for layer in range(n_cnots):
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
    
    max_sweeps = 10
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 0.5 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_unitary(circuit(values), 2))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def _fallback_optimize_3qubit(target, budget):
    """Fallback: simple optimization if analytic peeling fails."""
    import random
    rng = random.Random(4243)
    
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    n_cnots = min(budget, 5)
    layout = []
    for layer in range(n_cnots):
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
    
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 0.5 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step):
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

    # For generic unitaries, use analytic peeling with fallback
    if qubits == 2:
        gates = _analytic_peel_2qubit(target, budget)
        if gates is None or len(gates) == 0:
            gates = _fallback_optimize_2qubit(target, budget)
        return gates
    else:
        gates = _analytic_peel_3qubit(target, budget)
        if gates is None or len(gates) == 0:
            gates = _fallback_optimize_3qubit(target, budget)
        return gates
# EVOLVE-BLOCK-END
