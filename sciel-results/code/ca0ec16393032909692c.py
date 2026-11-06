"""Quantum circuit synthesis: exact decompositions + SVD-based peeling for generic unitaries."""
import math
import time
import numpy as np

_DEADLINE = 250.0
_T0 = time.monotonic()


def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _u3_matrix(q, theta, phi, lam, nq):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
    op = np.array([[1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for k in range(nq):
        op = np.kron(op, single if k == q else eye2)
    return op


def _cx_matrix(control, target, nq):
    n = 2 ** nq
    idx = np.arange(n)
    c_bit = (idx >> (nq - 1 - control)) & 1
    t_shift = (nq - 1 - target)
    new_idx = idx ^ (c_bit << t_shift)
    op = np.zeros((n, n), dtype=complex)
    op[new_idx, idx] = 1.0
    return op


def _build_fast(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            op = _u3_matrix(q, theta, phi, lam, qubits)
        else:
            _, c, t = gate
            op = _cx_matrix(c, t, qubits)
        matrix = op @ matrix
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3_from_matrix(mat2):
    """Extract (theta, phi, lam) from a 2x2 unitary, up to global phase."""
    det = mat2[0, 0] * mat2[1, 1] - mat2[0, 1] * mat2[1, 0]
    if abs(det) > 1e-15:
        phase = det / abs(det)
        mat2 = mat2 / phase

    a = mat2[0, 0]
    b = mat2[0, 1]
    c = mat2[1, 0]
    d = mat2[1, 1]

    theta = 2.0 * math.atan2(abs(b), abs(a))

    if abs(b) < 1e-12 and abs(c) < 1e-12:
        if abs(a) > 1e-12 and abs(d) > 1e-12:
            phi = 0.0
            lam = math.atan2(np.imag(d / a), np.real(d / a))
        else:
            phi = 0.0
            lam = 0.0
        return (theta, phi, lam)

    if abs(b) > 1e-12:
        lam = math.atan2(-np.imag(b), -np.real(b))
    else:
        lam = 0.0

    if abs(c) > 1e-12:
        phi = math.atan2(np.imag(c), np.real(c))
    else:
        phi = 0.0

    return (theta, phi, lam)


def _make_unitary_2x2(m2):
    U, _, Vh = np.linalg.svd(m2)
    return U @ Vh


def _make_unitary_4x4(m):
    U, _, Vh = np.linalg.svd(m)
    return U @ Vh


def _make_unitary_8x8(m):
    U, _, Vh = np.linalg.svd(m)
    return U @ Vh


def _extract_2qubit_product(m4):
    """
    Given a 4x4 unitary that should be of the form A ⊗ B,
    extract A (on qubit 0) and B (on qubit 1).
    """
    m = m4.reshape(2, 2, 2, 2)
    # m[i0, i1, j0, j1] where original index is (i0*2+i1, j0*2+j1)
    # We want: m4[(i0,i1),(j0,j1)] = A[i0,j0] * B[i1,j1]
    # So reshape to (i0, i1, j0, j1)
    m = m4.reshape(2, 2, 2, 2)
    
    # Get B: fix i0=0, j0=0 -> m[0, :, 0, :] = A[0,0] * B
    a00 = m4[0, 0]
    if abs(a00) > 1e-12:
        b = m[0, :, 0, :] / a00
    else:
        # Try a[1,1]
        a11 = m4[3, 3]
        if abs(a11) > 1e-12:
            b = m[1, :, 1, :] / a11
        else:
            b = np.eye(2, dtype=complex)
    
    b = _make_unitary_2x2(b)
    
    # Get A: fix i1=0, j1=0 -> m[:, 0, :, 0] = A * B[0,0]
    b00 = b[0, 0]
    if abs(b00) > 1e-12:
        a = m[:, 0, :, 0] / b00
    else:
        b11 = b[1, 1]
        if abs(b11) > 1e-12:
            a = m[:, 1, :, 1] / b11
        else:
            a = np.eye(2, dtype=complex)
    
    a = _make_unitary_2x2(a)
    
    return a, b


def _extract_3qubit_product(m8):
    """
    Given an 8x8 unitary that should be A ⊗ B ⊗ C,
    extract A (qubit 0), B (qubit 1), C (qubit 2).
    """
    # Reshape to (i0, i1, i2, j0, j1, j2)
    m = m8.reshape(2, 2, 2, 2, 2, 2)
    
    # Get C: fix i0=0, i1=0, j0=0, j1=0 -> m[0,0,:,0,0,:] = A[0,0]*B[0,0]*C
    a00 = m8[0, 0]
    b00 = m8[0, 1] / a00 if abs(a00) > 1e-12 else 1.0
    c00 = m8[0, 2] / (a00 * b00) if abs(a00 * b00) > 1e-12 else 1.0
    
    if abs(a00 * b00) > 1e-12:
        c = m[0, 0, :, 0, 0, :] / (a00 * b00)
    else:
        c = np.eye(2, dtype=complex)
    c = _make_unitary_2x2(c)
    
    # Get B: fix i0=0, i2=0, j0=0, j2=0 -> m[0,:,0,0,:,0] = A[0,0]*B*C[0,0]
    c00_new = c[0, 0]
    if abs(a00 * c00_new) > 1e-12:
        b = m[0, :, 0, 0, :, 0] / (a00 * c00_new)
    else:
        b = np.eye(2, dtype=complex)
    b = _make_unitary_2x2(b)
    
    # Get A: fix i1=0, i2=0, j1=0, j2=0 -> m[:,0,0,:,0,0] = A*B[0,0]*C[0,0]
    b00_new = b[0, 0]
    if abs(b00_new * c00_new) > 1e-12:
        a = m[:, 0, 0, :, 0, 0] / (b00_new * c00_new)
    else:
        a = np.eye(2, dtype=complex)
    a = _make_unitary_2x2(a)
    
    return a, b, c


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]
    
    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    
    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates
    
    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates
    
    return None


def _peel_2qubit(target, budget):
    """
    SVD layer peeling for 2-qubit unitaries.
    We want to express target as a product of single-qubit gates and CNOTs.
    Using the fact that any 2-qubit unitary can be written as:
    U = (A ⊗ B) CX (C ⊗ D) CX (E ⊗ F) CX (G ⊗ H) * global_phase
    """
    residual = target.copy()
    gates = []
    
    # CNOT pattern
    cnot_pattern = [(0, 1), (1, 0), (0, 1)]
    
    for layer in range(min(budget, 3)):
        if _time_left() < 5.0:
            break
        
        c, t = cnot_pattern[layer]
        cx_op = _cx_matrix(c, t, 2)
        
        # We want: residual = S_left @ CX @ S_right
        # where S_left = A ⊗ B and S_right = C ⊗ D
        # 
        # Iterative approach:
        # 1. Start with S_right = I
        # 2. Compute M = residual @ S_right^† @ CX^†
        #    This should equal S_left
        # 3. Extract A, B from S_left
        # 4. Update S_right using: S_right = CX^† @ S_left^† @ residual
        # 5. Iterate
        
        C = np.eye(2, dtype=complex)
        D = np.eye(2, dtype=complex)
        
        for _iter in range(50):
            if _time_left() < 3.0:
                break
            
            s_right = np.kron(C, D)
            # M = residual @ S_right^† @ CX^† should be S_left = A ⊗ B
            m = residual @ s_right.conj().T @ cx_op.conj().T
            
            # Extract A, B from m (should be product form)
            a, b = _extract_2qubit_product(m)
            
            # Update S_right: residual = (A⊗B) @ CX @ S_right
            # S_right = CX^† @ (A⊗B)^† @ residual
            s_right_new = cx_op.conj().T @ np.kron(a, b).conj().T @ residual
            s_right_new = _make_unitary_4x4(s_right_new)
            
            c_new, d_new = _extract_2qubit_product(s_right_new)
            
            # Check convergence
            if np.allclose(C, c_new, atol=1e-10) and np.allclose(D, d_new, atol=1e-10):
                C, D = c_new, d_new
                break
            
            C, D = c_new, d_new
        
        # Add gates: S_right first, then CX, then S_left
        theta_c, phi_c, lam_c = _u3_from_matrix(C)
        theta_d, phi_d, lam_d = _u3_from_matrix(D)
        gates.append(_u3(0, theta_c, phi_c, lam_c))
        gates.append(_u3(1, theta_d, phi_d, lam_d))
        gates.append(_cx(c, t))
        
        # Update residual: residual_new = residual @ S_right^† @ CX^†
        s_right = np.kron(C, D)
        residual = residual @ s_right.conj().T @ cx_op.conj().T
    
    # Final single-qubit layer
    if _time_left() > 1.0:
        a, b = _extract_2qubit_product(residual)
        theta_a, phi_a, lam_a = _u3_from_matrix(a)
        theta_b, phi_b, lam_b = _u3_from_matrix(b)
        gates.append(_u3(0, theta_a, phi_a, lam_a))
        gates.append(_u3(1, theta_b, phi_b, lam_b))
    
    return gates


def _peel_3qubit(target, budget):
    """
    SVD layer peeling for 3-qubit unitaries.
    Any 3-qubit unitary can be decomposed with at most 7 CNOTs (or fewer for special cases).
    We use a known decomposition pattern.
    """
    residual = target.copy()
    gates = []
    
    # Standard 3-qubit CNOT pattern (7 CNOTs)
    cnot_pattern = [
        (0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2), (0, 1)
    ]
    
    num_cnots = min(budget, 7)
    
    for layer in range(num_cnots):
        if _time_left() < 5.0:
            break
        
        c, t = cnot_pattern[layer]
        cx_op = _cx_matrix(c, t, 3)
        
        # Iterative extraction of single-qubit layers
        C0 = np.eye(2, dtype=complex)
        C1 = np.eye(2, dtype=complex)
        C2 = np.eye(2, dtype=complex)
        
        for _iter in range(30):
            if _time_left() < 3.0:
                break
            
            s_right = np.kron(np.kron(C0, C1), C2)
            m = residual @ s_right.conj().T @ cx_op.conj().T
            
            # Extract A, B, C from m (should be product form)
            a, b, cc = _extract_3qubit_product(m)
            
            # Update S_right
            s_right_new = cx_op.conj().T @ np.kron(np.kron(a, b), cc).conj().T @ residual
            s_right_new = _make_unitary_8x8(s_right_new)
            
            c0_new, c1_new, c2_new = _extract_3qubit_product(s_right_new)
            
            if (np.allclose(C0, c0_new, atol=1e-10) and 
                np.allclose(C1, c1_new, atol=1e-10) and 
                np.allclose(C2, c2_new, atol=1e-10)):
                C0, C1, C2 = c0_new, c1_new, c2_new
                break
            
            C0, C1, C2 = c0_new, c1_new, c2_new
        
        # Add gates
        theta0, phi0, lam0 = _u3_from_matrix(C0)
        theta1, phi1, lam1 = _u3_from_matrix(C1)
        theta2, phi2, lam2 = _u3_from_matrix(C2)
        gates.append(_u3(0, theta0, phi0, lam0))
        gates.append(_u3(1, theta1, phi1, lam1))
        gates.append(_u3(2, theta2, phi2, lam2))
        gates.append(_cx(c, t))
        
        # Update residual
        s_right = np.kron(np.kron(C0, C1), C2)
        residual = residual @ s_right.conj().T @ cx_op.conj().T
    
    # Final single-qubit layer
    if _time_left() > 1.0:
        a, b, c = _extract_3qubit_product(residual)
        theta_a, phi_a, lam_a = _u3_from_matrix(a)
        theta_b, phi_b, lam_b = _u3_from_matrix(b)
        theta_c, phi_c, lam_c = _u3_from_matrix(c)
        gates.append(_u3(0, theta_a, phi_a, lam_a))
        gates.append(_u3(1, theta_b, phi_b, lam_b))
        gates.append(_u3(2, theta_c, phi_c, lam_c))
    
    return gates


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build_fast(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # For generic unitaries, use SVD peeling
    if qubits == 2:
        result = _peel_2qubit(target, budget)
    else:
        result = _peel_3qubit(target, budget)
    
    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated
    
    if len(result) > 400:
        result = result[:400]
    
    return result
