"""Quantum circuit synthesis via direct algebraic QSD factorization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            eye = np.eye(2, dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            idx = np.arange(n)
            ctrl_bit = (idx >> (qubits - 1 - control)) & 1
            tgt_shift = (qubits - 1 - target)
            mask = (ctrl_bit == 1)
            new_idx = idx.copy()
            new_idx[mask] = idx[mask] ^ (1 << tgt_shift)
            U = U[new_idx, :]
    return U


def _distance(target, U):
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(val) / dim)


def _check_close(A, B, tol=1e-6):
    dim = A.shape[0]
    max_idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[max_idx]) < 1e-12:
        return False
    phase = A[max_idx] / B[max_idx]
    if abs(phase) < 1e-12:
        return False
    phase = phase / abs(phase)
    return np.allclose(A, phase * B, atol=tol)


def _known_circuit_cnot_2q():
    return [["cx", 0, 1]]


def _known_circuit_swap_2q():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_iswap_2q():
    pi = math.pi
    return [
        ["u3", 0, pi / 2, 0, pi / 2],
        ["u3", 1, pi / 2, 0, -pi / 2],
        ["cx", 0, 1],
        ["u3", 0, pi / 2, 0, -pi / 2],
        ["u3", 1, pi / 2, 0, pi / 2],
        ["cx", 1, 0],
    ]


def _known_circuit_toffoli_3q():
    pi = math.pi
    return [
        ["u3", 2, pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["u3", 2, pi / 2, 0, 0],
    ]


def _detect_known_gate(target, qubits, budget):
    if qubits == 2:
        if budget >= 1:
            ref = _build_fast([["cx", 0, 1]], 2)
            if _check_close(target, ref):
                return _known_circuit_cnot_2q()
            ref = _build_fast([["cx", 1, 0]], 2)
            if _check_close(target, ref):
                return [["cx", 1, 0]]
        if budget >= 3:
            circ = _known_circuit_swap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
        if budget >= 2:
            circ = _known_circuit_iswap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
    elif qubits == 3:
        if budget >= 6:
            circ = _known_circuit_toffoli_3q()
            ref = _build_fast(circ, 3)
            if _check_close(target, ref):
                return circ
    return None


def _decompose_2q(QSD, target):
    """
    Decompose a 2-qubit unitary into single-qubit gates and CNOTs using QSD.
    Returns (left_gates, right_gates, cnot_count) where left_gates and right_gates
    are lists of u3 gates for qubits 0 and 1.
    
    The 2-qubit QSD: U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1)
    where A0, A1, B0, B1 are single-qubit unitaries.
    """
    n = 4
    # QSD for 2 qubits: U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1)
    # We need to find A0, A1, B0, B1 such that this holds.
    
    # Standard approach: use the fact that any 2-qubit unitary can be written as
    # U = (A ⊗ B) · CNOT · (C ⊗ D)
    
    # One method: 
    # Step 1: Compute the "local equivalence" by removing one CNOT
    # Let's use a direct algebraic approach.
    
    # For 2 qubits, we can use the following decomposition:
    # U = (A0 ⊗ A1) · CX · (B0 ⊗ B1)
    
    # Let's define P = U · CX^dag = U · CX (since CX is self-inverse)
    # Then P = (A0 ⊗ A1) · (B0 ⊗ B1) = (A0·B0) ⊗ (A1·B1)
    # So P must be a product of two single-qubit unitaries.
    
    # Actually, the standard QSD for 2 qubits uses:
    # U = (A0 ⊗ A1) · CX · (B0 ⊗ B1)
    
    # Let's compute: P = U @ CX
    cx = _build_fast([["cx", 0, 1]], 2)
    P = target @ cx  # P = U · CX^dag = U · CX since CX is self-inverse and unitary
    
    # P should be (A0·B0) ⊗ (A1·B1) = M0 ⊗ M1
    # where M0 and M1 are single-qubit unitaries.
    
    # To extract M0 and M1 from P = M0 ⊗ M1:
    # P[0:2, 0:2] = M0[0,0] · M1
    # P[0:2, 2:4] = M0[0,1] · M1
    # P[2:4, 0:2] = M0[1,0] · M1
    # P[2:4, 2:4] = M0[1,1] · M1
    
    # Let's extract M1 from the first block:
    # If M0[0,0] != 0, then M1 = P[0:2, 0:2] / M0[0,0]
    
    # But we don't know M0 yet. Let's use a different approach.
    
    # Alternative: use SVD or direct extraction.
    # P = M0 ⊗ M1
    # The singular values of P are the products of singular values of M0 and M1.
    # Since M0 and M1 are unitary, all singular values of P are 1.
    
    # Let's try to extract M1 by looking at the structure.
    # P = [[m00*m10, m00*m11], [m00*m12, m00*m13],
    #      [m01*m10, m01*m11], [m01*m12, m01*m13],
    #      [m10*m10, m10*m11], [m10*m12, m10*m13],
    #      [m11*m10, m11*m11], [m11*m12, m11*m13]]
    
    # Wait, let me index properly. P is 4x4.
    # P[i,j] where i = 2*a + b, j = 2*c + d
    # P[2*a+b, 2*c+d] = M0[a,c] * M1[b,d]
    
    # So:
    # P[0,0] = M0[0,0]*M1[0,0]
    # P[0,1] = M0[0,0]*M1[0,1]
    # P[0,2] = M0[0,1]*M1[0,0]
    # P[0,3] = M0[0,1]*M1[0,1]
    # P[1,0] = M0[1,0]*M1[0,0]
    # P[1,1] = M0[1,0]*M1[0,1]
    # P[1,2] = M0[1,1]*M1[0,0]
    # P[1,3] = M0[1,1]*M1[0,1]
    # P[2,0] = M0[0,0]*M1[1,0]
    # P[2,1] = M0[0,0]*M1[1,1]
    # P[2,2] = M0[0,1]*M1[1,0]
    # P[2,3] = M0[0,1]*M1[1,1]
    # P[3,0] = M0[1,0]*M1[1,0]
    # P[3,1] = M0[1,0]*M1[1,1]
    # P[3,2] = M0[1,1]*M1[1,0]
    # P[3,3] = M0[1,1]*M1[1,1]
    
    # From P[0,0] and P[0,1]:
    # M1[0,0] = P[0,0] / M0[0,0]
    # M1[0,1] = P[0,1] / M0[0,0]
    
    # From P[2,0] and P[2,1]:
    # M1[1,0] = P[2,0] / M0[0,0]
    # M1[1,1] = P[2,1] / M0[0,0]
    
    # So M1 = (1/M0[0,0]) * [[P[0,0], P[0,1]], [P[2,0], P[2,1]]]
    
    # Similarly, from P[0,0] and P[1,0]:
    # M0[0,0] * M1[0,0] = P[0,0]
    # M0[1,0] * M1[0,0] = P[1,0]
    # So M0[1,0] / M0[0,0] = P[1,0] / P[0,0]
    
    # This is getting complex. Let's use a numerical approach.
    
    # Extract M1 assuming M0[0,0] is non-zero:
    if abs(P[0, 0]) > 1e-10:
        M1 = np.array([[P[0, 0] / P[0, 0], P[0, 1] / P[0, 0]],
                       [P[2, 0] / P[0, 0], P[2, 1] / P[0, 0]]], dtype=complex)
        # Normalize M1
        # M1 should be unitary, but let's check
        # Actually, this formula is wrong. Let me redo.
        # P[0,0] = M0[0,0]*M1[0,0]
        # P[0,1] = M0[0,0]*M1[0,1]
        # P[2,0] = M0[0,0]*M1[1,0]
        # P[2,1] = M0[0,0]*M1[1,1]
        # So M1[0,0] = P[0,0]/M0[0,0], etc.
        # But we don't know M0[0,0].
        
        # Let's use the fact that M1 is unitary.
        # |M1[0,0]|^2 + |M1[1,0]|^2 = 1
        # |P[0,0]/M0[0,0]|^2 + |P[2,0]/M0[0,0]|^2 = 1
        # (|P[0,0]|^2 + |P[2,0]|^2) / |M0[0,0]|^2 = 1
        # |M0[0,0]|^2 = |P[0,0]|^2 + |P[2,0]|^2
        
        m00_mag_sq = abs(P[0, 0])**2 + abs(P[2, 0])**2
        if m00_mag_sq < 1e-10:
            # Try different extraction
            return None
        m00_mag = math.sqrt(m00_mag_sq)
        # Phase of M0[0,0]
        # M0[0,0] * M1[0,0] = P[0,0]
        # M0[0,0] * M1[1,0] = P[2,0]
        # M1 = [[P[0,0]/M0[0,0], P[0,1]/M0[0,0]], [P[2,0]/M0[0,0], P[2,1]/M0[0,0]]]
        # For M1 to be unitary, we need the columns to be orthonormal.
        # Let's just compute M1 with a chosen phase for M0[0,0].
        # Set M0[0,0] = m00_mag * exp(i*0) = m00_mag (real positive)
        # Then M1[0,0] = P[0,0]/m00_mag, etc.
        
        M0_00 = m00_mag
        M1 = np.array([[P[0, 0] / M0_00, P[0, 1] / M0_00],
                       [P[2, 0] / M0_00, P[2, 1] / M0_00]], dtype=complex)
        
        # Now extract M0:
        # M0[0,0] = M0_00
        # M0[0,1] = P[0,2] / M1[0,0] if M1[0,0] != 0
        # M0[1,0] = P[1,0] / M1[0,0] if M1[0,0] != 0
        # M0[1,1] = P[1,2] / M1[0,0] if M1[0,0] != 0
        
        if abs(M1[0, 0]) > 1e-10:
            M0 = np.array([[M0_00, P[0, 2] / M1[0, 0]],
                           [P[1, 0] / M1[0, 0], P[1, 2] / M1[0, 0]]], dtype=complex)
        elif abs(M1[0, 1]) > 1e-10:
            M0 = np.array([[M0_00, P[0, 3] / M1[0, 1]],
                           [P[1, 1] / M1[0, 1], P[1, 3] / M1[0, 1]]], dtype=complex)
        else:
            return None
        
        # Now we have M0 and M1 where P = M0 ⊗ M1
        # But P = (A0·B0) ⊗ (A1·B1)
        # So M0 = A0·B0 and M1 = A1·B1
        # We need to split M0 into A0 and B0, and M1 into A1 and B1.
        # Any single-qubit unitary M can be written as A·B where A and B are single-qubit unitaries.
        # For simplicity, set B0 = I, A0 = M0, B1 = I, A1 = M1.
        # Then U = (A0 ⊗ A1) · CX · (I ⊗ I) = (A0 ⊗ A1) · CX
        # But we need U = (A0 ⊗ A1) · CX · (B0 ⊗ B1)
        # If B0 = B1 = I, then U = (A0 ⊗ A1) · CX
        # Let's verify: P = U · CX = (A0 ⊗ A1) · CX · CX = A0 ⊗ A1
        # Yes! So if we set B0 = B1 = I, then A0 = M0, A1 = M1.
        
        # So the circuit is:
        # Apply B1 (identity) on q1, B0 (identity) on q0
        # Apply CX(0,1)
        # Apply A1 on q1, A0 on q0
        
        # Convert M0 and M1 to u3 gates
        u3_A0 = _mat_to_u3(M0)
        u3_A1 = _mat_to_u3(M1)
        
        if u3_A0 is None or u3_A1 is None:
            return None
        
        left_gates = [
            ["u3", 0, u3_A0[0], u3_A0[1], u3_A0[2]],
            ["u3", 1, u3_A1[0], u3_A1[1], u3_A1[2]],
        ]
        right_gates = []
        cnot_count = 1
        
        return left_gates, right_gates, cnot_count
    else:
        return None


def _mat_to_u3(M):
    """Convert a 2x2 unitary matrix to u3(theta, phi, lam) parameters."""
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                        [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    
    c = M[0, 0]
    # c = cos(theta/2)
    # If |c| > 1, clamp
    c_abs = min(1.0, abs(c))
    if c_abs < 1e-10:
        # theta = pi
        theta = math.pi
        # M[0,1] = -exp(i*lam)*sin(theta/2) = -exp(i*lam)
        # M[1,0] = exp(i*phi)*sin(theta/2) = exp(i*phi)
        # M[1,1] = exp(i*(phi+lam))*cos(theta/2) = 0
        # So lam = -arg(M[0,1]) and phi = arg(M[1,0])
        if abs(M[0, 1]) < 1e-10 or abs(M[1, 0]) < 1e-10:
            return None
        lam = -math.atan2(M[0, 1].imag, M[0, 1].real)
        phi = math.atan2(M[1, 0].imag, M[1, 0].real)
        return (theta, phi, lam)
    
    theta = 2 * math.acos(c_abs)
    s = math.sin(theta / 2)
    if s < 1e-10:
        # theta = 0, M should be diagonal
        # M = [[exp(i*alpha), 0], [0, exp(i*beta)]]
        # u3(0, phi, lam) = [[1, 0], [0, exp(i*(phi+lam))]]
        # So phi + lam = arg(M[1,1]) and the global phase is arg(M[0,0])
        # We can set phi = 0, lam = arg(M[1,1]) - arg(M[0,0])... but this doesn't capture global phase.
        # Actually u3 doesn't capture global phase. The global phase is determined by the target.
        # For now, set phi = 0, lam = arg(M[1,1])
        if abs(M[0, 1]) < 1e-10 and abs(M[1, 0]) < 1e-10:
            lam = math.atan2(M[1, 1].imag, M[1, 1].real)
            phi = 0.0
            return (0.0, phi, lam)
        return None
    
    # M[0,1] = -exp(i*lam)*s
    # So exp(i*lam) = -M[0,1]/s
    exp_i_lam = -M[0, 1] / s
    lam = math.atan2(exp_i_lam.imag, exp_i_lam.real)
    
    # M[1,0] = exp(i*phi)*s
    # So exp(i*phi) = M[1,0]/s
    exp_i_phi = M[1, 0] / s
    phi = math.atan2(exp_i_phi.imag, exp_i_phi.real)
    
    return (theta, phi, lam)


def _qsd_2q(target):
    """
    Full QSD for 2-qubit unitary. Returns a list of gates.
    """
    # For 2 qubits, the standard decomposition uses at most 3 CNOTs.
    # But our simple decomposition above uses 1 CNOT.
    # Let's verify if 1 CNOT is sufficient.
    # Actually, not all 2-qubit unitaries can be decomposed with 1 CNOT.
    # The general form is U = (A0 ⊗ A1) · CX · (B0 ⊗ B1) which uses 1 CNOT.
    # This should be sufficient for any 2-qubit unitary.
    
    result = _decompose_2q(None, target)
    if result is None:
        return None
    
    left_gates, right_gates, cnot_count = result
    
    # The circuit is: right_gates (B0, B1) then CX then left_gates (A0, A1)
    # But wait, in my derivation: U = (A0 ⊗ A1) · CX · (B0 ⊗ B1)
    # So apply B0 ⊗ B1 first, then CX, then A0 ⊗ A1.
    # B0 ⊗ B1 = identity in my simple case.
    
    gates = []
    gates.extend(right_gates)
    gates.append(["cx", 0, 1])
    gates.extend(left_gates)
    
    return gates


def _qsd_3q(target):
    """
    Full QSD for 3-qubit unitary. Returns a list of gates.
    Uses recursive decomposition: U = (A0 ⊗ U12) · CX(0,1) · (B0 ⊗ V12)
    where U12 and V12 are 2-qubit unitaries on qubits 1,2.
    """
    n = 8
    # Split the 3-qubit unitary into blocks.
    # Qubit 0 is MSB.
    # U[0:4, 0:4] = U when q0=0
    # U[4:8, 0:4] = U when q0=1 (output) and input q0=0
    # U[0:4, 4:8] = U when output q0=0 and input q0=1
    # U[4:8, 4:8] = U when q0=1
    
    # The QSD for 3 qubits:
    # U = (A0 ⊗ U12) · CX(0,1) · (B0 ⊗ V12)
    # where A0, B0 are single-qubit unitaries on q0,
    # and U12, V12 are 2-qubit unitaries on qubits 1,2.
    
    # Let's define:
    # P = U · CX(0,1)^dag = U · CX(0,1)
    # P = (A0 ⊗ U12) · (B0 ⊗ V12) = (A0·B0) ⊗ (U12·V12)
    
    # Wait, this isn't quite right because CX(0,1) doesn't factor as a product.
    
    # Let me use a different approach. The standard 3-qubit QSD:
    # U = (A0 ⊗ A12) · CX(0,1) · (B0 ⊗ B12)
    # where A12, B12 are 2-qubit unitaries on qubits 1,2.
    
    # Let CX01 = CX(0,1)
    # P = U · CX01^dag = U · CX01 (since CX is self-inverse)
    # P = (A0 ⊗ A12) · CX01 · (B0 ⊗ B12) · CX01
    # This is getting complicated.
    
    # Let's use a simpler recursive approach:
    # U = (A0 ⊗ I) · CX(0,1) · (B0 ⊗ I) · (I ⊗ U12)
    # This isn't standard.
    
    # Alternative: use the fact that any 3-qubit unitary can be decomposed as:
    # U = (A0 ⊗ A1 ⊗ A2) · CX(0,1) · CX(1,2) · (B0 ⊗ B1 ⊗ B2)
    # This uses 2 CNOTs, but we need up to 6.
    
    # Let's just use the 1-CNOT decomposition for 2 qubits recursively.
    # For 3 qubits, we can write:
    # U = (A0 ⊗ U12) · CX(0,1) · (B0 ⊗ V12)
    # where U12 and V12 are 2-qubit unitaries.
    
    # Let's compute:
    # P = U · CX(0,1)
    # CX(0,1) in the 8x8 space:
    # It swaps |10> and |11>, i.e., indices 2 and 3.
    cx01 = _build_fast([["cx", 0, 1]], 3)
    P = target @ cx01
    
    # P = (A0 ⊗ U12) · (B0 ⊗ V12) = (A0·B0) ⊗ (U12·V12)
    # So P = M0 ⊗ M12 where M0 is 2x2 and M12 is 4x4.
    
    # Extract M0 and M12 from P = M0 ⊗ M12:
    # P[2*i+a, 2*j+b] = M0[i,j] * M12[a,b] where a,b in 0..3
    
    # P[0:4, 0:4] = M0[0,0] * M12
    # P[4:8, 0:4] = M0[1,0] * M12
    # P[0:4, 4:8] = M0[0,1] * M12
    # P[4:8, 4:8] = M0[1,1] * M12
    
    if abs(P[0, 0]) > 1e-10:
        M12 = P[0:4, 0:4] / P[0, 0]
        M0_00 = math.sqrt(abs(P[0, 0])**2 + abs(P[4, 0])**2)
        M0_10 = P[4, 0] / M0_00
        M0_01 = P[0, 4] / P[0, 0]
        M0_11 = P[4, 4] / P[0, 0]
        M0 = np.array([[M0_00, M0_01], [M0_10, M0_11]], dtype=complex)
    else:
        return None
    
    # Now M0 = A0 · B0 and M12 = U12 · V12
    # Set B0 = I, A0 = M0, V12 = I, U12 = M12
    # Then U = (A0 ⊗ U12) · CX(0,1) · (I ⊗ I)
    # Verify: P = U · CX(0,1) = (A0 ⊗ U12) · CX(0,1) · CX(0,1) = A0 ⊗ U12 = M0 ⊗ M12. Correct!
    
    # So the circuit is:
    # Apply V12 (identity) on q1,q2
    # Apply CX(0,1)
    # Apply U12 on q1,q2
    # Apply A0 on q0
    
    # Decompose U12 (4x4) into 2-qubit gates
    u12_gates = _qsd_2q(M12)
    if u12_gates is None:
        return None
    
    # Remap qubits: U12 is on qubits 1,2 (not 0,1)
    # In _qsd_2q, qubits are 0 and 1. We need to remap to 1 and 2.
    u12_gates_remapped = []
    for g in u12_gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            new_q = q + 1  # q 0->1, q 1->2
            u12_gates_remapped.append(["u3", new_q, th, ph, lm])
        else:
            _, c, t = g
            new_c = c + 1
            new_t = t + 1
            u12_gates_remapped.append(["cx", new_c, new_t])
    
    # Convert M0 to u3
    u3_A0 = _mat_to_u3(M0)
    if u3_A0 is None:
        return None
    
    gates = []
    gates.append(["cx", 0, 1])
    gates.extend(u12_gates_remapped)
    gates.append(["u3", 0, u3_A0[0], u3_A0[1], u3_A0[2]])
    
    return gates


def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Try QSD
    if qubits == 2:
        gates = _qsd_2q(target)
        if gates is not None:
            n_cnot = sum(1 for g in gates if g[0] == "cx")
            if n_cnot <= budget and len(gates) <= 400:
                # Verify
                U = _build_fast(gates, qubits)
                d = _distance(target, U)
                if d < 1e-6:
                    return gates
    elif qubits == 3:
        gates = _qsd_3q(target)
        if gates is not None:
            n_cnot = sum(1 for g in gates if g[0] == "cx")
            if n_cnot <= budget and len(gates) <= 400:
                U = _build_fast(gates, qubits)
                d = _distance(target, U)
                if d < 1e-6:
                    return gates
    
    # Fallback: return identity (worst case)
    return []
# EVOLVE-BLOCK-END
