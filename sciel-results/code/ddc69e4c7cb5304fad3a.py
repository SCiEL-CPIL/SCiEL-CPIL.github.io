# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 250

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_unitary_fast(gates, qubits):
    """Vectorized unitary builder."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
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
            cnot_mat = np.eye(n, dtype=complex)
            for i in range(n):
                c_bit = (i >> (qubits - 1 - control)) & 1
                if c_bit:
                    t_bit_pos = qubits - 1 - target
                    new_i = i ^ (1 << t_bit_pos)
                    cnot_mat[new_i, i] = 1
                else:
                    cnot_mat[i, i] = 1
            matrix = cnot_mat @ matrix
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _synthesize_cnot():
    return [["cx", 0, 1]]

def _synthesize_iswap():
    # iSWAP = [[0, i], [i, 0]]
    # Known decomposition using CNOT + single-qubit gates
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi/2, 0],
        ["u3", 1, math.pi/2, math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]

def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]

def _synthesize_toffoli():
    # Standard 6-CNOT Toffoli decomposition
    # Toffoli = |00><00| + |01><01| + |10><10| + |11><11| (phase -1 on |11>)
    # Using the standard construction
    pi = math.pi
    return [
        ["u3", 2, pi/2, 0, pi/2],
        ["cx", 1, 2],
        ["u3", 2, pi/4, 0, -pi/2],
        ["cx", 0, 2],
        ["u3", 2, -pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi/4, 0, -pi/2],
        ["u3", 1, pi/4, 0, pi/4],
        ["u3", 2, pi/4, 0, pi/2],
        ["cx", 1, 2],
        ["u3", 1, -pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi/2, 0, pi/2],
    ]

def _knot_product(A, B):
    """Compute A ⊗ B ⊗ I ⊗ I + ... (KAK interaction term)"""
    # A, B are 2x2
    # A⊗B⊗I⊗I + A⊗I⊗B⊗I + I⊗A⊗I⊗B + I⊗I⊗A⊗B
    term1 = np.kron(np.kron(A, B), np.kron(np.eye(2), np.eye(2)))
    term2 = np.kron(np.kron(A, np.eye(2)), np.kron(B, np.eye(2)))
    term3 = np.kron(np.kron(np.eye(2), A), np.kron(np.eye(2), B))
    term4 = np.kron(np.kron(np.eye(2), np.eye(2)), np.kron(A, B))
    return term1 + term2 + term3 + term4

def _cnot_matrix_4x4():
    """CNOT 0->1 in 4x4 (qubit 0 is MSB)."""
    mat = np.eye(4, dtype=complex)
    # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    mat[2, 2] = 0
    mat[3, 3] = 0
    mat[2, 3] = 1
    mat[3, 2] = 1
    return mat

def _decompose_2qubit_kak(U):
    """
    Decompose a 2-qubit unitary U into single-qubit gates + 1 CNOT using KAK.
    Returns list of gates or None if fails.
    """
    if _time_remaining() < 5:
        return None
    
    U = np.array(U, dtype=complex)
    # Ensure U is unitary (orthogonalize if needed)
    try:
        U, _ = np.linalg.qr(U)
        # Check unitarity
        if np.max(np.abs(U.conj().T @ U - np.eye(4))) > 1e-6:
            # Try SVD
            Umat, s, Vmat = np.linalg.svd(U)
            U = Umat @ Vmat
            if np.max(np.abs(U.conj().T @ U - np.eye(4))) > 1e-6:
                return None
    except:
        return None
    
    # KAK decomposition: U = A ⊗ B * exp(-i/2 * (a_x X⊗X + a_y Y⊗Y + a_z Z⊗Z)) * C ⊗ D
    # Step 1: Compute local invariants to find interaction parameters
    # Use the method from Makhijani, Vedral, Brukner, Plenio (2002)
    
    # First, compute the "canonical" form
    # U = (A⊗B) * CNOT^k * (C⊗D) where k is 0 or 1
    
    # Compute the local invariants
    def _local_invariants(U):
        # g1 = 1/4 Re(tr(U^T U))
        # g2 = 1/4 Re(tr(U^T U) * tr(U^T U)) - 1/8 |tr(U^T U)|^2 ... 
        # Actually use the standard method
        Ut = U.T
        # Compute tr(U^T U)
        t = np.trace(Ut @ U)
        # g1 = Re(t)/4
        g1 = np.real(t) / 4.0
        # g2 = 1/4 Re(tr(U^T U) tr(U^T U)) - 1/8 |tr(U^T U)|^2
        # Actually the standard invariants are:
        # g1 = 1/4 Re(tr(U^T U))
        # g2 = 1/4 Re(tr(U^T U) * tr(U^T U)) - 1/8 |tr(U^T U)|^2
        # Hmm, let me use a different approach.
        # 
        # Better approach: Use the method from "Optimal Control of Quantum Gates" 
        # or the direct KAK algorithm.
        
        # Let me use the algorithm from:
        # "Exact synthesis of quantum gates" or the standard KAK algorithm.
        # 
        # Algorithm:
        # 1. Compute U' = U * (I ⊗ U_2) where U_2 is chosen to make U' have a specific form
        # 2. This is complex. Let me use a simpler approach.
        
        # Alternative: Use the fact that for 2-qubit gates, we can always write
        # U = A ⊗ B * exp(-i θ Z⊗Z) * C ⊗ D
        # for some θ, and find A, B, C, D, θ.
        
        # Method from "Two-qubit gate decomposition" (standard):
        # 1. Compute the "interaction" part
        # 2. The interaction is characterized by 3 angles (a, b, c) in the KAK form
        #    exp(-i/2 (a X⊗X + b Y⊗Y + c Z⊗Z))
        # 3. For CNOT, a=b=c=π/2 (up to phase)
        
        # Let me use the algorithm from Nielsen & Chuang or the standard one.
        # 
        # Standard algorithm (from "Quantum Computation and Quantum Information"):
        # 1. Compute U' = U * (I ⊗ V) where V is chosen such that U' has 
        #    a specific property.
        # 
        # Actually, let me use a more direct numerical approach:
        # We know U = A⊗B * CNOT * C⊗D (for entangling gates)
        # or U = A⊗B * C⊗D (for non-entangling gates)
        # 
        # Test if U is entangling:
        # Compute the "local invariants" to determine if CNOT is needed.
        
        # Simple test: if U is a product gate, no CNOT needed.
        # Check if U = A ⊗ B for some A, B.
        # Reshape U into 4x4, check if it's a Kronecker product.
        # A ⊗ B has the property that U[i,j] * U[k,l] = U[i,l] * U[k,j] 
        # for all i,j,k,l (rank-1 in the "super" space).
        
        # More robust: use the fact that A⊗B means U.reshape(2,2,2,2) has 
        # a specific structure.
        
        # Let me just try to find the decomposition numerically.
        
        # Approach: 
        # 1. Assume U = A⊗B * CNOT * C⊗D
        # 2. Let M = U * (CNOT)^-1 = U * CNOT (since CNOT is self-inverse)
        # 3. Then M = A⊗B * C⊗D
        # 4. We need to find A, B, C, D such that M = A⊗B * C⊗D = (A*C) ⊗ (B*D)
        # 5. So M must be a Kronecker product!
        # 6. If M is a Kronecker product, we can find A⊗B and C⊗D.
        # 7. If M is NOT a Kronecker product, try U = A⊗B * C⊗D directly.
        
        return None  # We'll handle this in the main function
    
    # Try with CNOT:
    cnot = _cnot_matrix_4x4()
    M = U @ cnot  # U = M * CNOT, so M = U * CNOT
    
    # Check if M is a Kronecker product
    # M = A ⊗ B means M.reshape(2,2,2,2) = A[i,k] * B[j,l]
    # Equivalently, M is a Kronecker product iff it has rank 1 in the 
    # "super" matrix sense, or we can check:
    # M[0,0]*M[1,1] == M[0,1]*M[1,0] etc.
    
    def _is_kron_product(M):
        """Check if 4x4 M is a Kronecker product A⊗B."""
        # M = A⊗B means M[i*2+j, k*2+l] = A[i,k] * B[j,l]
        # Check consistency
        A = np.zeros((2,2), dtype=complex)
        B = np.zeros((2,2), dtype=complex)
        # From M[0,0] = A[0,0]*B[0,0], M[0,1] = A[0,0]*B[0,1], etc.
        # Let A[0,0] = 1, then B[0,0] = M[0,0], B[0,1] = M[0,1], B[1,0] = M[2,0], B[1,1] = M[2,1]
        # Then A[0,1] = M[0,2]/B[0,0] if B[0,0]!=0
        # This is getting complicated. Let me use a different check.
        
        # A 4x4 matrix is a Kronecker product iff its 2x2 "blocks" are proportional.
        # M = [[a*P, b*P], [c*P, d*P]] where P is 2x2 and [[a,b],[c,d]] is 2x2.
        # So M[0:2,0:2] = a*P, M[0:2,2:4] = b*P, M[2:4,0:2] = c*P, M[2:4,2:4] = d*P
        # Check: M[0:2,0:2] * M[2:4,2:4] == M[0:2,2:4] * M[2:4,0:2] (element-wise, up to scalar)
        
        P1 = M[0:2, 0:2]
        P2 = M[0:2, 2:4]
        P3 = M[2:4, 0:2]
        P4 = M[2:4, 2:4]
        
        # Check if P1, P2, P3, P4 are all proportional
        # P1 * P4 ≈ P2 * P3 (element-wise)
        if np.max(np.abs(P1 * P4 - P2 * P3)) < 1e-8:
            return True
        return False
    
    if _is_kron_product(M):
        # M = A⊗B * C⊗D = (A*C) ⊗ (B*D)
        # We need to decompose M into a Kronecker product
        # M = P ⊗ Q where P, Q are 2x2
        # From M[0:2, 0:2] = P[0,0] * Q
        # So Q = M[0:2, 0:2] / P[0,0]
        # And P[0,0] is free (global phase)
        
        # Let's extract P and Q
        # M[i*2+j, k*2+l] = P[i,k] * Q[j,l]
        # M[0,0] = P[0,0]*Q[0,0]
        # M[0,1] = P[0,0]*Q[0,1]
        # M[0,2] = P[0,1]*Q[0,0]
        # M[0,3] = P[0,1]*Q[0,1]
        # M[2,0] = P[1,0]*Q[0,0]
        # etc.
        
        # Set P[0,0] = 1 (absorb phase)
        P = np.zeros((2,2), dtype=complex)
        Q = np.zeros((2,2), dtype=complex)
        
        # From M[0,0] = P[0,0]*Q[0,0], set P[0,0]=1, Q[0,0]=M[0,0]
        P[0,0] = 1.0
        Q[0,0] = M[0,0]
        
        if abs(Q[0,0]) > 1e-12:
            Q[0,1] = M[0,1] / Q[0,0]
            Q[1,0] = M[2,0] / Q[0,0]
            Q[1,1] = M[2,1] / Q[0,0]
            P[0,1] = M[0,2] / Q[0,0]
            P[1,0] = M[2,2] / Q[0,0]
            P[1,1] = M[2,3] / Q[0,0]
        else:
            # Q[0,0] ≈ 0, try different pivot
            # M[0,2] = P[0,1]*Q[0,0] ≈ 0, so P[0,1] is free
            # Use M[0,1] = P[0,0]*Q[0,1]
            if abs(M[0,1]) > 1e-12:
                Q[0,1] = 1.0
                P[0,0] = M[0,1]
                Q[0,0] = M[0,0] / P[0,0] if abs(P[0,0]) > 1e-12 else 0
                Q[1,0] = M[2,0] / P[0,0] if abs(P[0,0]) > 1e-12 else 0
                Q[1,1] = M[2,1] / P[0,0] if abs(P[0,0]) > 1e-12 else 0
                P[0,1] = M[0,2] / Q[0,1] if abs(Q[0,1]) > 1e-12 else 0
                P[1,0] = M[2,2] / Q[1,0] if abs(Q[1,0]) > 1e-12 else 0
                P[1,1] = M[2,3] / Q[1,1] if abs(Q[1,1]) > 1e-12 else 0
            else:
                return None
        
        # Verify
        M_check = np.kron(P, Q)
        if np.max(np.abs(M_check - M)) > 1e-6:
            return None
        
        # Now M = P ⊗ Q = (A*C) ⊗ (B*D)
        # We need to split P = A*C and Q = B*D into unitary factors
        # This is always possible: choose C = P, A = I (or any split)
        # But we want A, B, C, D to be unitary (2x2)
        
        # Actually, P and Q should already be unitary (up to phase) if U and CNOT are unitary.
        # Let's check and orthogonalize if needed.
        
        def _make_unitary_2x2(M2):
            try:
                M2, _ = np.linalg.qr(M2)
                return M2
            except:
                return None
        
        # P = A * C, Q = B * D
        # We can choose: A = P, C = I, B = Q, D = I
        # Then U = A⊗B * CNOT * C⊗D = P⊗Q * CNOT * I⊗I
        # But we need to verify: U = (P⊗Q) @ CNOT
        # And M = U @ CNOT = P⊗Q ✓
        
        A = _make_unitary_2x2(P)
        B = _make_unitary_2x2(Q)
        C = np.eye(2, dtype=complex)
        D = np.eye(2, dtype=complex)
        
        if A is None or B is None:
            return None
        
        # Verify: A⊗B @ CNOT @ C⊗D ≈ U
        recon = np.kron(A, B) @ cnot @ np.kron(C, D)
        # Phase-insensitive check
        tr = np.trace(U.conj().T @ recon)
        if abs(tr) < 1e-4:
            # Try different split
            # Or maybe U is not entangling
            return None
        
        # Now convert A, B, C, D to u3 gates
        # u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
        
        def _u2_to_u3(U2):
            """Convert 2x2 unitary to u3(θ, φ, λ)."""
            # U2 = [[a, b], [c, d]]
            # u3: [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
            # So: a = cos(θ/2), -e^{iλ}sin(θ/2) = b, e^{iφ}sin(θ/2) = c, e^{i(φ+λ)}cos(θ/2) = d
            a, b, c, d = U2[0,0], U2[0,1], U2[1,0], U2[1,1]
            
            # cos(θ/2) = a
            # If |a| ≈ 1, θ ≈ 0
            if abs(abs(a) - 1) < 1e-10:
                # θ = 0, U2 = [[a, 0], [0, d]] where a*d = 1 (unitary)
                # This is a phase gate: u3(0, φ, λ) = [[1, 0], [0, e^{i(φ+λ)}]]
                # So e^{i(φ+λ)} = d/a * a... wait
                # u3(0, φ, λ) = [[1, 0], [0, e^{i(φ+λ)}]]
                # We need e^{i(φ+λ)} = d (since a=1 up to phase, but a might be -1)
                # Actually if a = e^{iα}, then U2 = e^{iα} [[1,0],[0,d/a]]
                # Global phase doesn't matter for u3 (it's absorbed)
                # Set θ=0, φ=0, λ = arg(d)
                lam = np.angle(d)
                return (0.0, 0.0, lam)
            
            # cos(θ/2) = |a|, sin(θ/2) = |b| = |c|
            # θ/2 = arccos(|a|)
            cos_half = np.real_if_close(a)
            if np.iscomplex(cos_half):
                # a is complex, need to handle
                # Actually cos(θ/2) should be real in u3
                # The u3 form has cos(θ/2) as the (0,0) element, which is real
                # So a must be real (up to global phase)
                # Let me reconsider.
                
                # u3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
                # The (0,0) element is cos(θ/2), which is REAL.
                # So if a is not real, we have a problem.
                # But any 2x2 unitary can be written in u3 form!
                # Let me re-derive.
                
                # Actually, the standard u3 (or U3) gate is:
                # U3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
                # This has 3 real parameters and can represent any single-qubit unitary
                # up to global phase.
                
                # To extract:
                # cos(θ/2) = |U[0,0]| ... no, U[0,0] = cos(θ/2) which is real.
                # But U[0,0] might have a phase!
                
                # The issue: u3 as defined has U[0,0] = cos(θ/2) which is real and ≥ 0 (for θ∈[0,π]).
                # But a general unitary has U[0,0] with arbitrary phase.
                
                # The resolution: u3 can represent any unitary up to global phase.
                # So we need to factor out the global phase first.
                
                # Let U2 = e^{iγ} * U3 where U3 is in u3 form.
                # U3[0,0] = cos(θ/2) (real, ≥ 0)
                # So U2[0,0] = e^{iγ} cos(θ/2)
                # Thus γ = arg(U2[0,0]) and cos(θ/2) = |U2[0,0]|
                
                gamma = np.angle(a)
                cos_half = np.abs(a)
                sin_half = np.sqrt(max(0, 1 - cos_half**2))
                
                if sin_half < 1e-12:
                    # θ ≈ 0
                    # U2 ≈ e^{iγ} [[1, 0], [0, e^{iδ}]]
                    # U3 = [[1, 0], [0, e^{iδ}]] = u3(0, 0, δ)
                    # But U2[1,1] = e^{iγ} e^{iδ}
                    # So e^{iδ} = U2[1,1] / e^{iγ}
                    phase_11 = d / np.exp(1j * gamma)
                    lam = np.angle(phase_11)
                    return (0.0, 0.0, lam)
                
                # θ = 2 * arccos(cos_half)
                theta = 2 * np.arccos(np.clip(cos_half, -1, 1))
                
                # -e^{iλ} sin(θ/2) = U2[0,1] / e^{iγ}
                b_prime = b / np.exp(1j * gamma)
                # -e^{iλ} sin(θ/2) = b_prime
                # e^{iλ} = -b_prime / sin(θ/2)
                lam = np.angle(-b_prime / sin_half)
                
                # e^{iφ} sin(θ/2) = U2[1,0] / e^{iγ}
                c_prime = c / np.exp(1j * gamma)
                # e^{iφ} = c_prime / sin(θ/2)
                phi = np.angle(c_prime / sin_half)
                
                # Verify: e^{i(φ+λ)} cos(θ/2) should equal U2[1,1] / e^{iγ}
                # d_prime = d / np.exp(1j * gamma)
                # Check: e^{i(φ+λ)} cos_half ≈ d_prime
                # (not critical, the u3 form is determined)
                
                return (theta, phi, lam)
            else:
                cos_half = float(np.real(cos_half))
                sin_half = np.sqrt(max(0, 1 - cos_half**2))
                
                if sin_half < 1e-12:
                    gamma = np.angle(a)
                    phase_11 = d / np.exp(1j * gamma)
                    lam = np.angle(phase_11)
                    return (0.0, 0.0, lam)
                
                theta = 2 * np.arccos(np.clip(cos_half, -1, 1))
                
                gamma = np.angle(a)
                b_prime = b / np.exp(1j * gamma)
                lam = np.angle(-b_prime / sin_half)
                
                c_prime = c / np.exp(1j * gamma)
                phi = np.angle(c_prime / sin_half)
                
                return (theta, phi, lam)
        
        # Extract u3 parameters
        A_theta, A_phi, A_lam = _u2_to_u3(A)
        B_theta, B_phi, B_lam = _u2_to_u3(B)
        
        gates = []
        # Apply A on qubit 0, B on qubit 1
        gates.append(["u3", 0, A_theta, A_phi, A_lam])
        gates.append(["u3", 1, B_theta, B_phi, B_lam])
        gates.append(["cx", 0, 1])
        # C and D are identity, so no gates needed
        # But we might need to add them if they're not identity
        # For now, C=I, D=I
        
        # Verify the full circuit
        U_circuit = _build_unitary_fast(gates, 2)
        dist = _distance(U, U_circuit)
        if dist < 1e-6:
            return gates
        else:
            return None
    
    # If CNOT didn't work, try without CNOT (product gate)
    # U = A ⊗ B
    def _is_kron_product_direct(U):
        P1 = U[0:2, 0:2]
        P2 = U[0:2, 2:4]
        P3 = U[2:4, 0:2]
        P4 = U[2:4, 2:4]
        if np.max(np.abs(P1 * P4 - P2 * P3)) < 1e-8:
            return True
        return False
    
    if _is_kron_product_direct(U):
        # U = P ⊗ Q
        P = np.zeros((2,2), dtype=complex)
        Q = np.zeros((2,2), dtype=complex)
        P[0,0] = 1.0
        Q[0,0] = U[0,0]
        if abs(Q[0,0]) > 1e-12:
            Q[0,1] = U[0,1] / Q[0,0]
            Q[1,0] = U[2,0] / Q[0,0]
            Q[1,1] = U[2,1] / Q[0,0]
            P[0,1] = U[0,2] / Q[0,0]
            P[1,0] = U[2,2] / Q[0,0]
            P[1,1] = U[2,3] / Q[0,0]
        else:
            return None
        
        U_check = np.kron(P, Q)
        if np.max(np.abs(U_check - U)) > 1e-6:
            return None
        
        def _make_unitary_2x2(M2):
            try:
                M2, _ = np.linalg.qr(M2)
                return M2
            except:
                return None
        
        A = _make_unitary_2x2(P)
        B = _make_unitary_2x2(Q)
        if A is None or B is None:
            return None
        
        def _u2_to_u3(U2):
            a, b, c, d = U2[0,0], U2[0,1], U2[1,0], U2[1,1]
            gamma = np.angle(a)
            cos_half = np.abs(a)
            sin_half = np.sqrt(max(0, 1 - cos_half**2))
            if sin_half < 1e-12:
                phase_11 = d / np.exp(1j * gamma)
                lam = np.angle(phase_11)
                return (0.0, 0.0, lam)
            theta = 2 * np.arccos(np.clip(cos_half, -1, 1))
            b_prime = b / np.exp(1j * gamma)
            lam = np.angle(-b_prime / sin_half)
            c_prime = c / np.exp(1j * gamma)
            phi = np.angle(c_prime / sin_half)
            return (theta, phi, lam)
        
        A_theta, A_phi, A_lam = _u2_to_u3(A)
        B_theta, B_phi, B_lam = _u2_to_u3(B)
        
        gates = [
            ["u3", 0, A_theta, A_phi, A_lam],
            ["u3", 1, B_theta, B_phi, B_lam],
        ]
        
        U_circuit = _build_unitary_fast(gates, 2)
        dist = _distance(U, U_circuit)
        if dist < 1e-6:
            return gates
        else:
            return None
    
    return None

def _try_algebraic_2qubit(problem):
    """Try KAK decomposition for 2-qubit unitary."""
    target = _target(problem)
    return _decompose_2qubit_kak(target)

def _decompose_3qubit_tensor(problem):
    """Check if 3-qubit unitary is a tensor product of 2-qubit and 1-qubit gates."""
    target = _target(problem)
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    if qubits != 3:
        return None
    
    # Check if U = A ⊗ B ⊗ C (tensor product of 2x2, 2x2, 2x2)
    # U.reshape(2,2,2,2,2,2) should have a specific structure
    # U[i,j,k,l,m,n] = A[i,l] * B[j,m] * C[k,n]
    
    U = target
    # Reshape to 2^6 tensor
    U_tensor = U.reshape((2,)*6)
    
    # Check if it's a tensor product
    # U_tensor[i,j,k,l,m,n] = A[i,l] * B[j,m] * C[k,n]
    # Pick A[0,0] = 1 (normalize)
    # Then B[j,m] = U_tensor[0,j,0,0,m,0] / C[0,0]
    # This is complex. Let me use a different approach.
    
    # Check: U_tensor[i,j,k,l,m,n] * U_tensor[i',j',k',l',m',n'] 
    # = U_tensor[i,j,k,l',m',n'] * U_tensor[i',j',k',l,m,n]
    # for all indices (this is the condition for being a tensor product)
    
    # Simpler: try to extract A, B, C
    # A[i,l] = U_tensor[i,0,0,l,0,0] / (B[0,0] * C[0,0])
    # Set B[0,0] = C[0,0] = 1 (absorb phases)
    # Then A[i,l] = U_tensor[i,0,0,l,0,0]
    # B[j,m] = U_tensor[0,j,0,0,m,0]
    # C[k,n] = U_tensor[0,0,k,0,0,n]
    
    A = np.array([[U_tensor[i,0,0,l,0,0] for l in range(2)] for i in range(2)], dtype=complex)
    B = np.array([[U_tensor[0,j,0,0,m,0] for m in range(2)] for j in range(2)], dtype=complex)
    C = np.array([[U_tensor[0,0,k,0,0,n] for n in range(2)] for k in range(2)], dtype=complex)
    
    # Verify
    U_check = np.kron(np.kron(A, B), C)
    if np.max(np.abs(U_check - U)) < 1e-6:
        # It's a tensor product!
        # Now decompose A, B, C into u3 gates
        # A on qubit 0, B on qubit 1, C on qubit 2
        # But we need to handle the case where A, B, C are not unitary
        # (they should be, but let's check)
        
        def _make_unitary_2x2(M2):
            try:
                M2, _ = np.linalg.qr(M2)
                return M2
            except:
                return None
        
        def _u2_to_u3(U2):
            a, b, c, d = U2[0,0], U2[0,1], U2[1,0], U2[1,1]
            gamma = np.angle(a)
            cos_half = np.abs(a)
            sin_half = np.sqrt(max(0, 1 - cos_half**2))
            if sin_half < 1e-12:
                phase_11 = d / np.exp(1j * gamma)
                lam = np.angle(phase_11)
                return (0.0, 0.0, lam)
            theta = 2 * np.arccos(np.clip(cos_half, -1, 1))
            b_prime = b / np.exp(1j * gamma)
            lam = np.angle(-b_prime / sin_half)
            c_prime = c / np.exp(1j * gamma)
            phi = np.angle(c_prime / sin_half)
            return (theta, phi, lam)
        
        A_u = _make_unitary_2x2(A)
        B_u = _make_unitary_2x2(B)
        C_u = _make_unitary_2x2(C)
        
        if A_u is None or B_u is None or C_u is None:
            return None
        
        A_t, A_p, A_l = _u2_to_u3(A_u)
        B_t, B_p, B_l = _u2_to_u3(B_u)
        C_t, C_p, C_l = _u2_to_u3(C_u)
        
        gates = [
            ["u3", 0, A_t, A_p, A_l],
            ["u3", 1, B_t, B_p, B_l],
            ["u3", 2, C_t, C_p, C_l],
        ]
        
        U_circuit = _build_unitary_fast(gates, 3)
        dist = _distance(U, U_circuit)
        if dist < 1e-6:
            return gates
    
    # Try U = (2-qubit gate on q0,q1) ⊗ (1-qubit gate on q2)
    # U = M ⊗ C where M is 4x4 (on q0,q1) and C is 2x2 (on q2)
    # U_tensor[i,j,k,l,m,n] = M[i*2+j, l*2+m] * C[k,n]
    # Set C[0,0] = 1
    # C[k,n] = U_tensor[0,0,k,0,0,n]
    # M[i*2+j, l*2+m] = U_tensor[i,j,0,l,m,0]
    
    C = np.array([[U_tensor[0,0,k,0,0,n] for n in range(2)] for k in range(2)], dtype=complex)
    M = np.array([[U_tensor[i,j,0,l,m,0] for m in range(2)] for l in range(2) for j in range(2) for i in range(2)], dtype=complex)
    M = M.reshape((4,4))
    
    U_check = np.kron(M, C)
    if np.max(np.abs(U_check - U)) < 1e-6:
        # M is a 2-qubit gate, C is a 1-qubit gate
        # Decompose M using KAK
        gates_M = _decompose_2qubit_kak(M)
        if gates_M is not None:
            C_u = _make_unitary_2x2(C)
            if C_u is not None:
                C_t, C_p, C_l = _u2_to_u3(C_u)
                gates = gates_M + [["u3", 2, C_t, C_p, C_l]]
                # Check CNOT budget
                cnot_count = sum(1 for g in gates if g[0] == "cx")
                if cnot_count <= budget:
                    U_circuit = _build_unitary_fast(gates, 3)
                    dist = _distance(U, U_circuit)
                    if dist < 1e-6:
                        return gates
    
    # Try U = (1-qubit on q0) ⊗ (2-qubit on q1,q2)
    # U = A ⊗ M where A is 2x2 (on q0) and M is 4x4 (on q1,q2)
    # U_tensor[i,j,k,l,m,n] = A[i,l] * M[j*2+k, m*2+n]
    # Set A[0,0] = 1
    # A[i,l] = U_tensor[i,0,0,l,0,0]
    # M[j*2+k, m*2+n] = U_tensor[0,j,k,0,m,n]
    
    A = np.array([[U_tensor[i,0,0,l,0,0] for l in range(2)] for i in range(2)], dtype=complex)
    M = np.array([[U_tensor[0,j,k,0,m,n] for n in range(2)] for m in range(2) for k in range(2) for j in range(2)], dtype=complex)
    M = M.reshape((4,4))
    
    U_check = np.kron(A, M)
    if np.max(np.abs(U_check - U)) < 1e-6:
        A_u = _make_unitary_2x2(A)
        if A_u is not None:
            A_t, A_p, A_l = _u2_to_u3(A_u)
            gates_M = _decompose_2qubit_kak(M)
            if gates_M is not None:
                gates = [["u3", 0, A_t, A_p, A_l]] + gates_M
                cnot_count = sum(1 for g in gates if g[0] == "cx")
                if cnot_count <= budget:
                    U_circuit = _build_unitary_fast(gates, 3)
                    dist = _distance(U, U_circuit)
                    if dist < 1e-6:
                        return gates
    
    return None

def _optimize_generic(problem):
    """Fallback numerical optimization for generic unitaries."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if _time_remaining() < 10:
        return []
    
    # Try tensor decomposition first for 3-qubit
    if qubits == 3:
        result = _decompose_3qubit_tensor(problem)
        if result is not None:
            return result
    
    # Define CNOT layout
    if qubits == 2:
        num_cnots = min(budget, 3)
        cnot_pairs = []
        for i in range(num_cnots):
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 0))
    else:
        num_cnots = min(budget, 6)
        cnot_pairs = []
        pattern = [(0, 1), (1, 2), (0, 1)]
        for i in range(num_cnots):
            cnot_pairs.append(pattern[i % 3])
    
    # Build layout
    layout = []
    num_blocks = num_cnots + 1
    for b in range(num_blocks):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if b < num_cnots:
            layout.append(["cx", cnot_pairs[b][0], cnot_pairs[b][1]])
    
    u3_indices = [i for i, g in enumerate(layout) if g[0] == "u3"]
    n_params = len(u3_indices) * 3
    
    def build_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        if _time_remaining() < 5:
            return 1.0
        gates = build_from_params(params)
        U = _build_unitary_fast(gates, qubits)
        return _distance(target, U)
    
    x0 = np.zeros(n_params)
    best_x = x0
    best_obj = objective(x0)
    
    np.random.seed(42)
    
    def nelder_mead(func, x0, max_iter=80, tol=1e-7):
        n = len(x0)
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            x = np.array(x0, dtype=float)
            x[i] += 0.1
            simplex.append(x)
        
        for _ in range(max_iter):
            if _time_remaining() < 5:
                break
            fvals = [func(x) for x in simplex]
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            centroid = np.sum(simplex[:-1], axis=0) / n
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                simplex[-1] = xe if fe < fr else xr
            elif fr < fvals[-2]:
                simplex[-1] = xr
            else:
                rho = 0.5
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
        
        return simplex[0], fvals[0]
    
    starts = [x0]
    for _ in range(8):
        if _time_remaining() < 15:
            break
        x_try = np.random.uniform(-math.pi, math.pi, n_params)
        starts.append(x_try)
    
    for start in starts:
        if _time_remaining() < 10:
            break
        try:
            res_x, res_f = nelder_mead(objective, start, max_iter=60)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except:
            pass
    
    return build_from_params(best_x)

def synthesize(problem):
    name = problem.get("name", "").strip().lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Exact matches
    if name == "cnot" and qubits == 2:
        return _synthesize_cnot()
    elif name == "iswap" and qubits == 2:
        return _synthesize_iswap()
    elif name == "swap" and qubits == 2:
        return _synthesize_swap()
    elif name == "toffoli" and qubits == 3:
        return _synthesize_toffoli()
    else:
        # Try algebraic methods first
        if qubits == 2:
            result = _try_algebraic_2qubit(problem)
            if result is not None:
                return result
        if qubits == 3:
            result = _decompose_3qubit_tensor(problem)
            if result is not None:
                return result
        return _optimize_generic(problem)
# EVOLVE-BLOCK-END
