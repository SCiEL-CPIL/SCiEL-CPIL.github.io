"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.linalg import svd, qr

# Global time tracking
_START_TIME = time.time()
TIME_LIMIT = 240  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
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


def _u3_from_matrix(M):
    """Extract u3 parameters (theta, phi, lam) from a 2x2 unitary matrix."""
    # M = [[a, -e^{i*lam}*b], [e^{i*phi}*b, e^{i*(phi+lam)}*a]]
    # where a = cos(theta/2), b = sin(theta/2)
    a = M[0, 0]
    b_val = -M[0, 1] / np.exp(1j * 0)  # will fix below
    
    # theta/2 from |a|
    cos_half = abs(a)
    sin_half = abs(M[0, 1])
    if cos_half > 0.9999:
        theta = 0.0
        phi = 0.0
        lam = 0.0
        # Still need to capture the phase
        # M ~ [[e^{i*alpha}, 0], [0, e^{i*beta}]]
        # u3(0, phi, lam) = [[1, 0], [0, e^{i*(phi+lam)}]]
        # So we need e^{i*alpha} ~ 1 and e^{i*beta} ~ e^{i*(phi+lam)}
        # Set phi=0, lam = arg(M[1,1]) - arg(M[0,0])... but u3(0,0,lam) = diag(1, e^{i*lam})
        # So global phase matters. Let's handle this differently.
        # Actually for our purposes, we just need the relative phases.
        # u3(0, phi, lam) = diag(1, e^{i(phi+lam)})
        # So we want e^{i(phi+lam)} = M[1,1]/M[0,0]
        phase = np.angle(M[1, 1] / M[0, 0]) if abs(M[0, 0]) > 1e-10 else 0.0
        phi = 0.0
        lam = phase
        return theta, phi, lam
    elif sin_half < 0.001:
        theta = math.pi
        # M ~ [[0, -e^{i*lam}], [e^{i*phi}, 0]]
        lam = -np.angle(M[0, 1])
        phi = np.angle(M[1, 0])
        return theta, phi, lam
    else:
        theta = 2.0 * math.atan2(sin_half, cos_half)
        # phi from M[1,0] / (sin_half)
        # M[1,0] = e^{i*phi} * sin_half
        phi = np.angle(M[1, 0])
        # lam from M[0,1]
        # M[0,1] = -e^{i*lam} * sin_half
        lam = -np.angle(M[0, 1])
        return theta, phi, lam


def _knot_2qubit(U):
    """
    KAK decomposition for 2-qubit unitary.
    Returns list of gates implementing U using at most 3 CNOTs.
    """
    # The KAK decomposition: U = (A ⊗ B) * exp(i*(c1*XX + c2*YY + c3*ZZ)) * (C ⊗ D)
    # The 3-CNOT circuit for exp(i*(c1*XX + c2*YY + c3*ZZ)) is:
    #   u3 on q0, u3 on q1
    #   CX 0->1
    #   u3 on q0, u3 on q1
    #   CX 1->0
    #   u3 on q0, u3 on q1
    #   CX 0->1
    #   u3 on q0, u3 on q1
    
    # To find A, B, C, D and c1, c2, c3:
    # We use the fact that for the KAK form, the "canonical" part has a specific structure.
    
    # Step 1: Find the KAK parameters using the standard algorithm
    # Compute M = U * (Z⊗Z) * U^† and find its eigenvalues
    
    # Actually, let me use a more direct approach:
    # For 2 qubits, we can use the following algorithm:
    # 1. Compute the "interaction" part
    # 2. Extract single-qubit parts
    
    # Let me use the standard KAK algorithm:
    # The KAK parameters can be found from the eigenvalues of:
    # W = U * (Z⊗Z) * U^†
    # The eigenvalues of W are e^{±2i*c1}, e^{±2i*c2}, e^{±2i*c3}, 1 (or similar)
    
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    ZZ = np.kron(Z, Z)
    
    W = U @ ZZ @ U.conj().T
    
    # Eigenvalues of W
    eigvals = np.linalg.eigvals(W)
    
    # The KAK parameters are related to the eigenvalues
    # For a 2-qubit unitary in KAK form, eigenvalues of W are:
    # e^{2i(c1+c2+c3)}, e^{2i(c1-c2-c3)}, e^{2i(-c1+c2-c3)}, e^{2i(-c1-c2+c3)}
    
    # Sort eigenvalues and extract phases
    phases = np.angle(eigvals)
    
    # The four phases should be:
    # 2(c1+c2+c3), 2(c1-c2-c3), 2(-c1+c2-c3), 2(-c1-c2+c3)
    
    # Let p1, p2, p3, p4 be the sorted phases
    # c1 = (p1 + p2 + p3 + p4) / 8  ... this is getting complicated
    
    # Let me use a different approach: direct optimization of the KAK parameters
    # with a good initial guess.
    
    # Actually, let me use a simpler but effective approach:
    # Use the "peeling" method for 2 qubits too.
    
    # Peel method for 2 qubits:
    # U = (A ⊗ B) * V where V has a specific structure
    # We can find A and B by looking at the 2x2 blocks of U.
    
    # U is 4x4. Write it as 2x2 blocks of 2x2:
    # U = [[U00, U01], [U10, U11]]
    # where each block is 2x2.
    
    # The first qubit (qubit 0, MSB) selects rows 0-1 vs 2-3.
    # U00 = U[0:2, 0:2], U01 = U[0:2, 2:4]
    # U10 = U[2:4, 0:2], U11 = U[2:4, 2:4]
    
    # For the KAK form, we can peel off single-qubit gates.
    
    # Let me implement a direct 3-CNOT synthesis using optimization
    # with a very good initial guess.
    
    # Actually, let me use the following approach which is known to work:
    # Any 2-qubit unitary can be written as:
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F) * CNOT * (G ⊗ H)
    # with at most 3 CNOTs.
    
    # I'll use a parameterized form and optimize.
    
    # Parameter count: 8 single-qubit gates * 3 params = 24 params
    # This is too many for coordinate descent in reasonable time.
    
    # Better approach: Use the fact that we can peel off one single-qubit gate at a time.
    
    # Peel algorithm for 2 qubits:
    # Step 1: Find a single-qubit gate on qubit 0 that simplifies U.
    # Step 2: Find a single-qubit gate on qubit 1.
    # Repeat.
    
    # Let me implement a cleaner version using SVD-based peeling.
    
    # For 2 qubits, I'll use the following known result:
    # U can be decomposed with at most 3 CNOTs.
    # The decomposition is:
    # U = (A⊗B) * K(c1,c2,c3) * (C⊗D)
    # where K(c1,c2,c3) = exp(i*c1*X⊗X) * exp(i*c2*Y⊗Y) * exp(i*c3*Z⊗Z)
    
    # K(c1,c2,c3) in terms of CNOTs:
    # This is a standard 3-CNOT circuit.
    
    # To find A, B, C, D, c1, c2, c3:
    # We use the following algorithm from the literature.
    
    # Compute:
    # V = U * (I⊗Z)
    # Then find the "interaction" part.
    
    # Let me just use a well-known implementation:
    
    # The KAK parameters can be found as follows:
    # Let U be the 2-qubit unitary.
    # Compute M = U * (Z⊗Z)
    # The eigenvalues of M*M† should give us the KAK parameters.
    
    # Actually, I'll use a different well-known method:
    # The "canonical" 2-qubit gate is:
    # K = exp(i*(c1*X⊗X + c2*Y⊗Y + c3*Z⊗Z))
    # 
    # The circuit for K with 3 CNOTs is:
    #   u3(q0, 2*c1, 0, 0), u3(q1, 2*c2, 0, 0)  [approximately]
    #   CX(0,1)
    #   u3(q0, 2*c3, 0, 0), u3(q1, -2*c3, 0, 0)
    #   CX(1,0)
    #   u3(q0, 2*c3, 0, 0), u3(q1, 2*c3, 0, 0)
    #   CX(0,1)
    #   u3(q0, -2*c1, 0, 0), u3(q1, -2*c2, 0, 0)
    
    # This is getting complex. Let me use a direct approach:
    # I'll implement the KAK decomposition using the method from
    # "Synthesis of Two-Qubit Quantum Gates" by Shende, Markov, Bullock.
    
    # The key insight is that we can write:
    # U = (A⊗B) * exp(i*theta1*X⊗X) * exp(i*theta2*Y⊗Y) * exp(i*theta3*Z⊗Z) * (C⊗D)
    
    # And find A, B, C, D, theta1, theta2, theta3.
    
    # Algorithm:
    # 1. Compute the "interaction" part of U
    # 2. Extract the KAK parameters
    # 3. Find the single-qubit gates
    
    # Step 1: Compute W = U * (Z⊗Z) * U†
    # The eigenvalues of W determine the KAK parameters.
    
    eigvals = np.linalg.eigvals(W)
    # Sort by angle
    angles = np.angle(eigvals)
    
    # The KAK parameters satisfy:
    # The four eigenvalue phases are:
    # 2(c1+c2+c3), 2(c1-c2-c3), 2(-c1+c2-c3), 2(-c1-c2+c3)
    
    # Let's denote them as a, b, c, d (sorted)
    # Then:
    # c1 = (a + b + c + d) / 8  ... no, this isn't right.
    
    # Let me use a different approach. I'll directly optimize the 3-CNOT circuit
    # with a good initial guess based on the trace.
    
    # For now, let me use a simpler but effective method:
    # Use 3 CNOTs with optimized single-qubit gates.
    # The layout is:
    # U = S0 * CX(0,1) * S1 * CX(1,0) * S2 * CX(0,1) * S3
    # where Si are single-qubit gates on both qubits.
    
    # That's 4 * 2 * 3 = 24 parameters.
    
    # Let me use a smarter approach: fix the CNOT structure and optimize
    # only the single-qubit gates using a gradient-free method with
    # a good initial guess.
    
    # Actually, let me use the following known exact method:
    
    # For 2 qubits, we can use the following decomposition:
    # U = (A⊗B) * K * (C⊗D)
    # where K is a "canonical" 2-qubit gate.
    
    # The canonical gate K(c1,c2,c3) can be implemented with 3 CNOTs:
    # K = u3(q0, 2c1, 0, 0) u3(q1, 2c2, 0, 0) CX(0,1) u3(q0, 2c3, 0, 0) u3(q1, -2c3, 0, 0) CX(1,0) u3(q0, 2c3, 0, 0) u3(q1, 2c3, 0, 0) CX(0,1) u3(q0, -2c1, 0, 0) u3(q1, -2c2, 0, 0)
    
    # To find c1, c2, c3 and A, B, C, D:
    # We use the following algorithm:
    
    # 1. Compute the "interaction" part:
    #    V = U * (I⊗Z)
    #    The "local" parts are extracted from V.
    
    # This is getting too complex for a quick implementation. Let me use
    # a numerical approach with a very good initial guess.
    
    # I'll use the following strategy:
    # 1. Start with the identity as initial guess for single-qubit gates
    # 2. Use the trace to get a good initial estimate
    # 3. Use Nelder-Mead optimization
    
    # Actually, let me just implement a clean 3-CNOT synthesis using
    # the peeling method.
    
    # Peeling method for 2 qubits:
    # U is 4x4. We want to write U = (A⊗I) * V1 where V1 has a simpler form.
    
    # U = [[U00, U01], [U10, U11]] where each is 2x2.
    
    # If we apply (A⊗I) from the left, we get:
    # (A⊗I)*U = [[A*U00, A*U01], [U10, U11]]
    
    # We want to choose A such that the resulting matrix has a simpler structure.
    # For example, we want A*U00 and A*U01 to be proportional.
    
    # This is equivalent to finding A such that:
    # A*U00 = lambda * A*U01 for some lambda, or
    # A*U00 * (A*U01)† is proportional to the identity.
    
    # Let me use SVD:
    # U00† * U01 = A† * (A*U00)† * (A*U01) = (A*U01) * (A*U00)†
    
    # This is getting complicated. Let me use a different approach.
    
    # I'll use the following well-known result:
    # Any 2-qubit unitary can be decomposed as:
    # U = (A⊗B) * CNOT * (C⊗D) * CNOT * (E⊗F) * CNOT * (G⊗H)
    # with at most 3 CNOTs, and the single-qubit gates can be found
    # using the following algorithm:
    
    # 1. Compute U1 = U * (I⊗H)
    # 2. Extract the first single-qubit gate
    # 3. Continue peeling
    
    # Let me implement this step by step.
    
    # Actually, I think the cleanest approach for this problem is:
    # Use the fact that the generic 2-qubit unitaries in the problem
    # likely have small CNOT budgets (3 or 4), and use a direct
    # numerical optimization with a good layout.
    
    # Let me use a 3-CNOT layout and optimize with Nelder-Mead.
    
    # Layout:
    # u3(q0), u3(q1), CX(0,1), u3(q0), u3(q1), CX(1,0), u3(q0), u3(q1), CX(0,1), u3(q0), u3(q1)
    
    # That's 6 single-qubit gates * 3 params = 18 parameters.
    
    # For the 3-qubit case, I'll use a similar approach with the given CNOT budget.
    
    # Let me implement this properly.
    
    return None  # Placeholder


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 2 CNOTs + single qubit rotations."""
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # Standard 2-CNOT decomposition
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
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def _peel_single_qubit(U, qubit, num_qubits):
    """
    Peel off a single-qubit gate from qubit in the unitary U.
    Returns (gate_params, remaining_unitary) where gate_params is (theta, phi, lam)
    and remaining_unitary is U with the single-qubit gate removed from the left.
    """
    n = 2 ** num_qubits
    half = n // 2
    
    # U is n x n. We want to find a 2x2 unitary A such that
    # (A ⊗ I) * U has a simpler structure.
    
    # The key idea: look at the 2x2 blocks corresponding to the qubit.
    # For qubit q, the unitary can be written as:
    # U = sum_{i,j} |i><j|_q ⊗ U_{ij}
    # where U_{ij} are (n/2) x (n/2) matrices.
    
    # We want to find A such that A * U_{00} and A * U_{01} are "aligned"
    # in some sense.
    
    # Extract the blocks:
    # For qubit q (0-indexed from MSB), the basis is:
    # |q=0, rest> and |q=1, rest>
    # The blocks are:
    # U_{00} = U[rest where q=0, rest where q=0]
    # U_{01} = U[rest where q=0, rest where q=1]
    # etc.
    
    # For qubit 0 (MSB):
    # U_{00} = U[0:half, 0:half]
    # U_{01} = U[0:half, half:n]
    # U_{10} = U[half:n, 0:half]
    # U_{11} = U[half:n, half:n]
    
    # For general qubit q:
    # The indices where qubit q is 0: those with bit (num_qubits-1-q) = 0
    # The indices where qubit q is 1: those with bit (num_qubits-1-q) = 1
    
    bit_pos = num_qubits - 1 - qubit
    indices_0 = [i for i in range(n) if (i >> bit_pos) & 1 == 0]
    indices_1 = [i for i in range(n) if (i >> bit_pos) & 1 == 1]
    
    U00 = U[np.ix_(indices_0, indices_0)]
    U01 = U[np.ix_(indices_0, indices_1)]
    U10 = U[np.ix_(indices_1, indices_0)]
    U11 = U[np.ix_(indices_1, indices_1)]
    
    # We want to find A (2x2 unitary) such that:
    # The "coherence" between the two subspaces is minimized after applying A.
    # Specifically, we want A such that:
    # (A⊗I) U has U'_{01} and U'_{10} as small as possible.
    
    # This is equivalent to finding A that diagonalizes:
    # M = U00† @ U01
    # using SVD: M = V_svd @ S @ W_svd†
    # Then A = V_svd† (or some variant)
    
    # Actually, the standard peeling uses:
    # Compute M = U00† @ U01
    # SVD: M = U_svd @ S @ V_svd†
    # Then the peeled gate is related to U_svd.
    
    # Let me use a simpler approach:
    # Compute the "off-diagonal" part and try to minimize it.
    
    # The Frobenius norm of the off-diagonal blocks after applying A:
    # ||A @ U01||_F^2 + ||A @ U10||_F^2 (approximately)
    
    # Actually, let me use the following known result:
    # The optimal A is given by the SVD of U00† @ U01.
    
    M = U00.conj().T @ U01
    
    # SVD
    U_svd, S, V_svd = svd(M)
    
    # The peeled gate is:
    # A = U_svd (up to a phase)
    
    # Let me verify: if we apply (U_svd† ⊗ I) from the left:
    # New U00' = U_svd† @ U00
    # New U01' = U_svd† @ U01 = U_svd† @ U00 @ U00† @ U01 = U_svd† @ U00 @ M
    # = U_svd† @ U00 @ U_svd @ S @ V_svd†
    
    # This doesn't immediately simplify things. Let me use a different approach.
    
    # I'll use the following method:
    # 1. Compute the 2x2 matrix:
    #    C = [[<U00, U00>, <U00, U01>], [<U10, U00>, <U10, U01>]]
    #    where <A, B> = tr(A† @ B)
    # 2. Find the unitary A that diagonalizes C.
    
    C00 = np.trace(U00.conj().T @ U00)
    C01 = np.trace(U00.conj().T @ U01)
    C10 = np.trace(U10.conj().T @ U00)
    C11 = np.trace(U10.conj().T @ U01)
    
    C = np.array([[C00, C01], [C10, C11]], dtype=complex)
    
    # Diagonalize C: C = V @ D @ V†
    eigvals, eigvecs = np.linalg.eig(C)
    
    # The optimal A is V† (or V, depending on convention)
    A = eigvecs.conj().T
    
    # Normalize A to be unitary
    # A should already be unitary since C is Hermitian (approximately)
    
    # Extract u3 parameters from A
    theta, phi, lam = _u3_from_matrix(A)
    
    # Apply (A† ⊗ I) from the left to get the remaining unitary
    # Actually, we want to "peel off" A, so the remaining unitary is:
    # U' = (A† ⊗ I) @ U
    
    # Build (A† ⊗ I)
    A_dag = A.conj().T
    I_half = np.eye(half, dtype=complex)
    left_op = np.kron(A_dag, I_half)
    
    U_remaining = left_op @ U
    
    return (theta, phi, lam), U_remaining


def _synthesize_2qubit_kak(U, budget):
    """
    Synthesize a 2-qubit unitary using KAK decomposition.
    Uses at most 3 CNOTs.
    """
    if budget < 1:
        # Can't do much, just return identity-like
        return [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
    
    # Use peeling method
    # Peel off single-qubit gates from U until we have a "simple" 2-qubit gate
    # that can be implemented with CNOTs.
    
    gates = []
    remaining = U.copy()
    
    # Peel up to 4 single-qubit gates (2 on each qubit, before and after CNOTs)
    # Layout: S0 * CX * S1 * CX * S2 * CX * S3 (up to 3 CNOTs)
    
    # For budget CNOTs, we peel off (budget+1) layers of single-qubit gates.
    
    # Let's peel off single-qubit gates from the left:
    # We'll peel off up to 2 gates on each qubit.
    
    # First, peel off a gate on qubit 0:
    for i in range(min(2, budget + 1)):
        if _time_remaining() < 10:
            break
        params, remaining = _peel_single_qubit(remaining, 0, 2)
        gates.append(["u3", 0, params[0], params[1], params[2]])
    
    # Now the remaining unitary should be "simpler".
    # Let's try to implement it with CNOTs and more single-qubit gates.
    
    # For now, let's just add the CNOTs and try to match.
    # This is a rough approach; let me use a better one.
    
    # Actually, let me use a direct optimization approach for the 2-qubit case.
    # I'll use a 3-CNOT layout and optimize the single-qubit gates.
    
    # Layout for 3 CNOTs:
    # u3(q0), u3(q1), CX(0,1), u3(q0), u3(q1), CX(1,0), u3(q0), u3(q1), CX(0,1), u3(q0), u3(q1)
    
    # That's 6 single-qubit gates, 18 parameters.
    
    # Let me use Nelder-Mead from scipy.
    
    from scipy.optimize import minimize
    
    layout = []
    for i in range(3):
        for q in range(2):
            layout.append(("u3", q))
        if i == 0:
            layout.append(("cx", 0, 1))
        elif i == 1:
            layout.append(("cx", 1, 0))
        else:
            layout.append(("cx", 0, 1))
    for q in range(2):
        layout.append(("u3", q))
    
    n_params = 18
    
    def objective(values):
        gates = []
        k = 0
        for item in layout:
            if item[0] == "u3":
                gates.append(["u3", item[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", item[1], item[2]])
        matrix = _build_fast(gates, 2)
        return _distance(U, matrix)
    
    # Initial guess: zeros
    x0 = np.zeros(n_params)
    
    # Try a few starting points
    best_x = x0
    best_val = objective(x0)
    
    for start in range(3):
        if _time_remaining() < 15:
            break
        if start == 0:
            x = x0
        else:
            x = np.random.uniform(-math.pi, math.pi, n_params)
        
        try:
            result = minimize(objective, x, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except:
            pass
    
    # Build the final gate list
    gates = []
    k = 0
    for item in layout:
        if item[0] == "u3":
            gates.append(["u3", item[1], best_x[k], best_x[k+1], best_x[k+2]])
            k += 3
        else:
            gates.append(["cx", item[1], item[2]])
    
    return gates


def _synthesize_3qubit_peel(U, budget, qubits=3):
    """
    Synthesize a 3-qubit unitary using peeling method.
    """
    n = 2 ** qubits
    
    if budget < 1:
        return [["u3", q, 0, 0, 0] for q in range(qubits)]
    
    # Use peeling to extract single-qubit gates, then handle the remainder
    # with CNOTs.
    
    # For 3 qubits with budget CNOTs, we can peel off single-qubit gates
    # and use the CNOTs to handle the entanglement.
    
    # Layout: alternate CNOTs on different pairs, with single-qubit gates between.
    
    # Pairs to use: (0,1), (1,2), (0,2)
    pairs = [(0, 1), (1, 2), (0, 2), (2, 1), (1, 0), (2, 0)]
    
    # Build layout
    layout = []
    for i in range(budget):
        for q in range(qubits):
            layout.append(("u3", q))
        c, t = pairs[i % len(pairs)]
        layout.append(("cx", c, t))
    for q in range(qubits):
        layout.append(("u3", q))
    
    n_params = 3 * (budget + 1) * qubits
    
    # Use Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(values):
        gates = []
        k = 0
        for item in layout:
            if item[0] == "u3":
                gates.append(["u3", item[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", item[1], item[2]])
        matrix = _build_fast(gates, qubits)
        return _distance(U, matrix)
    
    x0 = np.zeros(n_params)
    
    best_x = x0
    best_val = objective(x0)
    
    # Try a few starting points
    max_starts = min(5, max(1, int(_time_remaining() / 10)))
    for start in range(max_starts):
        if _time_remaining() < 15:
            break
        if start == 0:
            x = x0
        else:
            x = np.random.uniform(-math.pi, math.pi, n_params)
        
        try:
            result = minimize(objective, x, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except:
            pass
    
    # Build the final gate list
    gates = []
    k = 0
    for item in layout:
        if item[0] == "u3":
            gates.append(["u3", item[1], best_x[k], best_x[k+1], best_x[k+2]])
            k += 3
        else:
            gates.append(["cx", item[1], item[2]])
    
    return gates


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try to use known exact syntheses first
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        # Generic synthesis
        if qubits == 2:
            gates = _synthesize_2qubit_kak(target, budget)
        else:
            gates = _synthesize_3qubit_peel(target, budget, qubits)
    
    # Verify the result
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    # If the result is poor, try an alternative
    if dist > 0.01:
        if qubits == 2:
            # Try with a different CNOT layout
            alt_gates = _synthesize_2qubit_kak(target, min(budget, 3))
            alt_matrix = _build_fast(alt_gates, qubits)
            alt_dist = _distance(target, alt_matrix)
            if alt_dist < dist:
                gates = alt_gates
        else:
            alt_gates = _synthesize_3qubit_peel(target, budget, qubits)
            alt_matrix = _build_fast(alt_gates, qubits)
            alt_dist = _distance(target, alt_matrix)
            if alt_dist < dist:
                gates = alt_gates
    
    return gates
# EVOLVE-BLOCK-END
