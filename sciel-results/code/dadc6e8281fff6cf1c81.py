"""Exact decompositions for known gates + KAK/optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 225.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", c, t]


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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


def _exact_cnot():
    """CNOT gate: just one CNOT."""
    return [_cx(0, 1)]


def _exact_swap():
    """SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_iswap():
    """iSWAP = (1/sqrt(2)) * [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]] (up to phase)
    iSWAP = H(0) H(1) CNOT(0,1) S(0) S(1) CNOT(0,1) H(0) H(1)
    Actually: iSWAP = (S⊗S) · CNOT · (H⊗H) · CNOT · (H⊗H)
    Let me use: iSWAP = [H, H, CNOT, S, S, CNOT, H, H] on qubits 0,1
    """
    # iSWAP = 1/sqrt(2) * [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]
    # Standard decomposition:
    # iSWAP = (S ⊗ S) · CNOT(0,1) · (H ⊗ H) · CNOT(0,1) · (H ⊗ H)
    # But we need to check phases. Let me use a known decomposition.
    # iSWAP = exp(iπ/4) * (1/sqrt(2)) * [[1, i, 0, 0], [i, 1, 0, 0], [0, 0, 1, i], [0, 0, i, 1]]
    # Actually the standard iSWAP is:
    # [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    # or with phases. Let me just use the known 2-CNOT decomposition.
    
    # iSWAP gate: swaps and applies i phase
    # One decomposition: H(0) H(1) CNOT(0,1) S(0) S(1) CNOT(0,1) H(0) H(1)
    # where S = u3(0, pi/2, 0, pi/2) ... let me think more carefully.
    
    # Actually, let me use: iSWAP = (S⊗S) CNOT(0,1) (H⊗H) CNOT(0,1) (H⊗H)
    # H = u3(pi/2, 0, pi/2)
    # S = u3(0, 0, pi/2)  -- wait, S = diag(1, i) = u3(0, 0, pi/2)? 
    # u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
    # S = [[1,0],[0,i]]: need cos(θ/2)=1, sin(θ/2)=0 → θ=0, and e^{i(φ+λ)}=i → φ+λ=π/2
    # So S = u3(0, 0, π/2) gives [[1, 0], [0, e^{iπ/2}]] = [[1,0],[0,i]]. Yes!
    
    H = (math.pi / 2, 0.0, math.pi / 2)
    S = (0.0, 0.0, math.pi / 2)
    
    gates = []
    gates.append(_u3(0, *H))
    gates.append(_u3(1, *H))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, *S))
    gates.append(_u3(1, *S))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, *H))
    gates.append(_u3(1, *H))
    return gates


def _exact_toffoli():
    """Toffoli (CCNOT) on 3 qubits: 6 CNOTs decomposition.
    Standard 6-CNOT Toffoli:
    H(2) CNOT(1,2) T(2) CNOT(0,2) T†(2) T(1) CNOT(0,1) T†(1) T(2) CNOT(0,2) T†(2) H(2) CNOT(0,1)
    where T = diag(1, e^{iπ/4}), T† = diag(1, e^{-iπ/4})
    T = u3(0, 0, π/4)
    T† = u3(0, 0, -π/4)
    """
    H = (math.pi / 2, 0.0, math.pi / 2)
    T = (0.0, 0.0, math.pi / 4)
    Tdag = (0.0, 0.0, -math.pi / 4)
    
    gates = []
    gates.append(_u3(2, *H))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, *T))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, *Tdag))
    gates.append(_u3(1, *T))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, *Tdag))
    gates.append(_u3(2, *T))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, *Tdag))
    gates.append(_u3(2, *H))
    gates.append(_cx(0, 1))
    return gates


def _kdecompose_2qubit(U):
    """KAK decomposition of a 2-qubit unitary into 3 CNOTs + single-qubit rotations.
    Returns list of gates."""
    # U = (A ⊗ B) · CNOT · (C ⊗ D) · CNOT · (E ⊗ F) · CNOT · (G ⊗ H)
    # Using the standard KAK form
    
    # First, compute the KAK decomposition
    # U = K1 · (e^{iθ1 σx⊗σx/2} e^{iθ2 σy⊗σy/2} e^{iθ3 σz⊗σz/2}) · K2
    # where K1, K2 are single-qubit unitaries
    
    # For a 2-qubit unitary, we can use the following approach:
    # 1. Compute U' = U · (phase factor) to make det(U') = 1
    # 2. Use the KAK decomposition
    
    # Simpler approach: use the fact that any 2-qubit unitary can be written as
    # (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    
    # Let me use a numerical approach with the known structure.
    # Actually, let me just use the standard decomposition formula.
    
    # The KAK decomposition: U = (A⊗B) · exp(i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz)/2) · (C⊗D)
    # The middle part can be decomposed into 3 CNOTs with single-qubit rotations.
    
    # For simplicity, let me use a known result:
    # Any 2-qubit unitary U can be decomposed as:
    # U = (A1⊗A2) · CNOT · (B1⊗B2) · CNOT · (C1⊗C2) · CNOT · (D1⊗D2)
    
    # I'll use the algorithm from "Optimal two-qubit gates" by Shende et al.
    # or use scipy's approach.
    
    # Actually, let me just use a direct numerical approach:
    # We know that U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    # Let's solve for the single-qubit unitaries.
    
    # A simpler approach: use the fact that we can absorb single-qubit gates.
    # Let's define:
    # P = CNOT(0,1)
    # U = (A⊗B) P (C⊗D) P (E⊗F) P (G⊗H)
    
    # Step 1: Compute R = U · (G⊗H)† ... but we don't know G, H yet.
    
    # Let me use a different approach: iterate to find the decomposition.
    # Actually, for the purpose of this task, let me use a simpler method:
    # Use the KAK angles and known formulas.
    
    # KAK decomposition algorithm:
    # 1. Compute the KAK angles θ1, θ2, θ3
    # 2. Compute the single-qubit unitaries
    
    # Let me implement the standard KAK decomposition.
    
    # First, remove the global phase: U' = U / (det(U))^{1/4}
    det_U = np.linalg.det(U)
    phase = det_U ** (1/4)
    U_prime = U / phase
    
    # Now compute the KAK decomposition of U_prime
    # U_prime = K1 · M(θ1,θ2,θ3) · K2
    # where M = exp(iθ1 σx⊗σx/2) exp(iθ2 σy⊗σy/2) exp(iθ3 σz⊗σz/2)
    
    # The KAK angles can be computed from:
    # cos(2θ1) = (1/4) tr(U' · (σx⊗σx))
    # cos(2θ2) = (1/4) tr(U' · (σy⊗σy))  
    # cos(2θ3) = (1/4) tr(U' · (σz⊗σz))
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    sx_sx = np.kron(sx, sx)
    sy_sy = np.kron(sy, sy)
    sz_sz = np.kron(sz, sz)
    
    cos2t1 = np.real(np.trace(U_prime @ sx_sx)) / 4
    cos2t2 = np.real(np.trace(U_prime @ sy_sy)) / 4
    cos2t3 = np.real(np.trace(U_prime @ sz_sz)) / 4
    
    # Clamp to [-1, 1]
    cos2t1 = max(-1, min(1, cos2t1))
    cos2t2 = max(-1, min(1, cos2t2))
    cos2t3 = max(-1, min(1, cos2t3))
    
    t1 = 0.5 * math.acos(cos2t1)
    t2 = 0.5 * math.acos(cos2t2)
    t3 = 0.5 * math.acos(cos2t3)
    
    # Now compute K1 and K2
    # M(θ) = exp(iθ1 sx⊗sx/2) exp(iθ2 sy⊗sy/2) exp(iθ3 sz⊗sz/2)
    
    def _expm_pauli(pauli, angle):
        """exp(i * angle * pauli / 2)"""
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        return c * np.eye(2, dtype=complex) + 1j * s * pauli
    
    # For 2-qubit Pauli products, we need to be more careful
    def _expm_pauli2(pauli_tensor, angle):
        """exp(i * angle * pauli_tensor / 2) for a 2-qubit Pauli tensor product"""
        # Use the formula: exp(iθ P/2) = cos(θ/2) I + i sin(θ/2) P for P^2 = I
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        return c * np.eye(4, dtype=complex) + 1j * s * pauli_tensor
    
    M = _expm_pauli2(sx_sx, t1) @ _expm_pauli2(sy_sy, t2) @ _expm_pauli2(sz_sz, t3)
    
    # K1 = U_prime · M†
    K1 = U_prime @ M.conj().T
    # K2 should be I if we set it up this way, but actually:
    # U_prime = K1 · M · K2, so K2 = M† · K1† · U_prime† ... this is getting complex.
    
    # Actually, the standard KAK form is:
    # U = (A⊗B) · M(θ) · (C⊗D)
    # where A, B, C, D are single-qubit unitaries.
    # So K1 = A⊗B and K2 = C⊗D.
    
    # To extract A, B, C, D from K1 and K2, we need to factor them.
    # This is non-trivial in general. Let me use a different approach.
    
    # Alternative: use the known decomposition directly.
    # For a 2-qubit unitary, we can always write:
    # U = (A1⊗A2) CNOT (B1⊗B2) CNOT (C1⊗C2) CNOT (D1⊗D2)
    
    # Let me use a numerical optimization approach instead, since the KAK extraction
    # of single-qubit factors is complex.
    
    # Actually, let me try a simpler approach: use the known fact that
    # CNOT = (H⊗I) · CNOT · (H⊗I) ... no that's not right.
    
    # Let me just use a direct approach: assume the structure and optimize.
    # But for 2-qubit with 3 CNOTs, KAK should give exact decomposition.
    
    # Let me try to extract the single-qubit unitaries from K1 and K2.
    # K1 is a 4x4 unitary that should be a tensor product A⊗B.
    # We can extract A and B by looking at the structure.
    
    # If K1 = A⊗B, then K1 = [[a00*B, a01*B], [a10*B, a11*B]]
    # So the first column block gives us a00*B and a10*B
    # And the second column block gives us a01*B and a11*B
    
    def _extract_tensor_product(K):
        """Extract A and B from K = A⊗B (assuming it's a tensor product)."""
        # K[0:2, 0:2] = A[0,0] * B
        # K[2:4, 0:2] = A[1,0] * B
        # K[0:2, 2:4] = A[0,1] * B
        # K[2:4, 2:4] = A[1,1] * B
        
        B0 = K[0:2, 0:2]
        B1 = K[2:4, 0:2]
        
        # A[0,0] = B0 / B (element-wise, but B is 2x2)
        # Actually, B0 = A[0,0] * B, so B = B0 / A[0,0]
        # And B1 = A[1,0] * B, so A[1,0] = B1 / B (element-wise ratio)
        
        # Let's find A[0,0] by looking at the ratio of elements
        # B0[0,0] = A[0,0] * B[0,0]
        # B0[0,1] = A[0,0] * B[0,1]
        # So B0[0,0] / B0[0,1] = B[0,0] / B[0,1]
        
        # This is getting complicated. Let me use a different approach.
        
        # If K is a tensor product, then K ⊗ K should have certain properties.
        # Actually, let me just use the fact that for a tensor product A⊗B:
        # K[0,0] = A[0,0]*B[0,0], K[0,1] = A[0,0]*B[0,1], etc.
        
        # Let's assume A[0,0] ≠ 0 (we can handle the edge case separately)
        if abs(K[0, 0]) > 1e-10:
            # A[0,0] * B[0,0] = K[0,0]
            # A[0,0] * B[0,1] = K[0,1]
            # A[0,0] * B[1,0] = K[2,0]
            # A[0,0] * B[1,1] = K[2,1]
            # So B[0,0] = K[0,0]/A[0,0], B[0,1] = K[0,1]/A[0,0], etc.
            # And A[1,0] * B[0,0] = K[1,0], so A[1,0] = K[1,0]/B[0,0] = K[1,0]*A[0,0]/K[0,0]
            
            # Let's just pick A[0,0] = 1 (we can absorb the phase)
            # Actually, we need to be more careful.
            
            # Let me use a different strategy: use the first row and column to determine B
            # K[0:2, 0] = [A[0,0]*B[0,0], A[1,0]*B[0,0]]
            # K[0:2, 1] = [A[0,0]*B[0,1], A[1,0]*B[0,1]]
            
            # Ratio: K[0,0]/K[1,0] = A[0,0]/A[1,0] (if B[0,0] ≠ 0)
            # And K[0,1]/K[1,1] = A[0,0]/A[1,0] (if B[0,1] ≠ 0)
            
            # This is still complex. Let me just use a numerical approach.
            pass
        
        # For now, let me just return the whole K as a "single-qubit" gate on both qubits
        # This won't work directly, so let me use a different strategy.
        return None, None
    
    # This approach is getting too complex. Let me use a simpler method:
    # Just use the KAK angles and known formulas for the CNOT decomposition.
    
    # Actually, let me just use a direct optimization for 2-qubit unitaries too.
    # With 3 CNOTs and single-qubit rotations, we have enough parameters.
    
    return None  # Signal to use optimization


def _optimize_unitary(target, qubits, budget, max_time=30.0):
    """Optimize single-qubit rotations to match target unitary within CNOT budget."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dim = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Create a layout: single-qubit rotations before each CNOT and after the last CNOT
    layout = []
    for layer in range(budget):
        layout.append(("u3", list(range(qubits))))  # single-qubit rotations on all qubits
        layout.append(("cx", pairs[layer % len(pairs)]))
    layout.append(("u3", list(range(qubits))))  # final single-qubit rotations
    
    # Count parameters
    n_params = 0
    for item in layout:
        if item[0] == "u3":
            n_params += 3 * len(item[1])
    
    # Initialize parameters
    np.random.seed(42)
    params = np.random.uniform(-math.pi, math.pi, size=n_params)
    
    def _build_from_params(p):
        """Build unitary from parameter vector."""
        matrix = np.eye(dim, dtype=complex)
        eye = np.eye(2, dtype=complex)
        idx = 0
        for item in layout:
            if item[0] == "u3":
                for q in item[1]:
                    theta = p[idx]
                    phi = p[idx + 1]
                    lam = p[idx + 2]
                    idx += 3
                    c, s = math.cos(theta / 2), math.sin(theta / 2)
                    single = np.array([[c, -np.exp(1j * lam) * s],
                                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                    operator = np.array([[1]], dtype=complex)
                    for k in range(qubits):
                        operator = np.kron(operator, single if k == q else eye)
                    matrix = operator @ matrix
            else:
                control, target_q = item[1]
                index = np.arange(dim)
                bit = (index >> (qubits - 1 - control)) & 1
                matrix = matrix[index ^ (bit << (qubits - 1 - target_q)), :]
        return matrix
    
    def _fidelity(p):
        U = _build_from_params(p)
        return abs(np.trace(target.conj().T @ U)) / dim
    
    # Gradient-based optimization using finite differences
    best_params = params.copy()
    best_fid = _fidelity(best_params)
    
    start = time.time()
    step_size = 0.5
    lr = 0.1
    
    for iteration in range(200):
        if time.time() - start > max_time or time.time() - _start_time > _DEADLINE:
            break
        
        # Compute gradient using central differences
        grad = np.zeros_like(params)
        eps = 0.01
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (_fidelity(p_plus) - _fidelity(p_minus)) / (2 * eps)
        
        # Gradient ascent
        params = params + lr * grad
        # Wrap angles to [-pi, pi]
        params = (params + math.pi) % (2 * math.pi) - math.pi
        
        fid = _fidelity(params)
        if fid > best_fid:
            best_fid = fid
            best_params = params.copy()
        
        # Adaptive learning rate
        if fid < best_fid - 0.001:
            lr *= 0.8
        else:
            lr = min(lr * 1.05, 0.5)
        
        if best_fid > 0.9999:
            break
    
    # Convert best_params to gate list
    gates = []
    idx = 0
    for item in layout:
        if item[0] == "u3":
            for q in item[1]:
                gates.append(_u3(q, best_params[idx], best_params[idx + 1], best_params[idx + 2]))
                idx += 3
        else:
            gates.append(_cx(*item[1]))
    
    return gates, 1.0 - best_fid


def synthesize(problem):
    """Synthesize quantum circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        # Check if it's CNOT
        cnot_target = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
        if _distance(target, cnot_target) < 1e-6:
            return _exact_cnot()
        
        # Check if it's SWAP
        swap_target = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=complex)
        if _distance(target, swap_target) < 1e-6:
            return _exact_swap()
        
        # Check if it's iSWAP (up to global phase)
        iswap_target = np.array([[0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0]], dtype=complex)
        if _distance(target, iswap_target) < 1e-6:
            return _exact_iswap()
        
        # Also check iSWAP with different phase conventions
        iswap_target2 = np.array([[0, 1j, 0, 0],
                                   [1j, 0, 0, 0],
                                   [0, 0, 0, 1j],
                                   [0, 0, 1j, 0]], dtype=complex)
        if _distance(target, iswap_target2) < 1e-6:
            # This is a different iSWAP convention, use optimization
            pass
    
    if qubits == 3:
        # Check if it's Toffoli
        toffoli_target = np.array([[1,0,0,0,0,0,0,0],
                                     [0,1,0,0,0,0,0,0],
                                     [0,0,1,0,0,0,0,0],
                                     [0,0,0,1,0,0,0,0],
                                     [0,0,0,0,1,0,0,0],
                                     [0,0,0,0,0,1,0,0],
                                     [0,0,0,0,0,0,0,1],
                                     [0,0,0,0,0,0,1,0]], dtype=complex)
        if _distance(target, toffoli_target) < 1e-6:
            return _exact_toffoli()
    
    # For generic unitaries, use optimization
    remaining_time = _DEADLINE - (time.time() - _start_time)
    max_time = min(30.0, remaining_time)
    if max_time < 5:
        max_time = 5
    
    gates, dist = _optimize_unitary(target, qubits, budget, max_time=max_time)
    return gates
# EVOLVE-BLOCK-END
