import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def coefficients(stages):
    start_time = time.time()
    deadline = 140.0
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    
    U_exact_cache = {}
    for t in times:
        exp_h = np.exp(-1j * t * H_eigvals)
        U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T

    # 6th-order Suzuki-Trotter coefficients
    c = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 - c)
    c2 = -c / (2.0 - c)
    c3 = c1
    
    # 6th order scheme: S4(gamma t) S4((1-2gamma)t) S4(gamma t)
    # gamma = c1
    # A coeffs: 13 factors
    # B coeffs: 12 factors
    # We need 12 A factors.
    # Merge first and last A factors.
    
    # A factors from S4: [c1, c3, c1] scaled by time factor
    # Block 1 (time gamma t): A: [gamma*c1, gamma*c3, gamma*c1], B: [gamma*c2, gamma*c2]
    # Block 2 (time (1-2gamma)t): A: [(1-2gamma)*c1, (1-2gamma)*c3, (1-2gamma)*c1], B: [(1-2gamma)*c2, (1-2gamma)*c2]
    # Block 3 (time gamma t): A: [gamma*c1, gamma*c3, gamma*c1], B: [gamma*c2, gamma*c2]
    
    # Total A sequence (13 factors):
    # 1. gamma*c1
    # 2. gamma*c3
    # 3. gamma*c1 + (1-2gamma)*c1 = (1-gamma)*c1
    # 4. (1-2gamma)*c3
    # 5. (1-2gamma)*c1 + gamma*c1 = (1-gamma)*c1
    # 6. gamma*c3
    # 7. gamma*c1
    # Symmetric: 1-7-1...
    # Wait, the sequence is:
    # B1: B(gamma*c2) A(gamma*c1) B(gamma*c2) A(gamma*c3) B(gamma*c2) A(gamma*c1)
    # B2: B((1-2gamma)*c2) A((1-2gamma)*c1) B((1-2gamma)*c2) A((1-2gamma)*c3) B((1-2gamma)*c2) A((1-2gamma)*c1)
    # B3: B(gamma*c2) A(gamma*c1) B(gamma*c2) A(gamma*c3) B(gamma*c2) A(gamma*c1)
    
    # The product is B A B A B A B A B A B A B A B A B A B A B A B A
    # This is 12 B and 13 A.
    
    a_13 = [
        gamma_c1 := c1 * c1,
        c3 * c1,
        c1 * (1 - 2 * c1) + c1 * c1, # (1-c1)*c1
        c3 * (1 - 2 * c1),
        c1 * (1 - 2 * c1) + c1 * c1, # (1-c1)*c1
        c3 * c1,
        c1 * c1
    ]
    # Symmetric 13 A
    a_13_full = a_13 + a_13[-2::-1]
    
    b_12 = [
        c2 * c1,
        c2 * (1 - 2 * c1),
        c2 * c1
    ]
    b_12_full = b_12 + b_12[::-1]
    
    # We have 13 A and 12 B.
    # Merge A[0] and A[12] to get 12 A.
    a_12 = [0.0] * 12
    a_12[0] = a_13_full[0] + a_13_full[12]
    for i in range(1, 12):
        a_12[i] = a_13_full[i]
        
    b_12_final = b_12_full
    
    # Normalize
    sum_a = sum(a_12)
    sum_b = sum(b_12_final)
    
    if abs(sum_a) > 1e-10:
        a_12 = [x / sum_a for x in a_12]
    if abs(sum_b) > 1e-10:
        b_12_final = [x / sum_b for x in b_12_final]
        
    return a_12, b_12_final
