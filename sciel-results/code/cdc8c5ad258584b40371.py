import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time

def coefficients(stages):
    """
    Optimizes the 12-stage product formula coefficients for the 6-site Heisenberg chain.
    Uses a symmetric parameterization and local optimization to minimize spectral norm error.
    """
    if stages != 12:
        # Fallback to Strang for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    t_start = time.time()
    deadline = t_start + 50.0  # Safety margin

    # 1. Build Hamiltonians
    # Single site operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    N = 6
    # Build bond operators
    # H_bond(j, j+1) = Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + 0.8 Sz_j Sz_{j+1}
    # Field: -0.3 Sz_j
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field terms: half to A, half to B
    field_coeff = -0.3 / 2.0
    for j in range(N):
        ops = [I] * N
        ops[j] = Sz
        Sz_j = kron_all(ops)
        A += field_coeff * Sz_j
        B += field_coeff * Sz_j
        
    # Bond terms
    # Odd bonds: 1-2 (j=0,1), 3-4 (j=2,3), 5-6 (j=4,5) -> Indices 0, 2, 4 for start site
    # Even bonds: 2-3 (j=1,2), 4-5 (j=3,4) -> Indices 1, 3 for start site
    # Note: Python 0-indexed.
    # Bond 0: site 0-1 (Odd)
    # Bond 1: site 1-2 (Even)
    # Bond 2: site 2-3 (Odd)
    # Bond 3: site 3-4 (Even)
    # Bond 4: site 4-5 (Odd)
    
    delta = 0.8
    
    def add_bond(matrix, j_start, j_end, coeff=1.0):
        # Sx_j Sx_k + Sy_j Sy_k + delta Sz_j Sz_k
        ops_x = [I] * N
        ops_x[j_start] = Sx
        ops_x[j_end] = Sx
        Sx_jk = kron_all(ops_x)
        
        ops_y = [I] * N
        ops_y[j_start] = Sy
        ops_y[j_end] = Sy
        Sy_jk = kron_all(ops_y)
        
        ops_z = [I] * N
        ops_z[j_start] = Sz
        ops_z[j_end] = Sz
        Sz_jk = kron_all(ops_z)
        
        matrix += coeff * (Sx_jk + Sy_jk + delta * Sz_jk)

    # A: Odd bonds (0-1, 2-3, 4-5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # B: Even bonds (1-2, 3-4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)

    # 2. Precompute exact propagators
    times = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in times:
        U_exacts.append(la.expm(-1j * t * (A + B)))

    # 3. Diagonalize A and B for fast exponentials
    # A = VA DA VA^dag
    wA, VA = la.eigh(A)
    wB, VB = la.eigh(B)
    
    # 4. Optimization
    # Variables: a1..a5, b1..b5
    # a6 = 0.5 - sum(a1..a5)
    # b6 = 0.5 - sum(b1..b5)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10 # Penalty
        
        a_vars = x[:5]
        b_vars = x[5:10]
        
        a6 = 0.5 - np.sum(a_vars)
        b6 = 0.5 - np.sum(b_vars)
        
        # Check bounds for all 12 coefficients
        a_full = np.array([a_vars[0], a_vars[1], a_vars[2], a_vars[3], a_vars[4], a6, a6, a_vars[4], a_vars[3], a_vars[2], a_vars[1], a_vars[0]])
        b_full = np.array([b_vars[0], b_vars[1], b_vars[2], b_vars[3], b_vars[4], b6, b6, b_vars[4], b_vars[3], b_vars[2], b_vars[1], b_vars[0]])
        
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
            
        error = 0.0
        for t, U_exact in zip(times, U_exacts):
            U_prod = np.eye(2**N, dtype=complex)
            
            for k in range(12):
                # Compute exp(-i b_k t B)
                phase_b = -1j * b_full[k] * t * wB
                exp_B = (VB * np.diag(np.exp(phase_b))) @ VA.conj().T # Wait, VB @ diag @ VB^dag
                # Correction: VB @ diag(exp) @ VB.conj().T
                exp_B = (VB * np.exp(phase_b)) @ VB.conj().T
                # Note: VB * np.exp(phase_b) scales columns of VB
                
                # Compute exp(-i a_k t A)
                phase_a = -1j * a_full[k] * t * wA
                exp_A = (VA * np.exp(phase_a)) @ VA.conj().T
                
                M_k = exp_B @ exp_A
                # U_prod = M_12 ... M_1
                # We are iterating k=0..11 which corresponds to 1..12
                # The product is M_12 M_11 ... M_1
                # So we should multiply U_prod = M_k @ U_prod?
                # If U_prod starts as I.
                # k=0 (M_1): U = M_1
                # k=1 (M_2): U = M_2 M_1
                # ...
                # k=11 (M_12): U = M_12 ... M_1
                # So U_prod = M_k @ U_prod is correct.
                U_prod = M_k @ U_prod
                
            diff = U_prod - U_exact
            # Spectral norm: max singular value
            # For 64x64, svd is fast enough
            error += np.linalg.norm(diff, 2)
            
        return error / 3.0

    # Initial guess: Uniform 1/12 for each? Or 4th order?
    # Let's try a mix of uniform and slight variation
    x0 = np.array([1/12.0]*5 + [1/12.0]*5)
    
    # Optimize
    result = opt.minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if time.time() > deadline or not result.success:
        # Fallback to best found or initial
        x_best = result.x if hasattr(result, 'x') else x0
    else:
        x_best = result.x
        
    a_vars = x_best[:5]
    b_vars = x_best[5:10]
    a6 = 0.5 - np.sum(a_vars)
    b6 = 0.5 - np.sum(b_vars)
    
    a = [a_vars[0], a_vars[1], a_vars[2], a_vars[3], a_vars[4], a6, a6, a_vars[4], a_vars[3], a_vars[2], a_vars[1], a_vars[0]]
    b = [b_vars[0], b_vars[1], b_vars[2], b_vars[3], b_vars[4], b6, b6, b_vars[4], b_vars[3], b_vars[2], b_vars[1], b_vars[0]]
    
    # Final verification
    if abs(sum(a) - 1.0) > 1e-9 or abs(sum(b) - 1.0) > 1e-9:
        # Normalize if drifted
        a = [x / sum(a) for x in a]
        b = [x / sum(b) for x in b]
        
    if any(x < -50 or x > 50 for x in a) or any(x < -50 or x > 50 for x in b):
        # Clip
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]
        # Renormalize
        a = [x / sum(a) for x in a]
        b = [x / sum(b) for x in b]

    return a, b
