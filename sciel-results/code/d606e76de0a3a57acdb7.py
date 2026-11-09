import numpy as np
import scipy.linalg
import scipy.optimize
import time

def _build_hamiltonians():
    """
    Builds the odd (A) and even (B) bond Hamiltonians for a 6-site open XXZ chain.
    Anisotropy = 0.8, Field = 0.3.
    A: Odd bonds (0,1), (2,3), (4,5) + half field.
    B: Even bonds (1,2), (3,4) + half field.
    """
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bond Hamiltonian: J * (sx⊗sx + sy⊗sy + anisotropy*sz⊗sz) / 2
    # Assuming J=1 for the bond strength, as is standard in these tasks unless specified.
    # The field is separate.
    
    # Add bonds to A (odd bonds: 0-1, 2-3, 4-5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    for i, j in odd_bonds:
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        A += 0.5 * kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        A += 0.5 * kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        A += 0.5 * anisotropy * kron_all(ops)
        
    # Add bonds to B (even bonds: 1-2, 3-4)
    even_bonds = [(1, 2), (3, 4)]
    for i, j in even_bonds:
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        B += 0.5 * kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        B += 0.5 * kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        B += 0.5 * anisotropy * kron_all(ops)
        
    # Add Field: H_field = field * sum(sz_i) / 2
    # Split half to A, half to B.
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        field_term = 0.5 * field * kron_all(ops)
        A += 0.5 * field_term
        B += 0.5 * field_term

    return A, B

def coefficients(stages):
    """
    Optimizes coefficients for the product formula.
    """
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # 2. Pre-compute exact propagators
    times = [0.25, 0.5, 1.0]
    H = A + B
    U_exact = [scipy.linalg.expm(-1j * t * H) for t in times]
    
    # 3. Define objective function
    # We use symmetry: a_k = a_{stages-k+1}, b_k = b_{stages-k+1}
    # For stages=12, we have 6 unique a's and 6 unique b's.
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # Sum(a) = 2*(x1+...+x6) = 1 => x6 = 0.5 - (x1+...+x5)
    # Similarly for b.
    
    def unpack(x):
        # x has 10 elements: a1..a5, b1..b5
        a_free = x[:5]
        b_free = x[5:]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, 
                      a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
        b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, 
                      b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
        
        # Check bounds [-50, 50]
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10, None, None
            
        return 0, a, b

    def objective(x):
        penalty, a, b = unpack(x)
        if penalty > 0:
            return penalty
            
        total_error = 0.0
        for t, U_ex in zip(times, U_exact):
            U_app = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B) exp(-i a_k t A)
                U_app = scipy.linalg.expm(-1j * b[k] * t * B) @ U_app
                U_app = scipy.linalg.expm(-1j * a[k] * t * A) @ U_app
            err = scipy.linalg.norm(U_app - U_ex, 2)
            total_error += err
            
        return total_error / 3.0

    # 4. Optimization
    # Initial guess: Strang splitting (2nd order)
    # Strang: a = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5] ? No.
    # Strang is usually 2 stages: exp(-i t A/2) exp(-i t B) exp(-i t A/2).
    # For 12 stages, we can just use uniform small steps or the seed's approach.
    # Seed: a = [1/12]*12? No, seed was substeps.
    # Let's start with a symmetric 4th order-like guess or just uniform.
    # Uniform: a_k = 1/12, b_k = 1/12.
    # x = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12]
    x0 = np.full(10, 1.0/12.0)
    
    # Check if x0 is valid
    p0, a0, b0 = unpack(x0)
    if p0 > 0:
        x0 = np.zeros(10) # Fallback to 0, then a6=0.5, b6=0.5
    
    best_x = x0
    best_val = objective(x0)
    
    # Use Powell method, robust for non-smooth
    # maxiter and maxfev to control time
    # Each eval is ~0.01-0.05s. 1000 evals = 10-50s.
    # We have ~140s. Let's allow 2000 evals.
    
    result = None
    try:
        result = scipy.optimize.minimize(
            objective, x0, method='Powell',
            options={'maxiter': 100, 'maxfev': 2000, 'disp': False}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # If time permits, do a second pass with tighter tolerance or different method
    if time.time() < deadline - 20.0:
        try:
            # Try Nelder-Mead with the current best as start
            result2 = scipy.optimize.minimize(
                objective, best_x, method='Nelder-Mead',
                options={'maxiter': 50, 'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass
            
    p_final, a_final, b_final = unpack(best_x)
    if p_final > 0:
        # Fallback to seed if optimization failed badly
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
        
    return list(a_final), list(b_final)
