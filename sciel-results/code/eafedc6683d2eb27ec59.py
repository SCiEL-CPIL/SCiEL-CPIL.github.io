import numpy as np
from scipy.optimize import minimize
import time
import sys

def coefficients(stages):
    if stages != 12:
        # Fallback to uniform if stages != 12
        return [1.0/stages]*stages, [1.0/stages]*stages
    
    # Start timer
    start_time = time.time()
    deadline = 50.0 # Leave 10s margin for 60s limit
    
    # 1. Construct Hamiltonians A and B for 6-site chain
    # Sites 0..5
    # H = sum_{i=0}^{4} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + 0.8 Sz_i Sz_{i+1}) + 0.3 sum_i Sz_i
    
    # Spin operators
    Sx = np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res
    
    # Build A (Even bonds: 0-1, 2-3, 4-5) + half field
    # Build B (Odd bonds: 1-2, 3-4) + half field
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    lam = 0.8
    h = 0.3
    
    # Helper to add term to A or B
    def add_term(mat, ops, coeff=1.0):
        mat += coeff * kron_all(ops)
        
    # Even bonds: 0-1, 2-3, 4-5
    # Bond 0-1
    add_term(A, [Sx, Sx, Id, Id, Id, Id], 1.0)
    add_term(A, [Sy, Sy, Id, Id, Id, Id], 1.0)
    add_term(A, [Sz, Sz, Id, Id, Id, Id], lam)
    # Bond 2-3
    add_term(A, [Id, Id, Sx, Sx, Id, Id], 1.0)
    add_term(A, [Id, Id, Sy, Sy, Id, Id], 1.0)
    add_term(A, [Id, Id, Sz, Sz, Id, Id], lam)
    # Bond 4-5
    add_term(A, [Id, Id, Id, Id, Sx, Sx], 1.0)
    add_term(A, [Id, Id, Id, Id, Sy, Sy], 1.0)
    add_term(A, [Id, Id, Id, Id, Sz, Sz], lam)
    
    # Odd bonds: 1-2, 3-4
    # Bond 1-2
    add_term(B, [Id, Sx, Sx, Id, Id, Id], 1.0)
    add_term(B, [Id, Sy, Sy, Id, Id, Id], 1.0)
    add_term(B, [Id, Sz, Sz, Id, Id, Id], lam)
    # Bond 3-4
    add_term(B, [Id, Id, Id, Sx, Sx, Id], 1.0)
    add_term(B, [Id, Id, Id, Sy, Sy, Id], 1.0)
    add_term(B, [Id, Id, Id, Sz, Sz, Id], lam)
    
    # Field: h * sum Sz_i. Split half to A, half to B.
    for i in range(6):
        ops = [Id]*6
        ops[i] = Sz
        add_term(A, ops, h/2.0)
        add_term(B, ops, h/2.0)
        
    H = A + B
    
    # 2. Compute exact propagators
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        # Use scipy.linalg.expm for accuracy
        from scipy.linalg import expm
        U_ex = expm(-1j * t * H)
        U_exacts.append(U_ex)
        
    # 3. Define objective function
    # Symmetric parameters: a1..a6, b1..b6
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Constraints: sum(a)=1, sum(b)=1
    
    def objective(params):
        a_half = params[:6]
        b_half = params[6:12]
        
        # Enforce sum constraint by normalizing? 
        # Or use a penalty? 
        # Better to parameterize such that sum is 1.
        # Let a_i = x_i / sum(x). But optimization on simplex is tricky.
        # Let's just normalize inside.
        a_sum = np.sum(a_half)
        b_sum = np.sum(b_half)
        
        if np.isclose(a_sum, 0) or np.isclose(b_sum, 0):
            return 1e10
            
        a_full = np.array([a_half[0], a_half[1], a_half[2], a_half[3], a_half[4], a_half[5],
                           a_half[5], a_half[4], a_half[3], a_half[2], a_half[1], a_half[0]])
        b_full = np.array([b_half[0], b_half[1], b_half[2], b_half[3], b_half[4], b_half[5],
                           b_half[5], b_half[4], b_half[3], b_half[2], b_half[1], b_half[0]])
        
        # Normalize to sum 1
        a_full = a_full / (2.0 * a_sum)
        b_full = b_full / (2.0 * b_sum)
        
        # Compute U_approx for each t
        error = 0.0
        for t, U_ex in zip(t_values, U_exacts):
            U_app = np.eye(64, dtype=complex)
            for k in range(12):
                # U = ... exp(-i b_k t B) exp(-i a_k t A)
                # Note: order is k=1..12. 
                # U = M_12 ... M_1 where M_k = exp(-i b_k t B) exp(-i a_k t A)
                # So we multiply from left to right? 
                # U(t) = prod_{k=1}^{12} M_k. 
                # In matrix multiplication, U = M_12 @ ... @ M_1.
                # So we start with I, then U = M_1 @ U? No.
                # U = M_12 M_11 ... M_1.
                # If we iterate k=0 to 11 (indices), we want U = M_k @ U.
                # Let's verify: 
                # k=0: U = M_0
                # k=1: U = M_1 @ M_0. 
                # Yes.
                
                exp_A = expm(-1j * a_full[k] * t * A)
                exp_B = expm(-1j * b_full[k] * t * B)
                M_k = exp_B @ exp_A
                U_app = M_k @ U_app
                
            # Spectral norm error
            diff = U_app - U_ex
            # Spectral norm is largest singular value
            try:
                err = np.linalg.norm(diff, 2)
            except:
                err = 1e10
            error += err
            
        return error / len(t_values)
        
    # Initial guess: Suzuki 4th order 6-stage repeated
    z = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c1 = 1.0 / z
    c2 = 1.0 - 2.0 / z
    d1 = 1.0 - 1.0 / (2.0 * z)
    d2 = -1.0 / z
    d3 = 1.0 / (2.0 * z)
    
    a_6 = [c1, c2, c3 if (c3:=1.0/z) else 0, 1.0/z, c2, c1] # Wait, c3 is c1 in my previous var name? 
    # Suzuki 4th: c1 = 1/(2-2^(1/3)), c2 = -2^(1/3)/(2-2^(1/3))
    # My previous code: c1=1/z, c2=1-2/z. 
    # Let's stick to the standard Suzuki 4th order 6-stage coefficients for initialization.
    # a_6 = [c1, c2, c1, c1, c2, c1] ? No.
    # Standard 4th order 6-stage (symmetric):
    # a = [c1, c2, c3, c3, c2, c1]
    # b = [d1, d2, d3, d3, d2, d1]
    # where c1 = 1/z, c2 = -2/z + 1? No.
    # Let's use a simple uniform init if Suzuki is complex.
    a_init = np.ones(6) / 6.0
    b_init = np.ones(6) / 6.0
    
    x0 = np.concatenate([a_init, b_init])
    
    # Optimize
    best_params = x0
    best_score = objective(x0)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception as e:
        pass
        
    # Check time
    if time.time() - start_time > deadline:
        # Return best found
        pass
        
    a_half = best_params[:6]
    b_half = best_params[6:12]
    
    a_full = np.array([a_half[0], a_half[1], a_half[2], a_half[3], a_half[4], a_half[5],
                       a_half[5], a_half[4], a_half[3], a_half[2], a_half[1], a_half[0]])
    b_full = np.array([b_half[0], b_half[1], b_half[2], b_half[3], b_half[4], b_half[5],
                       b_half[5], b_half[4], b_half[3], b_half[2], b_half[1], b_half[0]])
    
    # Normalize
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    
    if a_sum != 0:
        a_full = a_full / a_sum
    if b_sum != 0:
        b_full = b_full / b_sum
        
    # Ensure finite and in range
    a_list = [float(x) for x in a_full]
    b_list = [float(x) for x in b_full]
    
    # Clip to [-50, 50] just in case, though they should be small
    a_list = [max(-50, min(50, x)) for x in a_list]
    b_list = [max(-50, min(50, x)) for x in b_list]
    
    return a_list, b_list

