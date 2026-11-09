import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    """
    Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Anisotropy gamma = 0.8, Field h = 0.3
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    # Initialize A and B as zero matrices
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    iden = np.eye(2, dtype=complex)
    
    # Helper to apply operator at site i
    def apply_op(op, site):
        ops = [iden] * N
        ops[site] = op
        result = ops[0]
        for op_next in ops[1:]:
            result = np.kron(result, op_next)
        return result
    
    # XXZ interaction: 0.5 * (Sx Sx + Sy Sy + gamma Sz Sz)
    # S = 0.5 * sigma
    # Term = 0.5 * (0.25 sx sx + 0.25 sy sy + 0.25 gamma sz sz) = 1/8 (sx sx + sy sy + gamma sz sz)
    # Actually standard Heisenberg is S_i . S_j.
    # Sx Sx + Sy Sy = 0.5 (S+ S- + S- S+)
    
    for site in range(N):
        # Field term: h * Sz
        # Split field: half to A, half to B
        field_term = h * apply_op(0.5 * sz, site)
        A += field_term
        B += field_term
        
        # Bonds
        if site < N - 1:
            # Interaction between site and site+1
            sx_sx = apply_op(sx, site) @ apply_op(sx, site + 1)
            sy_sy = apply_op(sy, site) @ apply_op(sy, site + 1)
            sz_sz = apply_op(sz, site) @ apply_op(sz, site + 1)
            
            interaction = 0.25 * (sx_sx + sy_sy + gamma * sz_sz)
            
            if site % 2 == 0:
                # Odd bond (0-1, 2-3, 4-5) -> A
                A += interaction
            else:
                # Even bond (1-2, 3-4) -> B
                B += interaction
                
    return A, B

def _compute_error(a, b, A, B, exact_propagators, times):
    """
    Compute the mean spectral norm error.
    a, b: lists of 12 coefficients.
    """
    a = np.array(a)
    b = np.array(b)
    error_sum = 0.0
    
    for t, U_exact in zip(times, exact_propagators):
        U = np.eye(2 ** 6, dtype=complex)
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Note: The task says prod_{k=1..12} exp(-i b_k t B) exp(-i a_k t A)
        # This means for each k, we multiply by exp(-i b_k t B) then exp(-i a_k t A)
        
        # Optimization: Pre-compute exp(-i a_k t A) and exp(-i b_k t B) is expensive if done inside loop.
        # But A and B are constant. We can diagonalize them once? 
        # A and B are Hermitian. 
        # For 64x64 matrices, expm is fast enough for 12 stages * 3 times * ~1000 iterations?
        # 12 * 2 * 3 * 1000 = 72000 expm calls. 64x64 expm is ~1ms? 72s. Too slow.
        
        # We need a faster way.
        # We can pre-compute the eigendecomposition of A and B.
        # A = V_A D_A V_A^H
        # exp(-i a_k t A) = V_A exp(-i a_k t D_A) V_A^H
        # This reduces expm to exp on diagonal.
        
        pass

# We need to optimize the evaluation speed.
# Let's pre-compute eigendecompositions.

_A_eig = None
_B_eig = None

def _init_eigs(A, B):
    global _A_eig, _B_eig
    _A_eig = np.linalg.eigh(A)
    _B_eig = np.linalg.eigh(B)

def _compute_error_fast(a, b, times):
    a = np.array(a)
    b = np.array(b)
    error_sum = 0.0
    
    # Pre-computed exact propagators are passed via closure or global? 
    # Better to pass them.
    # Let's assume _exact_propagators and _times are global or passed to coefficients.
    
    for t, U_exact in zip(_times, _exact_propagators):
        U = np.eye(64, dtype=complex)
        
        # Get eigendata
        V_A, D_A = _A_eig
        V_B, D_B = _B_eig
        
        # Construct exponentials
        # exp(-i a_k t A)
        # We can compute this on the fly.
        
        for k in range(12):
            # B part
            if b[k] != 0:
                exp_B = (V_B * np.exp(-1j * b[k] * t * D_B)) @ V_B.conj().T
                U = exp_B @ U
            # A part
            if a[k] != 0:
                exp_A = (V_A * np.exp(-1j * a[k] * t * D_A)) @ V_A.conj().T
                U = exp_A @ U
                
        # Spectral norm of U - U_exact
        diff = U - U_exact
        # Spectral norm is max singular value
        # For 64x64, svd is fast.
        try:
            s = np.linalg.svd(diff, compute_uv=False)
            err = s[0]
        except:
            err = 1e10
            
        error_sum += err
        
    return error_sum / len(times)

# Global variables for optimization
_times = [0.25, 0.5, 1.0]
_exact_propagators = []
_A_mat = None
_B_mat = None

def coefficients(stages):
    global _A_mat, _B_mat, _exact_propagators, _A_eig, _B_eig
    
    if _A_mat is None:
        _A_mat, _B_mat = _build_hamiltonians()
        _init_eigs(_A_mat, _B_mat)
        
        # Compute exact propagators
        for t in _times:
            H = _A_mat + _B_mat
            U_exact = np.linalg.expm(-1j * t * H)
            _exact_propagators.append(U_exact)
            
    # Initial guess: 4th order symmetric coefficients repeated
    alpha = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    beta = 1.0 - 2.0 * alpha
    
    # 4-stage 4th order block:
    # A coeffs: alpha/2, beta/2, beta/2, alpha/2
    # B coeffs: alpha, beta/2, alpha (interleaved)
    # We have 12 stages.
    # Let's use 3 blocks of 4 stages?
    # Block:
    # Stage 1: b=alpha, a=alpha/2
    # Stage 2: b=beta/2, a=beta/2
    # Stage 3: b=alpha, a=beta/2
    # Stage 4: b=0, a=alpha/2
    # Sum a: alpha/2 + beta/2 + beta/2 + alpha/2 = alpha + beta = 1
    # Sum b: alpha + beta/2 + alpha = 2alpha + beta/2 = 2(1.35) - 0.85 = 1.85? 
    # Wait, sum of b must be 1.
    # In the 4th order scheme, the total B time is 1.
    # My mapping above:
    # e^{t alpha/2 A} e^{t alpha B} e^{t beta/2 A} e^{t beta/2 B} e^{t beta/2 A} e^{t alpha B} e^{t alpha/2 A}
    # B terms: alpha, beta/2, alpha. Sum = 2alpha + beta/2.
    # alpha = 1/(2-2^{1/3}) ~ 1.3512
    # beta = 1 - 2alpha ~ -1.7024
    # Sum B = 2(1.3512) - 1.7024/2 = 2.7024 - 0.8512 = 1.8512.
    # This is not 1.
    # The standard 4th order symmetric scheme for e^{t(A+B)} is:
    # e^{t c_1 A} e^{t c_2 B} e^{t c_1 A} e^{t c_3 B} e^{t c_1 A} e^{t c_2 B} e^{t c_1 A}
    # where c_1 = alpha/2, c_2 = alpha, c_3 = beta.
    # Sum A = 4 c_1 = 2 alpha.
    # Sum B = 2 c_2 + c_3 = 2 alpha + beta.
    # We require Sum A = 1, Sum B = 1.
    # 2 alpha = 1 => alpha = 0.5.
    # But alpha is fixed by the order condition.
    # The scheme is for e^{t(A+B)}. The coefficients sum to 1.
    # c_1 = 1/(4-2^{4/3})? No.
    # Let's use the known 4th order coefficients:
    # c_1 = 1/(4 - 2 * 2^{1/3}) ?
    # Actually, let's just use Nelder-Mead starting from Strang (which is valid and sums to 1) 
    # and let it find the optimum.
    
    # Initial guess: Strang
    a_init = [1.0/12.0] * 12
    b_init = [1.0/12.0] * 12
    
    # Or better initial guess: 4th order scaled
    # Let's try to fix the 4th order coefficients to sum to 1.
    # The 4th order symmetric 4-stage (in terms of alternating A,B) has:
    # A: c1, c2, c2, c1
    # B: c1, c3, c1
    # Sum A = 2c1 + 2c2 = 1
    # Sum B = 2c1 + c3 = 1
    # Order 4 conditions:
    # c1 + c2 = 1/2
    # c1^2 + c1 c2 + c2^2 = 1/3 ?
    
    # Let's stick to a simple robust optimization.
    # Start with Strang.
    
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s for safety
    
    # Use Nelder-Mead
    # We need to handle the constraint sum(a)=1, sum(b)=1.
    # We can optimize a[0..10] and b[0..10] and set the last ones to enforce sum=1.
    # a[11] = 1 - sum(a[0:11])
    # b[11] = 1 - sum(b[0:11])
    
    def objective(x):
        a = list(x[:11])
        b = list(x[11:22])
        a.append(1.0 - sum(a))
        b.append(1.0 - sum(b))
        
        # Check bounds [-50, 50]
        if any(abs(v) > 50 for v in a + b):
            return 1e10
            
        return _compute_error_fast(a, b, _times)
        
    x0 = [1.0/12.0] * 22
    
    best_x = x0
    best_val = objective(x0)
    
    # Nelder-Mead
    options = {'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        
        # Check if it improved
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Polish with tighter tolerance if time remains
        if time.time() < deadline - 5.0:
            options2 = {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            res2 = minimize(objective, best_x, method='Nelder-Mead', options=options2)
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception as e:
        pass
        
    a_final = list(best_x[:11])
    b_final = list(best_x[11:22])
    a_final.append(1.0 - sum(a_final))
    b_final.append(1.0 - sum(b_final))
    
    # Ensure finite and in bounds
    for i in range(12):
        if not np.isfinite(a_final[i]) or abs(a_final[i]) > 50:
            a_final[i] = 1.0/12.0
        if not np.isfinite(b_final[i]) or abs(b_final[i]) > 50:
            b_final[i] = 1.0/12.0
            
    return a_final, b_final
