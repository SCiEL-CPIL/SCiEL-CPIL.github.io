"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)  
    # W shape: (p*n, R) = (12, R)
    
    def check_time():
        return (time.time() - _start_time) > TIME_LIMIT
    
    best_result = None
    best_rank = float('inf')
    
    def make_vec(size, idx_vals):
        vec = [0] * size
        for idx, val in idx_vals.items():
            vec[idx] = val
        return vec
    
    def build_from_terms(terms):
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c, (u,v,w) in enumerate(terms):
            for i in range(n*m):
                U[i][c] = u[i]
            for i in range(m*p):
                V[i][c] = v[i]
            for i in range(p*n):
                W[i][c] = w[i]
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construction 1: Trivial rank 36
    def trivial_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Construction 2: Strassen on 2x2 blocks + separate j=2
    # This gives rank 12 + 3*7 = 33
    def strassen_based_decomposition():
        terms = []
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                terms.append((make_vec(n*m, {u_idx: 1}),
                             make_vec(m*p, {v_idx: 1}),
                             make_vec(p*n, {w_idx: 1})))
        
        for j_block in range(3):
            b00 = 0 * p + 2*j_block
            b01 = 0 * p + 2*j_block + 1
            b10 = 1 * p + 2*j_block
            b11 = 1 * p + 2*j_block + 1
            
            c00 = (2*j_block) * n + 0
            c01 = (2*j_block + 1) * n + 0
            c10 = (2*j_block) * n + 1
            c11 = (2*j_block + 1) * n + 1
            
            a00 = 0 * m + 0
            a01 = 0 * m + 1
            a10 = 1 * m + 0
            a11 = 1 * m + 1
            
            terms.append((make_vec(6, {a00:1, a11:1}),
                         make_vec(18, {b00:1, b11:1}),
                         make_vec(12, {c00:1, c11:1})))
            terms.append((make_vec(6, {a10:1, a11:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c10:1, c11:-1})))
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b01:1, b11:-1}),
                         make_vec(12, {c01:1, c11:1})))
            terms.append((make_vec(6, {a11:1}),
                         make_vec(18, {b10:1, b00:-1}),
                         make_vec(12, {c00:1, c10:1})))
            terms.append((make_vec(6, {a00:1, a01:1}),
                         make_vec(18, {b11:1}),
                         make_vec(12, {c00:-1, c01:1})))
            terms.append((make_vec(6, {a10:1, a00:-1}),
                         make_vec(18, {b00:1, b01:1}),
                         make_vec(12, {c11:1})))
            terms.append((make_vec(6, {a01:1, a11:-1}),
                         make_vec(18, {b10:1, b11:1}),
                         make_vec(12, {c00:1})))
        
        return build_from_terms(terms)

    # Construction 3: Block structure - treat 3x6 as three 3x2 blocks
    # For each 3x2 block, we multiply 2x3 by 3x2 to get 2x2
    # The 2x2 matrix multiplication can be done with rank 7 (Strassen)
    # But we can also try to share structure across blocks
    # Let's try a different approach: use the fact that m=3 and split differently
    
    # Construction 4: Try to use rank-5 for 2x2x2 if possible (it's not, minimum is 7)
    # But maybe we can do better by exploiting the specific dimensions
    
    # Let's try a structured search for rank < 33
    def try_structured_search():
        """Try to find a lower rank decomposition using structured patterns"""
        import numpy as np
        
        # The tensor T has T[i,j,k] = 1 for all i,j,k
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's try to find a decomposition with rank R by solving a system
        # For small R, we can try to find integer/half-integer solutions
        
        # Strategy: Start from the Strassen decomposition and try to merge terms
        # or find a more efficient pattern
        
        # For now, let's just return the Strassen result as a baseline
        return None
    
    # Try constructions in order of increasing rank
    for name, decomp_func in [("strassen", strassen_based_decomposition),
                               ("trivial", trivial_decomposition)]:
        if check_time():
            break
        try:
            result = decomp_func()
            rank = len(result["U"][0])
            if verify_decomposition(result["U"], result["V"], result["W"]):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    # Try optimization-based search for lower rank
    if not check_time() and best_rank > 25:
        try:
            import numpy as np
            from scipy.optimize import minimize
            
            def get_strassen_init(R):
                """Create initial U,V,W matrices of rank R based on Strassen structure"""
                U = np.zeros((6, R))
                V = np.zeros((18, R))
                W = np.zeros((12, R))
                
                # First 12 columns: j=2 terms
                for i in range(n):
                    for k in range(p):
                        col = i * p + k
                        if col < R:
                            U[i*m+2, col] = 1
                            V[2*p+k, col] = 1
                            W[k*n+i, col] = 1
                
                # Next columns: Strassen terms for j=0,1 blocks
                strassen_patterns = [
                    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
                    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
                ]
                
                col = 12
                for j_block in range(3):
                    for (u_pat, v_pat, w_pat) in strassen_patterns:
                        if col >= R:
                            break
                        for (ii, jj), val in u_pat.items():
                            U[ii*m + jj, col] = val
                        for (jj, kk), val in v_pat.items():
                            V[jj*p + (2*j_block + kk), col] = val
                        for (kk, ii), val in w_pat.items():
                            W[(2*j_block + kk)*n + ii, col] = val
                        col += 1
                
                return U, V, W
            
            def compute_residual_squared(U, V, W):
                """Compute the squared residual using vectorized operations"""
                R = U.shape[1]
                # Residual[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
                residual_sq = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual_sq += (s - 1.0) ** 2
                return residual_sq
            
            def objective(x):
                """Objective function: sum of squared residuals"""
                R = len(x) // (6 + 18 + 12)
                U = x[:6*R].reshape(6, R)
                V = x[6*R:6*R+18*R].reshape(18, R)
                W = x[6*R+18*R:].reshape(12, R)
                return compute_residual_squared(U, V, W)
            
            def try_rounding(U_opt, V_opt, W_opt, precisions):
                """Try rounding to various precision levels and verify"""
                for prec in precisions:
                    scale = 2 ** prec
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                return None
            
            # Try ranks from current best-1 down to 20
            for target_rank in range(min(best_rank - 1, 32), 19, -1):
                if check_time():
                    break
                
                # Multiple random restarts
                for restart in range(5):
                    if check_time():
                        break
                    
                    np.random.seed(target_rank * 100 + restart)
                    
                    # Initialize based on Strassen structure
                    U0, V0, W0 = get_strassen_init(target_rank)
                    
                    # Add random perturbation
                    if target_rank > 27:
                        U0 += 0.05 * np.random.randn(6, target_rank)
                        V0 += 0.05 * np.random.randn(18, target_rank)
                        W0 += 0.05 * np.random.randn(12, target_rank)
                    
                    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                    
                    try:
                        result = minimize(objective, x0, method='L-BFGS-B', 
                                         options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10})
                        
                        if result.fun < 1e-10:
                            x_opt = result.x
                            U_opt = x_opt[:6*target_rank].reshape(6, target_rank)
                            V_opt = x_opt[6*target_rank:6*target_rank+18*target_rank].reshape(18, target_rank)
                            W_opt = x_opt[6*target_rank+18*target_rank:].reshape(12, target_rank)
                            
                            # Try rounding at multiple precision levels
                            found = try_rounding(U_opt, V_opt, W_opt, [0, 1, 2, 3])
                            if found is not None:
                                return found
                                
                    except Exception:
                        pass
        
        except Exception:
            pass
    
    if best_result is None:
        best_result = trivial_decomposition()
        best_rank = len(best_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
