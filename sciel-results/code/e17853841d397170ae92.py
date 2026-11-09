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
    
    # Construction 2: Correct Strassen + Tail decomposition (Rank 27)
    def strassen_based_decomposition():
        terms = []
        
        # Part 1: Tail (A[:, 2] * B[2, :])
        # A_R is 2x1 (col 2 of A), B_R is 1x6 (row 2 of B)
        # C_tail[i, k] = A[i, 2] * B[2, k]
        # Term for each k:
        # U[i, 2] = A[i, 2] -> U has A_R at m=2
        # V[2, k] = B[2, k] -> V has B_R[k] at m=2, col=k
        # W[k, i] = 1 -> W has 1 at row=k, col=i
        for k in range(p):
            u_vals = {0*m + 2: 1, 1*m + 2: 1} # A[0,2]=1, A[1,2]=1 for identity check? No, U holds A's values.
            # Wait, in tensor decomposition, U, V, W are fixed matrices.
            # The decomposition is sum_c U_c V_c W_c = T.
            # T[i, j, k] = delta_{j,0}delta_{k,0}... no, T[i,j,k] = 1 if j is the middle index?
            # T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
            # For the tail part (j=2):
            # We need sum_c U_c[i, 2] V_c[2, k] W_c[k, i] = A[i, 2] B[2, k].
            # This must hold for all i, k.
            # If we use 6 terms (one per k), we can set:
            # For term c_k (associated with output col k):
            # U_c[i, 2] = A[i, 2]
            # V_c[2, k] = B[2, k]
            # V_c[2, k'] = 0 for k' != k
            # W_c[k, i] = 1
            # W_c[k', i] = 0 for k' != k
            # Then sum_c U_c[i, 2] V_c[2, k] W_c[k, i] = U_c_k[i, 2] V_c_k[2, k] W_c_k[k, i] = A[i, 2] B[2, k] * 1.
            # This works.
            # However, U_c must be the same for all k if we want to share? No, each term has its own U, V, W.
            # So we create 6 terms.
            for i in range(n):
                # We need U_c[i, 2] to be A[i, 2].
                # Since A is variable, U_c[i, 2] must be the coefficient of A[i, 2].
                # In the decomposition, U_c[i, 2] is a constant (or half-integer) that multiplies A[i, 2].
                # So U_c[i, 2] = 1.
                pass
            
            u_vals = {0*m + 2: 1, 1*m + 2: 1}
            v_vals = {2*p + k: 1}
            w_vals = {k*n + 0: 1, k*n + 1: 1}
            
            terms.append((make_vec(n*m, u_vals),
                         make_vec(m*p, v_vals),
                         make_vec(p*n, w_vals)))
        
        # Part 2: Strassen for 3 blocks of 2x2 x 2x2
        # A_L = A[:, 0:2] (2x2)
        # B_L_j = B[0:2, 2j:2j+2] (2x2)
        # C_L_j = C[:, 2j:2j+2] (2x2)
        
        # Strassen products M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_patterns = [
            # (U_pat, V_pat, W_pat)
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for j_block in range(3):
            # Map local 2x2 indices to global indices
            # A: rows 0,1; cols 0,1 -> global (i, j)
            # B: rows 0,1; cols 2j, 2j+1 -> global (j, k)
            # C: rows 0,1; cols 2j, 2j+1 -> global (i, k)
            
            for (u_pat, v_pat, w_pat) in strassen_patterns:
                u_vals = {}
                v_vals = {}
                w_vals = {}
                
                for (ii, jj), val in u_pat.items():
                    # A index: row ii, col jj (jj is 0 or 1)
                    u_vals[ii*m + jj] = val
                
                for (jj, kk), val in v_pat.items():
                    # B index: row jj (0 or 1), col 2j_block + kk
                    v_vals[jj*p + (2*j_block + kk)] = val
                
                for (kk, ii), val in w_pat.items():
                    # C index: row ii (0 or 1), col 2j_block + kk
                    # Wait, W index is (k, i).
                    # In Strassen, C00 is (0,0), C01 is (0,1), etc.
                    # So kk is local col (0 or 1), ii is local row (0 or 1).
                    # Global col = 2*j_block + kk. Global row = ii.
                    # W index = global_col * n + global_row = (2*j_block + kk)*n + ii
                    w_vals[(2*j_block + kk)*n + ii] = val
                
                terms.append((make_vec(n*m, u_vals),
                             make_vec(m*p, v_vals),
                             make_vec(p*n, w_vals)))
        
        return build_from_terms(terms)
    
    # Construction 3: Optimization-based search for lower rank
    def try_optimization_search():
        """Use structured optimization to find lower-rank decompositions"""
        import numpy as np
        from scipy.optimize import minimize
        
        def get_strassen_init(R):
            """Create initial U,V,W matrices of rank R based on Strassen structure"""
            U = np.zeros((6, R))
            V = np.zeros((18, R))
            W = np.zeros((12, R))
            
            # Fill with the 27-rank decomposition
            # First 6 cols: Tail
            for k in range(p):
                col = k
                if col < R:
                    U[0*m+2, col] = 1
                    U[1*m+2, col] = 1
                    V[2*p+k, col] = 1
                    W[k*n+0, col] = 1
                    W[k*n+1, col] = 1
            
            # Next 21 cols: Strassen
            strassen_patterns = [
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
                ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
            ]
            
            col = 6
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
        
        def compute_residual(U, V, W):
            """Compute the residual tensor and return its norm"""
            R = U.shape[1]
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual[i,j,k] = s - 1
            return np.linalg.norm(residual)
        
        def objective(x):
            """Objective function for optimization"""
            R = len(x) // (6 + 18 + 12)
            U = x[:6*R].reshape(6, R)
            V = x[6*R:6*R+18*R].reshape(18, R)
            W = x[6*R+18*R:].reshape(12, R)
            return compute_residual(U, V, W)
        
        # Try ranks from current best-1 down to 20
        for target_rank in range(min(best_rank - 1, 32), 19, -1):
            if check_time():
                break
            
            # Multiple random restarts
            for restart in range(3):
                if check_time():
                    break
                
                np.random.seed(target_rank * 100 + restart)
                
                # Initialize based on Strassen structure
                U0, V0, W0 = get_strassen_init(target_rank)
                
                # Add small random perturbation
                if target_rank > 27:
                    U0 += 0.01 * np.random.randn(6, target_rank)
                    V0 += 0.01 * np.random.randn(18, target_rank)
                    W0 += 0.01 * np.random.randn(12, target_rank)
                
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                
                try:
                    result = minimize(objective, x0, method='L-BFGS-B', 
                                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                    
                    if result.fun < 1e-8:
                        # Check if we can round to exact values
                        x_opt = result.x
                        U_opt = x_opt[:6*target_rank].reshape(6, target_rank)
                        V_opt = x_opt[6*target_rank:6*target_rank+18*target_rank].reshape(18, target_rank)
                        W_opt = x_opt[6*target_rank+18*target_rank:].reshape(12, target_rank)
                        
                        # Try to find a clean solution by rounding
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        
                        if verify_decomposition(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        
                        # If rounding doesn't work, try integer rounding
                        U_rounded = np.round(U_opt)
                        V_rounded = np.round(V_opt)
                        W_rounded = np.round(W_opt)
                        
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        
                        if verify_decomposition(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                            
                except Exception:
                    pass
        
        return None
    
    best_rank = float('inf')
    best_result = None
    
    # Try constructions
    for name, decomp_func in [("trivial", trivial_decomposition), 
                               ("strassen", strassen_based_decomposition)]:
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
    
    # Try optimization-based search
    if not check_time() and best_rank > 28:
        try:
            result = try_optimization_search()
            if result is not None:
                rank = len(result["U"][0])
                if verify_decomposition(result["U"], result["V"], result["W"]):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
        except Exception:
            pass
    
    if best_result is None:
        best_result = trivial_decomposition()
        best_rank = len(best_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
