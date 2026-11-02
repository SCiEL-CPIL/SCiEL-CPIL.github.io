import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    def try_elimination():
        """
        Try to reduce rank by 1 by removing one term and solving for the rest
        with fixed supports.
        """
        if time.time() > deadline - 20:
            return None
            
        base = current_solution()
        R = 50
        U0 = np.array(base["U"], dtype=np.float64)
        V0 = np.array(base["V"], dtype=np.float64)
        W0 = np.array(base["W"], dtype=np.float64)
        
        # Build the linear system for the full tensor
        # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # We want to find U', V', W' for c in 0..R-2 such that sum = T
        # We fix the supports (indices of non-zeros) to be the same as in base,
        # but we allow the values to change.
        # Actually, if we remove term r, the supports of the other terms are fixed.
        # Let x_c be the value of the non-zero entries?
        # No, each term c has multiple non-zeros.
        # Let's parameterize each term c by its non-zero entries.
        # For term c, let S_c be the set of indices (i,j,k) where the rank-1 tensor is non-zero?
        # No, the rank-1 tensor is U_c \otimes V_c \otimes W_c.
        # The non-zeros are determined by the non-zero entries of U_c, V_c, W_c.
        # If we fix the *positions* of non-zeros in U_c, V_c, W_c, then the term is
        # determined by the values at those positions.
        
        # For each c, let N_U(c) be the set of rows i*m+j where U_c is non-zero.
        # Let N_V(c) be the set of rows j*p+k where V_c is non-zero.
        # Let N_W(c) be the set of rows k*n+i where W_c is non-zero.
        
        # The contribution of term c to T[i,j,k] is:
        # sum_{a in N_U(c), b in N_V(c), d in N_W(c)} 
        #   U_c[a] V_c[b] W_c[d] * delta(a, i*m+j) delta(b, j*p+k) delta(d, k*n+i)
        # This is non-zero only if i*m+j in N_U(c), j*p+k in N_V(c), k*n+i in N_W(c).
        # And the value is U_c[i*m+j] V_c[j*p+k] W_c[k*n+i].
        
        # So for each (i,j,k), the equation is:
        # sum_{c != r, (i,j,k) in Support(c)} U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = 1
        
        # This is non-linear because of the products U*V*W.
        # However, if we fix U and V, W is linear.
        # Or if we fix U and W, V is linear.
        
        # Strategy:
        # For each term r to remove:
        # 1. Fix U_c and V_c for c != r to their base values.
        # 2. Solve for W_c for c != r.
        # 3. Check if integer solution exists.
        
        # This might fail if the base values are not "generic" enough.
        # But it's a good first try.
        
        for r in range(R):
            if time.time() > deadline - 5:
                return None
                
            U_fix = U0.copy()
            V_fix = V0.copy()
            W_fix = W0.copy()
            
            # Zero out term r
            U_fix[:, r] = 0
            V_fix[:, r] = 0
            W_fix[:, r] = 0
            
            # We want to find W_new such that:
            # sum_{c != r} U_fix[i*m+j,c] V_fix[j*p+k,c] W_new[k*n+i,c] = 1
            
            # Let M[i,j,k, c] = U_fix[i*m+j,c] * V_fix[j*p+k,c]
            # Then sum_c M[i,j,k,c] W_new[k*n+i,c] = 1
            
            # This is a linear system for W_new.
            # But W_new is not a single vector, it's a matrix.
            # The equations are coupled through the indices of W.
            # W_new[k*n+i, c] appears in equation (i,j,k).
            
            # Let's collect all equations.
            # Number of equations: n*m*p = 54.
            # Number of unknowns: p_rows * (R-1) = 18 * 49 = 882.
            # This is underdetermined. We can set many W entries to 0.
            
            # Constraint: W_new[k*n+i, c] should be non-zero only if it was non-zero in W0?
            # If we enforce this sparsity, the system becomes sparse.
            
            # Let's try to solve for W_new assuming it has the same sparsity pattern as W0 (for c != r).
            # Unknowns: W_new[idx, c] where W0[idx, c] != 0.
            
            unknown_indices = []
            eq_indices = []
            
            for c in range(R):
                if c == r: continue
                for idx in range(p_rows):
                    if W0[idx, c] != 0:
                        unknown_indices.append((idx, c))
                        
            # Map (idx, c) to variable index
            var_map = {}
            for idx_var, (idx, c) in enumerate(unknown_indices):
                var_map[(idx, c)] = idx_var
                
            num_vars = len(unknown_indices)
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for T[i,j,k]
                        for c in range(R):
                            if c == r: continue
                            u_val = U_fix[i*m+j, c]
                            v_val = V_fix[j*p+k, c]
                            if u_val == 0 or v_val == 0: continue
                            
                            w_idx = k*n + i
                            # Check if W0[w_idx, c] is non-zero
                            if W0[w_idx, c] != 0:
                                var_idx = var_map.get((w_idx, c))
                                if var_idx is not None:
                                    A[eq_idx, var_idx] = u_val * v_val
                        eq_idx += 1
                        
            # Solve A x = b
            try:
                # Use lstsq for least squares, then round
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                x_int = np.round(x).astype(np.int64)
                
                # Verify residual
                if np.allclose(A @ x_int, b, atol=1e-10):
                    # Construct W_new
                    W_new = np.zeros((p_rows, R), dtype=np.int64)
                    for var_idx, (idx, c) in enumerate(unknown_indices):
                        W_new[idx, c] = x_int[var_idx]
                    
                    # Full verification
                    if verify_decomposition(U_fix.astype(np.int64).tolist(), V_fix.astype(np.int64).tolist(), W_new.tolist(), R-1):
                         # We found a solution of rank R-1?
                         # Wait, the rank is the number of non-zero columns.
                         # We zeroed column r, and others might have become zero?
                         # We should check the rank.
                         rank_found = 0
                         for c in range(R):
                             if np.any(W_new[:, c] != 0) or np.any(U_fix[:, c] != 0) or np.any(V_fix[:, c] != 0):
                                 rank_found += 1
                                 
                         if rank_found < R:
                             return {
                                 "U": U_fix.astype(np.int64).tolist(),
                                 "V": V_fix.astype(np.int64).tolist(),
                                 "W": W_new.tolist()
                             }
            except:
                pass
                
        return None

    def try_aggressive_search(r, num_trials=50):
        """
        More aggressive random search with better structure.
        """
        if time.time() > deadline - 10:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 3:
                break
            
            # W structure: each column has exactly 2 non-zeros, one for i=0, one for i=1
            W = np.zeros((p_rows, r), dtype=np.int64)
            for c in range(r):
                k0 = random.randint(0, p-1)
                k1 = random.randint(0, p-1)
                a = random.choice([-1, 1, 2, -2, 3, -3])
                b = random.choice([-1, 1, 2, -2, 3, -3])
                W[k0*n + 0, c] = a
                W[k1*n + 1, c] = b
            
            # V structure: each column has 1 non-zero
            V = np.zeros((m_rows, r), dtype=np.int64)
            for c in range(r):
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                V[j*p+k, c] = random.choice([-1, 1, 2, -2])
            
            # Solve for U
            U = np.zeros((n_rows, r), dtype=np.int64)
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    try:
                        u_float, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u_float).astype(np.int64)
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None

    # Try to find lower rank solutions
    best_result = None
    
    # Try elimination first (fast, structured)
    if time.time() < deadline - 30:
        res = try_elimination()
        if res:
            best_result = res
    
    # Try aggressive search for ranks 49 down to 45
    if best_result is None:
        for r in range(49, 44, -1):
            if time.time() > deadline - 10:
                break
            res = try_aggressive_search(r, num_trials=50)
            if res:
                best_result = res
                break
    
    # Fallback to rank 50
    if best_result is None:
        best_result = current_solution()
    
    return best_result
