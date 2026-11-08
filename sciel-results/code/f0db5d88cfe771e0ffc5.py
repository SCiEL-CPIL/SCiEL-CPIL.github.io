import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.dot(U[ui, :] * V[vj, :], W[wk, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    # Rank-7 Strassen decomposition for 2x2x2
    def strassen_terms_2x2():
        terms = []
        # M1
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        
        # M3
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        
        # M4
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        
        # M5
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        
        # M6
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        
        # M7
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms_2x2()
    
    def apply_strassen_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col, U, V, W):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_col_off * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off * 2 + r
                g_c = b_col_off * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    # Build the rank-28 solution
    rank_28 = 28
    U_28 = np.zeros((nm, rank_28), dtype=float)
    V_28 = np.zeros((mp, rank_28), dtype=float)
    W_28 = np.zeros((pn, rank_28), dtype=float)
    
    apply_strassen_block(0, 0, 0, 0, 0, U_28, V_28, W_28)
    apply_strassen_block(1, 1, 0, 0, 7, U_28, V_28, W_28)
    apply_strassen_block(0, 0, 1, 1, 14, U_28, V_28, W_28)
    apply_strassen_block(1, 1, 1, 1, 21, U_28, V_28, W_28)
    
    # Verify the rank-28 solution
    if not verify_decomposition(U_28, V_28, W_28):
        return {"U": to_list(U_28), "V": to_list(V_28), "W": to_list(W_28)}
    
    best_rank = rank_28
    best_U = U_28.copy()
    best_V = V_28.copy()
    best_W = W_28.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Attempt to find rank 22 solution
    # Strategy: Take the rank 28 solution, remove 6 columns, and solve for the remaining 22.
    # This is a non-linear least squares problem.
    
    target_rank = 22
    
    # Define the residual function for least_squares
    def residuals(params, U_fixed, V_fixed, W_fixed, cols_to_keep):
        # params: flattened array of the 22 columns of U, V, W
        # We need to map params back to U, V, W
        # U shape: (8, 22), V shape: (16, 22), W shape: (8, 22)
        # Total params: 8*22 + 16*22 + 8*22 = 22 * 32 = 704
        
        size_U = nm * target_rank
        size_V = mp * target_rank
        size_W = pn * target_rank
        
        U_params = params[:size_U].reshape(nm, target_rank)
        V_params = params[size_U:size_U+size_V].reshape(mp, target_rank)
        W_params = params[size_U+size_V:].reshape(pn, target_rank)
        
        # We are optimizing the 22 columns that we keep.
        # The other 6 columns are zeroed out (removed).
        # We check the tensor condition:
        # sum_r U[i,r] V[j,r] W[k,r] = 1 if i,j,k correspond to a 1 in the tensor, else 0.
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.dot(U_params[ui, :] * V_params[vj, :], W_params[wk, :])
                    # The target is 1.0 for all valid (i,j,k)
                    res.append(val - 1.0)
                    
        return np.array(res)
    
    # We need to choose which 22 columns to keep from the 28.
    # A good strategy is to keep the first 22 columns and optimize them.
    # Or, we can try to merge some columns.
    # Let's try to optimize the first 22 columns of the rank-28 solution.
    
    cols_to_keep = list(range(22))
    
    U_init = U_28[:, cols_to_keep]
    V_init = V_28[:, cols_to_keep]
    W_init = W_28[:, cols_to_keep]
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # We can also try other subsets of columns if the first one fails.
    # But let's start with this.
    
    try:
        # Use least_squares with 'lm' (Levenberg-Marquardt) or 'trf' (Trust Region Reflective)
        # 'lm' is good for small problems, but this is 704 variables, 32 residuals.
        # It's underdetermined. 'lm' requires M >= N. Here M=32, N=704.
        # So we must use 'trf' or 'dogbox'.
        
        # However, since it's underdetermined, we might need to regularize or fix some variables.
        # Let's try 'trf' with bounds to keep values reasonable.
        
        # Actually, a better approach for underdetermined systems is to use SVD or a specific solver.
        # But scipy's least_squares with 'trf' can handle it if we provide bounds.
        
        # Let's try to solve it.
        # We might need to fix some variables to break symmetry/gauge freedom.
        # For now, let's just try to solve it directly.
        
        # To improve convergence, we can use a smaller number of iterations or a specific method.
        # Given the time limit, we'll do a few trials with different initializations.
        
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + trial)
            
            # Perturb the initial guess slightly
            noise_scale = 0.1
            U_init_pert = U_init + np.random.randn(*U_init.shape) * noise_scale
            V_init_pert = V_init + np.random.randn(*V_init.shape) * noise_scale
            W_init_pert = W_init + np.random.randn(*W_init.shape) * noise_scale
            
            x0_pert = np.concatenate([U_init_pert.flatten(), V_init_pert.flatten(), W_init_pert.flatten()])
            
            # Define bounds
            lower = -10.0
            upper = 10.0
            
            result = least_squares(
                residuals, 
                x0_pert, 
                args=(U_28, V_28, W_28, cols_to_keep), 
                method='trf', 
                bounds=(lower, upper),
                max_nfev=5000
            )
            
            if result.success or result.cost < 1e-8:
                # Extract the solution
                U_sol = result.x[:size_U].reshape(nm, target_rank)
                V_sol = result.x[size_U:size_U+size_V].reshape(mp, target_rank)
                W_sol = result.x[size_U+size_V:].reshape(pn, target_rank)
                
                # Snap to half-integers if close
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if verify_decomposition(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break
                else:
                    # Check if the non-snapped solution is good enough
                    if verify_decomposition(U_sol, V_sol, W_sol, tol=1e-6):
                        best_rank = target_rank
                        best_U = U_sol
                        best_V = V_sol
                        best_W = W_sol
                        break
                        
    except Exception as e:
        pass

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
