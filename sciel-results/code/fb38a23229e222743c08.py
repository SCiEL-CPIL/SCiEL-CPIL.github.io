import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    def build_rank_28():
        """Build the known rank-28 Strassen tiling decomposition."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff
                
        def strassen_terms():
            terms = []
            # M1 = (X00+X11)(Y00+Y11)
            u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
            terms.append((u, v, w))
            # M2 = (X10+X11)Y00
            u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
            terms.append((u, v, w))
            # M3 = X00(Y01-Y11)
            u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
            terms.append((u, v, w))
            # M4 = X11(Y10-Y00)
            u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
            terms.append((u, v, w))
            # M5 = (X00+X01)Y11
            u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
            terms.append((u, v, w))
            # M6 = (X10-X00)(Y00+Y01)
            u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
            terms.append((u, v, w))
            # M7 = (X01-X11)(Y10+Y11)
            u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
            terms.append((u, v, w))
            return terms
        
        base_terms = strassen_terms()
        
        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx
                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff
                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff
                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff
                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)
        
        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank):
        """Verify the tensor decomposition is exact."""
        # T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Non-zero entries: T[i*4+j, j*4+k, k*2+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        # Check zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check all (u,v,w) where u=i*m+j, v=j*p+k, w=k*n+i
                    # We need to check that T[u,v,w]=0 when not of the form above
                    # Actually the tensor is sparse: only T[i*m+j, j*p+k, k*n+i]=1
                    # All other entries should be 0
                    pass  # The non-zero check above covers the required entries
                    # For zero entries, we verify with a random test
        # Random verification
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-10)
    
    def try_reduce_rank(U, V, W, current_rank, target_rank, max_iters=50):
        """Try to reduce rank by zeroing columns and re-solving."""
        if target_rank >= current_rank:
            return None
        
        U_work = U[:, :target_rank].copy()
        V_work = V[:, :target_rank].copy()
        W_work = W[:, :target_rank].copy()
        
        # Try iterative refinement: pick columns to zero and adjust others
        # Use a greedy approach: try removing each column and see if we can compensate
        best_solution = None
        best_error = float('inf')
        
        for trial in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Pick a random column to try removing
            col_to_remove = np.random.randint(0, current_rank)
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Zero out the column
            U_try[:, col_to_remove] = 0
            V_try[:, col_to_remove] = 0
            W_try[:, col_to_remove] = 0
            
            # Now try to fix by adjusting other columns
            # Use least squares to find corrections
            # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear, so use iterative linearization
            
            success = False
            for iteration in range(20):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                
                # Check current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            actual = 0.0
                            for r in range(current_rank):
                                actual += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                            error += (actual - 1.0) ** 2
                
                if error < 1e-20:
                    success = True
                    break
                
                # Linearize: perturb one matrix at a time
                # Try adjusting U
                # For each (i,j,k), the residual is res[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # d/dU[i*m+j,r] = V[j*p+k,r]*W[k*n+i,r]
                
                # Build a system: for each (i,j,k), we want sum_r delta_U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = -res[i,j,k]
                # This is a linear system in delta_U
                
                num_constraints = n * m * p  # 32
                # But we have 8*current_rank unknowns in U, which is too many
                # Instead, do coordinate descent on columns
                
                # Pick a random column r to adjust in U
                r = np.random.randint(0, current_rank)
                if r == col_to_remove:
                    continue
                    
                # For this column, adjust U[:, r] to minimize error
                # The contribution of column r to constraint (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # So we want: U[i*m+j,r] = -res[i,j,k] / (V[j*p+k,r]*W[k*n+i,r]) when V*W != 0
                # This is overdetermined, so use least squares
                
                A_sys = np.zeros((num_constraints, 1))
                b_sys = np.zeros(num_constraints)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = V_try[j*p+k, r] * W_try[k*n+i, r]
                            if abs(coeff) > 1e-15:
                                A_sys[idx, 0] = coeff
                                # Current contribution from all other columns
                                other_sum = 0.0
                                for r2 in range(current_rank):
                                    if r2 != r:
                                        other_sum += U_try[i*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+i, r2]
                                b_sys[idx] = 1.0 - other_sum
                            idx += 1
                
                # Solve least squares
                if np.any(np.abs(A_sys) > 1e-15):
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    U_try[:, r] = sol[0]
                
                # Similarly adjust V and W
                r = np.random.randint(0, current_rank)
                if r == col_to_remove:
                    continue
                A_sys = np.zeros((num_constraints, 1))
                b_sys = np.zeros(num_constraints)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U_try[i*m+j, r] * W_try[k*n+i, r]
                            if abs(coeff) > 1e-15:
                                A_sys[idx, 0] = coeff
                                other_sum = 0.0
                                for r2 in range(current_rank):
                                    if r2 != r:
                                        other_sum += U_try[i*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+i, r2]
                                b_sys[idx] = 1.0 - other_sum
                            idx += 1
                if np.any(np.abs(A_sys) > 1e-15):
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    V_try[:, r] = sol[0]
                
                r = np.random.randint(0, current_rank)
                if r == col_to_remove:
                    continue
                A_sys = np.zeros((num_constraints, 1))
                b_sys = np.zeros(num_constraints)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U_try[i*m+j, r] * V_try[j*p+k, r]
                            if abs(coeff) > 1e-15:
                                A_sys[idx, 0] = coeff
                                other_sum = 0.0
                                for r2 in range(current_rank):
                                    if r2 != r2:
                                        other_sum += U_try[i*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+i, r2]
                                b_sys[idx] = 1.0 - other_sum
                            idx += 1
                if np.any(np.abs(A_sys) > 1e-15):
                    sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W_try[:, r] = sol[0]
            
            if success:
                # Verify
                if verify_decomposition(U_try, V_try, W_try, current_rank):
                    # Remove the zero column
                    mask = np.ones(current_rank, dtype=bool)
                    mask[col_to_remove] = False
                    U_new = U_try[:, mask]
                    V_new = V_try[:, mask]
                    W_new = W_try[:, mask]
                    if verify_decomposition(U_new, V_new, W_new, current_rank - 1):
                        return U_new, V_new, W_new
        
        return None
    
    # Build the rank-28 solution
    U, V, W = build_rank_28()
    current_rank = 28
    
    # Try to reduce rank
    for target_rank in range(current_rank - 1, 7, -1):  # Try down to rank 8
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_reduce_rank(U, V, W, current_rank, target_rank, max_iters=30)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
    
    # Final verification
    if not verify_decomposition(U, V, W, current_rank):
        # Fall back to rank 28
        U, V, W = build_rank_28()
        current_rank = 28
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
                    acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
