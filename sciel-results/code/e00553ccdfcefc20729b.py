import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s kill

    def check_decomposition(U, V, W, rank, strict=True):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check half-integer constraint
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5): return False
        
        # Check correctness
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        Cf = np.zeros((n, p))
        # Vectorized check
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need to compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape for easier broadcasting
        # U_reshaped: (n, m, rank)
        U_r = U.reshape(n, m, rank)
        # V_reshaped: (m, p, rank)
        V_r = V.reshape(m, p, rank)
        # W_reshaped: (p, n, rank)
        W_r = W.reshape(p, n, rank)
        
        # Product term for fixed r: U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Sum over r.
        
        # Let's compute T[i, j, k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # This is a bit tricky with simple broadcasting.
        # Let's loop over r for the check since it's just 16-28 iterations.
        
        for r in range(rank):
            # u_r: (n, m)
            u_r = U_r[:, :, r]
            # v_r: (m, p)
            v_r = V_r[:, :, r]
            # w_r: (p, n)
            w_r = W_r[:, :, r]
            
            # term_r = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # Sum over j.
            # temp[j, k] = sum_i u_r[i,j] * v_r[j,k] ... wait.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Decomposition: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) W[k,i,r] ?? No.
            
            # The tensor definition is:
            # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # We want M_{i,j,k} = delta_{j,k}? No.
            # M_{i,j,k} is the tensor for matrix multiplication.
            # M_{i,j,k} = 1 if the indices correspond to a valid multiplication term?
            # Actually, the matrix multiplication tensor T has entries T_{i,j,k} such that
            # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}
            # So T_{i,j,k} is 1 if the indices are valid? No, T is the structure tensor.
            # T_{i,j,k} = 1 if we are summing A_{i,j} B_{j,k}.
            # The indices are i (row of A, row of C), j (col of A, row of B), k (col of B, col of C).
            # So for every i, j, k, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
            # Thus T_{i,j,k} = 1 for all valid i,j,k?
            # Wait, the decomposition is T = sum_r u^{(r)} v^{(r)} w^{(r)}.
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # We need T_{i,j,k} to be 1 for all i,j,k?
            # No. The matrix multiplication tensor has non-zero entries for every (i,j,k) triple
            # because every A_{i,j} and B_{j,k} pair contributes to some C_{i,k}.
            # Specifically, C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # So the tensor is T_{i,j,k} = 1 for all i,j,k.
            
            # Let's verify this interpretation.
            # If T_{i,j,k} = 1, then we need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
            
            # Let's check the Strassen code.
            # In the seed code:
            # U[g_idx, col] += coeff where g_idx = r*m + c (A index)
            # V[g_idx, col] += coeff where g_idx = g_row*p + g_col (B index)
            # W[g_idx, col] += coeff where g_idx = g_col*n + g_row (C index? No, W index is k*n+i)
            
            # Let's re-read the seed code verification.
            # acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            # This computes sum_cc U[ii,m+jj,cc] V[jj,p+kk,cc] W[kk,n+ii,cc]
            # This is the term for C[ii, kk].
            # And it sums over jj (the inner dimension).
            # So the full computation is:
            # Cf[ii, kk] = sum_jj ( sum_cc U[ii*m+jj, cc] V[jj*p+kk, cc] W[kk*n+ii, cc] )
            # This matches C[ii, kk] = sum_jj A[ii, jj] B[jj, kk]
            # IF the decomposition satisfies:
            # sum_cc U[ii*m+jj, cc] V[jj*p+kk, cc] W[kk*n+ii, cc] = 1 for all ii, jj, kk.
            
            # So yes, the target tensor T_{i,j,k} is 1 for all valid indices.
            
            # So for the check:
            # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
            
            # Let's compute this sum.
            # u_r: (n, m)
            # v_r: (m, p)
            # w_r: (p, n)
            
            # We want S[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
            
            # For a fixed r, term_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is an outer product of sorts.
            
            # Let's just loop i, j, k for the check since n,m,p are small.
            pass
        
        # Efficient check:
        # Construct the tensor T_reconstructed
        T_rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U.reshape(n, m, rank)[:, :, r] # (n, m)
            v_r = V.reshape(m, p, rank)[:, :, r] # (m, p)
            w_r = W.reshape(p, n, rank)[:, :, r] # (p, n)
            
            # We need to add u_r[i,j] * v_r[j,k] * w_r[k,i] to T_rec[i,j,k]
            # This is tricky to vectorize directly without large intermediate arrays.
            # But n,m,p are small (2,4,4).
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rec[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                        
        # T_target is 1 everywhere
        T_target = np.ones((n, m, p))
        
        if not np.allclose(T_rec, T_target, atol=1e-5):
            return False
        return True

    # Strategy: Try to find a rank 16 decomposition.
    # If not found, fallback to rank 28 Strassen.
    
    best_result = None
    
    # Precompute indices for vectorized objective if we were to use gradient descent
    # But for now, let's try a simple random search or L-BFGS-B for rank 16.
    
    # The problem is non-convex.
    # Let's try L-BFGS-B with multiple restarts for rank 16.
    
    def solve_rank(rank):
        # Parameters: U, V, W flattened
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w
        
        # Objective: minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        # This is a sum of squares of cubic terms.
        
        def objective(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u+size_v].reshape(m, p, rank)
            W_flat = params[size_u+size_v:].reshape(p, n, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over r
                        s = 0.0
                        for r in range(rank):
                            s += U_flat[i, j, r] * V_flat[j, k, r] * W_flat[k, i, r]
                        err += (s - 1.0) ** 2
            return err
        
        def gradient(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u+size_v].reshape(m, p, rank)
            W_flat = params[size_u+size_v:].reshape(p, n, rank)
            
            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_flat[i, j, r] * V_flat[j, k, r] * W_flat[k, i, r]
                        err_term = 2.0 * (s - 1.0)
                        
                        for r in range(rank):
                            # d/dU[i,j,r] = err_term * V[j,k,r] * W[k,i,r]
                            grad_U[i, j, r] += err_term * V_flat[j, k, r] * W_flat[k, i, r]
                            # d/dV[j,k,r] = err_term * U[i,j,r] * W[k,i,r]
                            grad_V[j, k, r] += err_term * U_flat[i, j, r] * W_flat[k, i, r]
                            # d/dW[k,i,r] = err_term * U[i,j,r] * V[j,k,r]
                            grad_W[k, i, r] += err_term * U_flat[i, j, r] * V_flat[j, k, r]
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        best_params = None
        best_fun = np.inf
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + restart)
            # Initialize with small random values
            params = np.random.randn(total_size) * 0.1
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_params = res.x
                    
                if best_fun < 1e-10:
                    break
                    
            except Exception as e:
                pass
        
        if best_params is not None and best_fun < 1e-8:
            U_opt = best_params[:size_u].reshape(n, m, rank)
            V_opt = best_params[size_u:size_u+size_v].reshape(m, p, rank)
            W_opt = best_params[size_u+size_v:].reshape(p, n, rank)
            
            # Round to half-integers
            U_int = np.round(2 * U_opt) / 2.0
            V_int = np.round(2 * V_opt) / 2.0
            W_int = np.round(2 * W_opt) / 2.0
            
            if check_decomposition(U_int, V_int, W_int, rank):
                return {"U": U_int.reshape(n*m, rank).tolist(), 
                        "V": V_int.reshape(m*p, rank).tolist(), 
                        "W": W_int.reshape(p*n, rank).tolist()}
            
            # If rounding failed, try to adjust
            # Sometimes the solution is close but not exactly on the grid.
            # We can try to snap to nearest half-integer and check.
            # Already done above.
            
        return None

    # Try rank 16
    result = solve_rank(16)
    if result:
        best_result = result
    else:
        # Try rank 18? 20?
        # The theoretical minimum is 16.
        # If 16 fails, try 17, 18... up to 27.
        for rank in range(17, 28):
            if time.time() > deadline:
                break
            result = solve_rank(rank)
            if result:
                best_result = result
                break

    if best_result:
        return best_result

    # Fallback to Strassen Rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print("C:")
        print(C)
        print("Cf:")
        print(Cf)
    else:
        print("Verification Passed")
        
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
