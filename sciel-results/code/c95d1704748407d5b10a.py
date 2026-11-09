"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        n, m, p = DIMENSIONS
        np.random.seed(42)
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((p, n), dtype=float)
        for k in range(p):
            for i in range(n):
                val = 0.0
                for c in range(rank):
                    wc = W_arr[k*n+i, c]
                    if wc != 0:
                        for j in range(m):
                            val += wc * U_arr[i*m+j, c] * V_arr[j*p+k, c] * A[i,j] * B[j,k]
                C_computed[k, i] = val
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Schoolbook rank = n*m*p = 36
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Build the tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T has 1s at positions corresponding to (i,j,k)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a lower-rank decomposition using iterative optimization
    # We'll use a gradient-free approach: random initialization + local search
    
    def try_find_decomposition(target_rank, max_iter=5000, seed=0):
        """Try to find a rank-target_rank decomposition using iterative methods."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize randomly
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1
        
        # Objective: minimize ||T - sum_r U_r * V_r * W_r||^2
        # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
        # The tensor entry T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        def compute_residual(U_r, V_r, W_r, r):
            """Compute T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]"""
            # Efficient computation
            # For each r, outer product of U[:,r], V[:,r], W[:,r]
            residual = T.copy()
            for c in range(r):
                residual -= np.einsum('a,b,c->abc', U_r[:,c], V_r[:,c], W_r[:,c])
            return residual
        
        def objective(params):
            U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            res = compute_residual(U_r, V_r, W_r, target_rank)
            return np.sum(res**2)
        
        # Use Nelder-Mead or similar... but for 336+ dims this is too slow
        # Instead, use ALS (Alternating Least Squares) style
        
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        # Precompute some structures for speed
        # For each c (output index k*n+i), we need:
        # T[:, :, c] = sum_r U[:,r] * (V[:,r] * W[c,r])
        # For each r: U[:,r] should approximate (T[:, :, c] * something)
        
        # ALS approach:
        # Fix V, W: solve for U
        # Fix U, W: solve for V  
        # Fix U, V: solve for W
        
        def solve_U(V_fixed, W_fixed):
            """Solve for U given V and W."""
            U_new = np.zeros((n*m, target_rank))
            for a in range(n*m):
                for c in range(p*n):
                    # T[a, :, c] = sum_r U[a,r] * V[:,r] * W[c,r]
                    # This is: T[a, :, c] = sum_r U[a,r] * (V[:,r] * W[c,r])
                    # Let M[a, b, c] = T[a,b,c]
                    # For fixed a, c: T[a, :, c] = U[a, :] @ (V * W[c, :])^T
                    # where (V * W[c, :]) has entries V[b,r] * W[c,r]
                    
                    # Actually: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed a, c: T[a, :, c] = sum_r U[a,r] * V[:,r] * W[c,r]
                    # Let vwc = V * W[c, :][None, :]  # shape (m*p, target_rank)
                    # Then T[a, :, c] = vwc @ U[a, :].T
                    # This is a linear system in U[a, :]
                    
                    # We have: for all b, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # Let's form the matrix: A_mat[b, r] = V[b,r] * W[c,r]
                    # Then T[a, :, c] = A_mat @ U[a, :].T
                    # So U[a, :].T = A_mat^{-1} @ T[a, :, c]  (if invertible)
                    
                    A_mat = V_fixed * W_fixed[c, :][None, :]  # (m*p, target_rank)
                    target_vec = T[a, :, c]  # (m*p,)
                    
                    # Solve A_mat @ x = target_vec where x = U[a, :].T
                    try:
                        x = np.linalg.lstsq(A_mat, target_vec, rcond=None)[0]
                        U_new[a, :] = x
                    except:
                        U_new[a, :] = 0
            
            return U_new
        
        def solve_V(U_fixed, W_fixed):
            """Solve for V given U and W."""
            V_new = np.zeros((m*p, target_rank))
            for b in range(m*p):
                for c in range(p*n):
                    # T[:, b, c] = sum_r U[:,r] * V[b,r] * W[c,r]
                    # For fixed b, c: T[:, b, c] = V[b, :] * (U[:,r] * W[c,r]) summed over r
                    # Let A_mat[a, r] = U[a,r] * W[c,r]
                    # Then T[:, b, c] = A_mat.T @ V[b, :].T
                    # So V[b, :].T = A_mat^{-T} @ T[:, b, c]
                    
                    A_mat = U_fixed * W_fixed[c, :][None, :]  # (n*m, target_rank)
                    target_vec = T[:, b, c]  # (n*m,)
                    
                    try:
                        x = np.linalg.lstsq(A_mat.T, target_vec, rcond=None)[0]
                        V_new[b, :] = x
                    except:
                        V_new[b, :] = 0
            
            return V_new
        
        def solve_W(U_fixed, V_fixed):
            """Solve for W given U and V."""
            W_new = np.zeros((p*n, target_rank))
            for c in range(p*n):
                for a in range(n*m):
                    # T[a, :, c] = sum_r U[a,r] * V[:,r] * W[c,r]
                    # For fixed c, a: T[a, :, c] = W[c, :] * (U[a,r] * V[:,r]) summed over r
                    # Let A_mat[b, r] = U[a,r] * V[b,r]
                    # Then T[a, :, c] = A_mat.T @ W[c, :].T
                    # So W[c, :].T = A_mat^{-T} @ T[a, :, c]
                    
                    A_mat = U_fixed[a, :][None, :] * V_fixed  # (m*p, target_rank)
                    target_vec = T[a, :, c]  # (m*p,)
                    
                    try:
                        x = np.linalg.lstsq(A_mat.T, target_vec, rcond=None)[0]
                        W_new[c, :] = x
                    except:
                        W_new[c, :] = 0
            
            return W_new
        
        # Run ALS iterations
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            U_curr = solve_U(V_curr, W_curr)
            V_curr = solve_V(U_curr, W_curr)
            W_curr = solve_W(U_curr, V_curr)
            
            # Check convergence
            if it % 100 == 0:
                res = compute_residual(U_curr, V_curr, W_curr, target_rank)
                obj = np.sum(res**2)
                if obj < 1e-6:
                    # Found a good solution!
                    U_list = [[float(U_curr[i, j]) for j in range(target_rank)] for i in range(n*m)]
                    V_list = [[float(V_curr[i, j]) for j in range(target_rank)] for i in range(m*p)]
                    W_list = [[float(W_curr[i, j]) for j in range(target_rank)] for i in range(p*n)]
                    if verify(U_list, V_list, W_list, target_rank):
                        return (U_list, V_list, W_list)
        
        return None
    
    # Try various ranks from low to high
    for target_rank in range(20, min(36, rank_schoolbook)):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):  # Try a few random seeds
            if time.time() > deadline - 10:
                break
            
            result = try_find_decomposition(target_rank, max_iter=2000, seed=seed)
            if result is not None:
                U_found, V_found, W_found = result
                if verify(U_found, V_found, W_found, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_found, V_found, W_found
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook as fallback
    return {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
# EVOLVE-BLOCK-END
