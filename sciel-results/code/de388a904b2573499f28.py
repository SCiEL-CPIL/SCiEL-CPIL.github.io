import time
import numpy as np
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Attempts to find a rank-minimal decomposition using Alternating Least Squares (ALS)
    on the all-ones tensor, falling back to schoolbook (rank 45) if a clean
    integer/half-integer solution is not found within the time limit.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Shape: (n, m, p)
    T = np.ones((n, m, p))
    
    # Function to check if a matrix has integer or half-integer entries
    def is_clean(mat):
        # Check if 2*mat is close to integer
        double_mat = 2.0 * mat
        rounded = np.round(double_mat)
        if np.allclose(double_mat, rounded, atol=1e-5):
            # Verify exactness of the decomposition with rounded values
            U_r = rounded[0] / 2.0
            V_r = rounded[1] / 2.0
            W_r = rounded[2] / 2.0
            
            # Reconstruct tensor
            # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Using np.einsum for efficiency
            T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_recon, T, atol=1e-5):
                return True
        return False

    def run_als(rank, max_iters=100, tol=1e-9, seed=0):
        np.random.seed(seed)
        # Initialize U, V, W
        # Scale initialization to avoid vanishing/exploding gradients
        U = np.random.randn(n, rank) * 0.1
        V = np.random.randn(m, rank) * 0.1
        W = np.random.randn(p, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # T_reshaped (n, m*p)
            # WtV (m*p, rank) -> W.T @ V? No.
            # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # Fix V, W.
            # T_reshaped[i, j*p + k] = sum_r U[i,r] * (V[j,r] * W[k,r])
            # Let M[j, k, r] = V[j,r] * W[k,r]
            # M_reshaped (m*p, rank)
            M = np.einsum('jr,kr->jkr', V, W).reshape(m * p, rank)
            T_reshaped = T.reshape(n, m * p)
            # Solve U @ M.T = T_reshaped
            # U = T_reshaped @ M.T @ inv(M @ M.T)
            try:
                U = np.linalg.solve(M.T @ M, M.T @ T_reshaped.T).T
            except np.linalg.LinAlgError:
                return None

            # Update V
            # T_reshaped[j, i*n + k] = sum_r V[j,r] * (U[i,r] * W[k,r])
            M = np.einsum('ir,kr->ikr', U, W).reshape(n * p, rank)
            T_reshaped = T.transpose(1, 0, 2).reshape(m, n * p)
            try:
                V = np.linalg.solve(M.T @ M, M.T @ T_reshaped.T).T
            except np.linalg.LinAlgError:
                return None

            # Update W
            # T_reshaped[k, i*n + j] = sum_r W[k,r] * (U[i,r] * V[j,r])
            M = np.einsum('ir,jr->ijr', U, V).reshape(n * m, rank)
            T_reshaped = T.transpose(2, 0, 1).reshape(p, n * m)
            try:
                W = np.linalg.solve(M.T @ M, M.T @ T_reshaped.T).T
            except np.linalg.LinAlgError:
                return None
            
            # Check convergence
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_recon - T)
            if error < tol:
                return (U, V, W)
                
        # Final check
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(T_recon - T)
        if error < 1e-5:
            return (U, V, W)
        return None

    # Try ranks from 35 to 44
    # Schoolbook is 45.
    # We know rank 45 works.
    # We try to find lower.
    for rank in range(35, 45):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            res = run_als(rank, max_iters=50, seed=seed)
            if res is not None:
                U, V, W = res
                if is_clean((U, V, W)):
                    # Convert to lists of lists as required
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a low error solution but not clean, we might still try to clean it
        # by checking if scaling helps. But is_clean already checks for half-integers.
        # If no clean solution found for this rank, try next.

    # Fallback: Schoolbook Rank 45
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col_idx] = 1
                V[row_v][col_idx] = 1
                W[row_w][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
