import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct the target multiplication tensor T
    # T[idx_U, idx_V, idx_W] = 1 if indices correspond to a term in the sum
    # idx_U = i*m + j
    # idx_V = j*p + k
    # idx_W = k*n + i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                T[idx_U, idx_V, idx_W] = 1.0

    start_time = time.time()
    deadline = 150  # seconds
    
    best_result = None
    min_rank_found = dim_U * dim_V * dim_W // max(dim_U, dim_V, dim_W) * 3 # heuristic start, actually schoolbook is n*m*p
    schoolbook_rank = n * m * p
    
    # Helper to convert factors to matrix
    def factors_to_dict(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Helper to compute error for given U, V, W
    # T_recon = sum(u_r outer v_r outer w_r)
    # This is computationally expensive to form fully each time for large R.
    # Instead, we can compute the error in unfolded form or use a fast check.
    # For small dims, we can just form the tensor.
    def compute_error(U, V, W):
        # U: (dim_U, R), V: (dim_V, R), W: (dim_W, R)
        # T_recon = einsum('ur,vr,wr->uvw', U, V, W)
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T - T_recon)
    
    def optimize_rank(R, num_restarts=20, max_iters=100):
        """
        Attempts to find a decomposition of rank R using ALS.
        """
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Heuristic lower bound check: Rank must be at least max(dim_U, dim_V, dim_W)?
        # Actually rank >= ceil((dim_U*dim_V*dim_W)^(1/3)) is weak.
        # For matrix mult, rank >= n*m*p / max(n,m,p) is a lower bound? No.
        # Let's just search R from schoolbook down.
        
        for restart in range(num_restarts):
            # Random initialization with small entries to avoid scaling issues
            U = np.random.randn(dim_U, R) * 0.1
            V = np.random.randn(dim_V, R) * 0.1
            W = np.random.randn(dim_W, R) * 0.1
            
            err = float('inf')
            # ALS Loop
            for it in range(max_iters):
                if time.time() > deadline - 5: break
                
                # Fix V, W. Find U.
                # T = sum u_r v_r^T w_r^T
                # Unfold along U: M_u = sum u_r (w_r^T v_r)
                # We want M_u to match T_u (matrix unfolding of T along U)
                # M_u = U diag(V^T W)
                # T_u = U (V^T W) ? No.
                # Let's use the linear system formulation for ALS.
                # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
                # For fixed V, W, this is linear in U.
                # U X = B where X_{jr, kr} = V_{jr} W_{kr}?
                # Let's reshape.
                # T_u (dim_U x dim_V*dim_W)
                # T_u[i, j*p + k] = T[i,j,k]
                # We want U M = T_u
                # M is (R x dim_V*dim_W) where M[r, idx_V, idx_W] = V[idx_V, r] * W[idx_W, r]
                # This M is sparse? No, dense.
                # dim_V*dim_W = 24*18 = 432. R ~ 70.
                # M size 70x432. T_u size 12x432.
                # Solving U (12x70) from U M = T_u.
                # Underdetermined if R > 12. Use least squares.
                
                # Construct M for U optimization
                # M is R x (dim_V * dim_W)
                # M[r, idx_V * dim_W + idx_W] = V[idx_V, r] * W[idx_W, r]
                # Vectorized: 
                # V_mat = V.T # R x dim_V
                # W_mat = W.T # R x dim_W
                # M = V_mat[:, :, None] * W_mat[:, None, :] -> R x dim_V x dim_W
                # M_flat = M.reshape(R, dim_V * dim_W)
                
                M_flat = (V.T[:, :, None] * W.T[:, None, :]).reshape(R, dim_V * dim_W)
                T_u = T.reshape(dim_U, dim_V * dim_W)
                
                # Solve U * M_flat = T_u
                # U = T_u * pinv(M_flat)
                try:
                    U_new = T_u @ np.linalg.pinv(M_flat, rcond=1e-10)
                except:
                    continue
                
                # Fix U, W. Find V.
                M_flat_V = (U.T[:, :, None] * W.T[:, None, :]).reshape(R, dim_U * dim_W)
                T_v = T.transpose(1, 2, 0).reshape(dim_V, dim_U * dim_W)
                try:
                    V_new = T_v @ np.linalg.pinv(M_flat_V, rcond=1e-10)
                except:
                    continue
                
                # Fix U, V. Find W.
                M_flat_W = (U.T[:, :, None] * V.T[:, None, :]).reshape(R, dim_U * dim_V)
                T_w = T.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
                try:
                    W_new = T_w @ np.linalg.pinv(M_flat_W, rcond=1e-10)
                except:
                    continue
                
                # Update and check error
                U, V, W = U_new, V_new, W_new
                err = compute_error(U, V, W)
                
                if err < 1e-9:
                    # Found a solution
                    # Snap to integers or half-integers if close?
                    # The task asks for integer/half-integer. 
                    # If error is very low, we can try to round.
                    return U, V, W, err
                
                if time.time() > deadline - 10:
                    break

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        return best_U, best_V, best_W, best_err

    # Search for lower rank
    # Schoolbook is n*m*p
    # Try ranks slightly below schoolbook
    # It is unlikely to work, but we try a few
    for R in [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 5]:
        if time.time() > deadline - 30:
            break
            
        U, V, W, err = optimize_rank(R, num_restarts=5, max_iters=20)
        
        if err < 1e-8:
            # Check if entries are close to integer/half-integer
            # If so, we have a valid result
            return factors_to_dict(U, V, W)
            
    # If no better rank found, return schoolbook
    # Construct schoolbook decomposition
    rank = schoolbook_rank
    U = [[0.0] * rank for _ in range(dim_U)]
    V = [[0.0] * rank for _ in range(dim_V)]
    W = [[0.0] * rank for _ in range(dim_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                U[idx_U][column] = 1.0
                V[idx_V][column] = 1.0
                W[idx_W][column] = 1.0
                column += 1
                
    return factors_to_dict(U, V, W)
# EVOLVE-BLOCK-END
