import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Target Ranks to try in order: 16, 17, 18
    # If 16 or 17 found, return it.
    # If not, return a valid 18.
    
    # Pre-compute target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries 0
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T
    
    T = get_target_tensor()
    
    def verify(U, V, W):
        # U, V, W are lists of lists or numpy arrays
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # This is the tensor contraction
        # Result shape (n*m, m*p, p*n)
        # Efficient way: 
        # Reshape U to (n*m, R), V to (m*p, R), W to (p*n, R)
        # T_comp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        R = U_np.shape[1]
        T_comp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        return np.allclose(T_comp, T, atol=1e-5)
    
    def try_find_rank(rank):
        """
        Attempt to find a decomposition of given rank using ALS.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None
            
        # Initialize
        # Use random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid divergence
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0: U[:, r] /= norm_u
            if norm_v > 0: V[:, r] /= norm_v
            if norm_w > 0: W[:, r] /= norm_w
        
        best_error = float('inf')
        best_U, best_V, best_W = U, V, W
        
        # ALS iterations
        max_iter = 2000
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            # Minimize sum_{i,j,k} (T_{ijk} - sum_r U_{ir} V_{jr} W_{kr})^2 wrt U
            # For each r, U[:, r] is updated.
            # Actually, update all rows of U for a fixed r? No, ALS updates one mode at a time.
            # Update U:
            # For each r, we can update U[:, r] by fixing V, W.
            # But U is a matrix. We update column by column?
            # Standard ALS:
            # For r in 0..R-1:
            #   Compute residual T_res = T - sum_{s != r} U_s V_s W_s
            #   Solve for U_r: minimize || T_res - U_r (V_r outer W_r) ||_F
            #   This is a linear least squares problem.
            
            # To speed up, we can update U in one shot using SVD?
            # No, standard ALS is faster for small ranks.
            
            # Let's do simple gradient descent on flattened U, V, W?
            # No, ALS is better.
            
            # Optimization: 
            # We can update U, V, W sequentially.
            
            # Update U
            U_new = np.zeros_like(U)
            for r in range(rank):
                # Compute sum of other terms
                T_others = np.zeros_like(T)
                for s in range(rank):
                    if s == r: continue
                    T_others += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
                
                T_target = T - T_others
                # We want U_r such that sum_i U_{ir} V_{jr} W_{kr} approx T_target_{ijk}
                # This is a linear system for U_{ir}.
                # Let B_r = V[:, r] outer W[:, r]. Shape (m*p, p*n).
                # Then sum_j,k U_{jr} B_{jk} ... wait.
                # T_{ijk} = sum_i' U_{i' r} (V_{j r} W_{k r}) ... No.
                # T_{ijk} = sum_r U_{i r} V_{j r} W_{k r}.
                # For fixed r, T_{ijk} \approx U_{i r} V_{j r} W_{k r}.
                # This is not a simple linear solve for U_r unless we fix V_r, W_r.
                # If we fix V_r, W_r, then for each (j,k), we have T_{ijk} / (V_{jr} W_{kr})? No.
                
                # The update rule for U in ALS:
                # U_new = (T contracted with V, W) (V^T W ... )^{-1} ?
                # Actually, if we treat U as a matrix, the update is:
                # U_new[:, r] is not independent.
                
                # Let's use a simpler approach: 
                # Minimize error by adjusting U, V, W with small steps (SGD)?
                # Too slow.
                
                # Let's stick to the standard ALS update for rank-1 components.
                # This assumes we are optimizing one term at a time while keeping others fixed.
                # This is valid if we iterate enough.
                
                # Solve for U[:, r]:
                # Let A_r = V[:, r] * W[:, r] (outer product). Shape (m*p, p*n).
                # We want min_U sum_{j,k} (T_{i j k} - sum_{s!=r} ... - U_{i r} A_{j k r})^2
                # This is a linear least squares problem for U[:, r].
                # Let M be the matrix where M_{i, (j,k)} = A_{j k r}.
                # Then U[:, r] = (M^T M)^{-1} M^T b, where b is the residual vectorized.
                
                A_r = np.outer(V[:, r], W[:, r])
                M = A_r.reshape(m*p * p*n, 1) # This is wrong. M should map U to the tensor slice.
                # T_{ijk} depends on U_{ir}.
                # For fixed r, the term is U_{ir} * (V_{jr} W_{kr}).
                # So for each i, we have a vector of coefficients (V_{jr} W_{kr}) for all j,k.
                # We want to find U_{ir} to match the residual.
                # This is a separate least squares for each i? No, U is a column vector.
                # The error is sum_{i,j,k} (Res_{ijk} - U_{ir} C_{jkr})^2.
                # This is a linear least squares for the vector U[:, r].
                # Matrix C has rows indexed by (j,k) and columns by i? No.
                # The term is U_{ir} C_{jkr}.
                # This is not separable in that way.
                
                # Correct ALS Update for U:
                # Fix V, W.
                # Minimize || T - sum_r U_r \otimes V_r \otimes W_r ||_F^2 wrt U.
                # This is equivalent to minimizing for each i? No.
                # Let's contract T with V and W.
                # T_U = np.einsum('ijk,jr,kr->ir', T, V, W)
                # V_W = np.einsum('jr,kr->rkj', V, W) ... no.
                # The Hessian is related to (V^T V) \otimes (W^T W).
                
                # Let's use the formula:
                # U_new = T_contracted * (Gram_inv)
                # Where T_contracted has shape (n*m, rank)
                # T_contracted[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r]
                
                T_con = np.einsum('ijk,jr,kr->ir', T, V, W)
                
                # Gram matrix G: G[r, s] = sum_{j,k} V[j,r] V[j,s] W[k,r] W[k,s]
                # G = (V^T V) * (W^T W) element-wise? No.
                # G[r,s] = sum_j V[j,r]V[j,s] * sum_k W[k,r]W[k,s]
                # Let Vg = V.T @ V (rank x rank)
                # Let Wg = W.T @ W (rank x rank)
                # Then G = Vg * Wg (Hadamard product)
                
                Vg = V.T @ V
                Wg = W.T @ W
                G = Vg * Wg
                
                # Solve G U_new = T_con
                # Add regularization
                G_reg = G + 1e-6 * np.eye(rank)
                U_new = np.linalg.solve(G_reg, T_con)
                
                # Update V
                T_con_v = np.einsum('ijk,ir,kr->jr', T, U_new, W)
                Ug = U_new.T @ U_new
                G_v = Ug * Wg
                G_v_reg = G_v + 1e-6 * np.eye(rank)
                V_new = np.linalg.solve(G_v_reg, T_con_v)
                
                # Update W
                T_con_w = np.einsum('ijk,ir,jr->kr', T, U_new, V_new)
                Ug_w = U_new.T @ U_new
                Vg_w = V_new.T @ V_new
                G_w = Ug_w * Vg_w
                G_w_reg = G_w + 1e-6 * np.eye(rank)
                W_new = np.linalg.solve(G_w_reg, T_con_w)
                
                U, V, W = U_new, V_new, W_new
                
            # Check convergence
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_comp - T)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-5:
                break
        
        # Check if best_U, best_V, best_W can be rationalized
        if best_error < 1e-4:
            # Try to snap to integers/half-integers
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if verify(U_r, V_r, W_r):
                return [U_r.tolist(), V_r.tolist(), W_r.tolist()]
            # Try integers only
            U_i = np.round(best_U)
            V_i = np.round(best_V)
            W_i = np.round(best_W)
            if verify(U_i, V_i, W_i):
                return [U_i.tolist(), V_i.tolist(), W_i.tolist()]
                
        return None
        
    # Try Rank 16
    res = try_find_rank(16)
    if res:
        return res
        
    # Try Rank 17
    res = try_find_rank(17)
    if res:
        return res
        
    # Try Rank 18
    res = try_find_rank(18)
    if res:
        return res
        
    # Fallback: Deterministic Rank 20 (or 19/18 if I can construct it)
    # Construct Rank 19 (Strassen + Schoolbook)
    # Or just return Rank 20 to be safe?
    # The island best is 18. I must return at least 18.
    # If try_find_rank(18) failed, it means it didn't find a *rational* solution.
    # I should construct a known valid rank 18 or 19.
    
    # Let's construct Rank 19 deterministically.
    # Strassen for cols 0,1 (7 mults)
    # Schoolbook for cols 2,3,4 (12 mults)
    # Total 19.
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    # Strassen for A * B[:, 0:2]
    # B indices:
    # B00 -> V[0], B01 -> V[1]
    # B10 -> V[2], B11 -> V[3]
    # Wait, V index is j*p + k.
    # j=0, k=0 -> V[0] (B00)
    # j=0, k=1 -> V[1] (B01)
    # j=1, k=0 -> V[5] (B10) -- p=5
    # j=1, k=1 -> V[6] (B11)
    
    # W indices:
    # C00 -> W[0] (k=0, i=0)
    # C01 -> W[1] (k=1, i=0)
    # C10 -> W[2] (k=0, i=1)
    # C11 -> W[3] (k=1, i=1)
    
    # Strassen Terms
    # M1 = a(B00+B11)
    # U[0, 0] = 1 (a)
    # V[0, 0] = 1 (B00), V[6, 0] = 1 (B11)
    # W[0, 0] = 1 (C00), W[3, 0] = 1 (C11)
    # Note: W contribution is added to C.
    # C00 += M1, C11 += M1
    
    U[0][0] += 1
    V[0][0] += 1
    V[6][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2 = (a+b)B10
    # U[0, 1] = 1, U[1, 1] = 1
    # V[5, 1] = 1 (B10)
    # W[2, 1] = 1 (C10), W[3, 1] = 1 (C11)
    # C10 += M2, C11 += M2 (with -1 sign? Strassen: C11 = M1 - M2 + M3 + M6)
    
    U[0][1] += 1
    U[1][1] += 1
    V[5][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3 = (c+d)B01
    # U[2, 2] = 1, U[3, 2] = 1
    # V[1, 2] = 1 (B01)
    # W[1, 2] = 1 (C01), W[3, 2] = 1 (C11)
    
    U[2][2] += 1
    U[3][2] += 1
    V[1][2] += 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4 = d(B00-B11)
    # U[3, 3] = 1
    # V[0, 3] = 1, V[6, 3] = -1
    # W[0, 3] = 1 (C00), W[2, 3] = 1 (C10)
    
    U[3][3] += 1
    V[0][3] += 1
    V[6][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5 = (a+d)(B01+B10)
    # U[0, 4] = 1, U[3, 4] = 1
    # V[1, 4] = 1, V[5, 4] = 1
    # W[0, 4] = -1 (C00), W[1, 4] = 1 (C01)
    
    U[0][4] += 1
    U[3][4] += 1
    V[1][4] += 1
    V[5][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6 = (b-d)(B11+B10)
    # U[1, 5] = 1, U[3, 5] = -1
    # V[6, 5] = 1, V[5, 5] = 1
    # W[3, 5] = 1 (C11)
    
    U[1][5] += 1
    U[3][5] -= 1
    V[6][5] += 1
    V[5][5] += 1
    W[3][5] += 1
    
    # M7 = (a-c)(B00+B01)
    # U[0, 6] = 1, U[2, 6] = -1
    # V[0, 6] = 1, V[1, 6] = 1
    # W[0, 6] = 1 (C00)
    
    U[0][6] += 1
    U[2][6] -= 1
    V[0][6] += 1
    V[1][6] += 1
    W[0][6] += 1
    
    # Now Schoolbook for Cols 2, 3, 4
    # Col 2: B02 (V[2]), B12 (V[7])
    # C02 (W[4]), C12 (W[5])
    # M8 = a B02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # M9 = b B12
    U[1][8] += 1
    V[7][8] += 1
    W[4][8] += 1
    
    # M10 = c B02
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # M11 = d B12
    U[3][10] += 1
    V[7][10] += 1
    W[5][10] += 1
    
    # Col 3: B03 (V[3]), B13 (V[8])
    # C03 (W[6]), C13 (W[7])
    # M12 = a B03
    U[0][11] += 1
    V[3][11] += 1
    W[6][11] += 1
    
    # M13 = b B13
    U[1][12] += 1
    V[8][12] += 1
    W[6][12] += 1
    
    # M14 = c B03
    U[2][13] += 1
    V[3][13] += 1
    W[7][13] += 1
    
    # M15 = d B13
    U[3][14] += 1
    V[8][14] += 1
    W[7][14] += 1
    
    # Col 4: B04 (V[4]), B14 (V[9])
    # C04 (W[8]), C14 (W[9])
    # M16 = a B04
    U[0][15] += 1
    V[4][15] += 1
    W[8][15] += 1
    
    # M17 = b B14
    U[1][16] += 1
    V[9][16] += 1
    W[8][16] += 1
    
    # M18 = c B04
    U[2][17] += 1
    V[4][17] += 1
    W[9][17] += 1
    
    # M19 = d B14
    U[3][18] += 1
    V[9][18] += 1
    W[9][18] += 1
    
    return {"U": U, "V": V, "W": W}

