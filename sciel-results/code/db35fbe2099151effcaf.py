import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0
        
    def check_exact(U, V, W, T, tol=1e-10):
        """Check if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T exactly (within tol)."""
        recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(recon, T, atol=tol)

    def find_decomp(n, m, p, target_rank, max_iters=500, seed=42):
        """
        Find a rank-r decomposition for n x m x p matrix multiplication.
        Returns (U, V, W) if found, else None.
        """
        # Build tensor T
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Lower bound on rank
        min_rank = max(n * m, m * p, p * n)
        if target_rank < min_rank:
            return None

        np.random.seed(seed)
        
        # Initialize with random sparse matrices to encourage low rank structure
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Randomly place 1s and -1s
        for c in range(target_rank):
            num_ones_U = np.random.randint(1, n + 1)
            idx_U = np.random.choice(n * m, num_ones_U, replace=False)
            U[idx_U, c] = np.random.choice([-1, 1], num_ones_U)
            
            num_ones_V = np.random.randint(1, m + 1)
            idx_V = np.random.choice(m * p, num_ones_V, replace=False)
            V[idx_V, c] = np.random.choice([-1, 1], num_ones_V)
            
            num_ones_W = np.random.randint(1, p + 1)
            idx_W = np.random.choice(p * n, num_ones_W, replace=False)
            W[idx_W, c] = np.random.choice([-1, 1], num_ones_W)

        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            VW = np.einsum('bd,cd->bcd', V, W)
            VW_2d = VW.transpose(2, 0, 1).reshape(target_rank, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            try:
                # Use lstsq for speed and stability
                U_new, _, _, _ = np.linalg.lstsq(VW_2d.T, T_2d.T, rcond=None)
                U = U_new.T
            except:
                U = U_new if 'U_new' in locals() else U
                
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)
            UW_2d = UW.transpose(2, 0, 1).reshape(target_rank, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_2d.T, T_2d_v.T, rcond=None)
                V = V_new.T
            except:
                V = V_new if 'V_new' in locals() else V

            # Update W
            UV = np.einsum('ad,bd->abd', U, V)
            UV_2d = UV.transpose(2, 0, 1).reshape(target_rank, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_2d.T, T_2d_w.T, rcond=None)
                W = W_new.T
            except:
                W = W_new if 'W_new' in locals() else W
            
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                break
        
        # Try rounding to half-integers
        U_h = to_half_int(best_U)
        V_h = to_half_int(best_V)
        W_h = to_half_int(best_W)
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
        
        # Try current U, V, W
        U_h2 = to_half_int(U)
        V_h2 = to_half_int(V)
        W_h2 = to_half_int(W)
        if check_exact(U_h2, V_h2, W_h2, T):
            return U_h2, V_h2, W_h2

        return None

    # Strategy 1: Exploit structure by partitioning m=4 into 2+2
    # Sub-problem: 3 x 2 x 5
    # If we find rank R_sub for 3x2x5, we get rank 2*R_sub for 3x4x5
    # Schoolbook for 3x2x5 is 30.
    # Let's try to find rank < 30 for 3x2x5.
    
    n_sub, m_sub, p_sub = 3, 2, 5
    
    # Try ranks for sub-problem
    # Lower bound for 3x2x5 is max(6, 10, 15) = 15.
    # Let's try 15, 16, ..., 20.
    found_sub = None
    for r_sub in range(15, 21):
        if time.time() > deadline - 20:
            break
        # Try multiple seeds for the sub-problem
        for seed in range(50):
            if time.time() > deadline - 20:
                break
            res = find_decomp(n_sub, m_sub, p_sub, r_sub, max_iters=200, seed=seed + r_sub * 100)
            if res is not None:
                found_sub = (r_sub, res)
                break
        if found_sub is not None:
            break

    if found_sub is not None:
        r_sub, (U_sub, V_sub, W_sub) = found_sub
        # Lift to 3x4x5
        rank_total = 2 * r_sub
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        # Fill first half (corresponding to A1 B1, j in 0..1)
        for i in range(n):
            for j in range(2):
                U[i * m + j, :r_sub] = U_sub[i * m_sub + j, :]
        
        for j in range(2):
            for k in range(p):
                V[j * p + k, :r_sub] = V_sub[j * p_sub + k, :]
        
        for k in range(p):
            for i in range(n):
                W[k * n + i, :r_sub] = W_sub[k * p_sub + i, :]
        
        # Fill second half (corresponding to A2 B2, j in 2..3)
        for i in range(n):
            for j in range(2, 4):
                U[i * m + j, r_sub:] = U_sub[i * m_sub + (j - 2), :]
        
        for j in range(2, 4):
            for k in range(p):
                V[j * p + k, r_sub:] = V_sub[(j - 2) * p_sub + k, :]
        
        for k in range(p):
            for i in range(n):
                W[k * n + i, r_sub:] = W_sub[k * p_sub + i, :]
                
        # Verify
        T_full = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i * m + j, j * p + k, k * n + i] = 1.0
                    
        if check_exact(U, V, W, T_full):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Direct search for 3x4x5
    # Schoolbook is 60.
    # Try ranks from 60 downwards? Or just return schoolbook if time is tight.
    # Given the time spent on sub-problems, we might not have much time left.
    # Let's just return schoolbook if we didn't find a better sub-problem solution.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
