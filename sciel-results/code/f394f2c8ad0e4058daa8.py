import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target: rank 24 via two blocks of rank 12 for 2x3x4
    
    # 1. Find rank-12 decomposition for 2x3x4
    # Tensor T_sub[i, j, k] = 1 for all i in 0..1, j in 0..2, k in 0..3
    # We want sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1
    # Note: W index in sub-problem is (k, i) which maps to k*n_sub + i? 
    # In the full problem, W is indexed by k_full * n + i.
    # For the sub-problem, let's just use W_sub[k, i, r].
    
    n_sub = 2
    m_sub = 3
    p_sub = 4
    rank_sub = 12
    
    # Initialize U, V, W
    # U shape: (n_sub * m_sub, rank_sub) -> (6, 12)
    # V shape: (m_sub * p_sub, rank_sub) -> (12, 12)
    # W shape: (p_sub * n_sub, rank_sub) -> (8, 12)
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 60  # 60 seconds for search
    
    # We will try a few random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(n_sub * m_sub, rank_sub)
        V = np.random.randn(m_sub * p_sub, rank_sub)
        W = np.random.randn(p_sub * n_sub, rank_sub)
        
        # ALS iterations
        for it in range(200):
            # Update U
            # For each i, j, the equation is sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            # This is a linear system for U[:, r] given V, W?
            # Actually, for fixed V, W, the term for U[i*m+j, r] is coupled across r?
            # No, the sum is over r.
            # Let's rewrite:
            # sum_r U[i*m+j, r] * Y[i*m+j, r] = 1, where Y[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # This is not a simple linear system for U because Y depends on r.
            # Wait, the standard ALS for tensor decomposition minimizes ||T - sum u v w||^2.
            # Here T is a 3-way tensor.
            # T[i, j, k] = 1.
            # We want to approximate T with a sum of rank-1 tensors.
            # The error is sum_{i,j,k} (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2.
            
            # Let's use the standard ALS update rules for tensor decomposition.
            # Unfold the tensor into matrices.
            
            # Update U:
            # For each r, U[:, r] is a vector.
            # The contribution of r to the tensor is u_r v_r w_r.
            # We want to minimize ||T - sum u v w||^2.
            # This is equivalent to minimizing ||T_1 - sum u_r (v_r w_r)^T||_F^2 for U.
            # Where T_1 is the unfolding along mode 1.
            # T_1[i*m+j, k*n+i']? No, the indices are mixed.
            
            # Let's stick to the direct equation:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a system of 24 equations (2*3*4) for 12*6 + 12*12 + 12*8 = 72+144+96 = 216 variables.
            # It's underdetermined.
            
            # Let's use a gradient descent approach on the variables.
            # Or simpler: Fix V and W, solve for U.
            # For each (i, j), the equation is:
            # sum_k sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # This is sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r]) = 1
            # Let Z[i*m+j, r] = sum_k V[j*p+k, r] W[k*n+i, r]
            # Then sum_r U[i*m+j, r] Z[i*m+j, r] = 1
            # This is a linear system for U[i*m+j, :] ? No, it's a dot product.
            # We have 1 equation for 12 unknowns U[i*m+j, 0..11].
            # We can minimize ||U[i*m+j, :]||^2 subject to U[i*m+j, :] . Z[i*m+j, :] = 1.
            # Solution: U[i*m+j, :] = Z[i*m+j, :] / (Z[i*m+j, :] . Z[i*m+j, :])
            
            # Similarly for V and W.
            
            # Let's implement this cyclic minimization.
            
            for _ in range(100):
                # Update U
                for i in range(n_sub):
                    for j in range(m_sub):
                        idx_u = i * m_sub + j
                        z = np.zeros(rank_sub)
                        for k in range(p_sub):
                            idx_v = j * p_sub + k
                            idx_w = k * n_sub + i
                            z += V[idx_v, :] * W[idx_w, :]
                        norm_sq = np.dot(z, z)
                        if norm_sq > 1e-10:
                            U[idx_u, :] = z / norm_sq
                        else:
                            U[idx_u, :] = 0 # or random
                
                # Update V
                for j in range(m_sub):
                    for k in range(p_sub):
                        idx_v = j * p_sub + k
                        z = np.zeros(rank_sub)
                        for i in range(n_sub):
                            idx_u = i * m_sub + j
                            idx_w = k * n_sub + i
                            z += U[idx_u, :] * W[idx_w, :]
                        norm_sq = np.dot(z, z)
                        if norm_sq > 1e-10:
                            V[idx_v, :] = z / norm_sq
                        else:
                            V[idx_v, :] = 0
                
                # Update W
                for k in range(p_sub):
                    for i in range(n_sub):
                        idx_w = k * n_sub + i
                        z = np.zeros(rank_sub)
                        for j in range(m_sub):
                            idx_u = i * m_sub + j
                            idx_v = j * p_sub + k
                            z += U[idx_u, :] * V[idx_v, :]
                        norm_sq = np.dot(z, z)
                        if norm_sq > 1e-10:
                            W[idx_w, :] = z / norm_sq
                        else:
                            W[idx_w, :] = 0
            
            # Check error
            error = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        idx_u = i * m_sub + j
                        idx_v = j * p_sub + k
                        idx_w = k * n_sub + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        error += (val - 1) ** 2
            
            if error < 1e-6:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded
                error_rounded = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            idx_u = i * m_sub + j
                            idx_v = j * p_sub + k
                            idx_w = k * n_sub + i
                            val = np.dot(U_rounded[idx_u, :], V_rounded[idx_v, :] * W_rounded[idx_w, :])
                            error_rounded += (val - 1) ** 2
                
                if error_rounded < 1e-9:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_U is not None:
            break

    if best_U is not None:
        # Construct full 2x3x8 decomposition
        rank = 24
        U_full = np.zeros((n * m, rank))
        V_full = np.zeros((m * p, rank))
        W_full = np.zeros((p * n, rank))
        
        # Block 1: k = 0..3
        for r in range(rank_sub):
            # U is same
            U_full[:, r] = best_U[:, r]
            
            # V: j in 0..2, k in 0..3
            for j in range(m_sub):
                for k in range(p_sub):
                    idx_v_sub = j * p_sub + k
                    idx_v_full = j * p + k
                    V_full[idx_v_full, r] = best_V[idx_v_sub, r]
            
            # W: k in 0..3, i in 0..1
            for k in range(p_sub):
                for i in range(n_sub):
                    idx_w_sub = k * n_sub + i
                    idx_w_full = k * n + i
                    W_full[idx_w_full, r] = best_W[idx_w_sub, r]
        
        # Block 2: k = 4..7
        for r in range(rank_sub):
            r2 = rank_sub + r
            # U is same
            U_full[:, r2] = best_U[:, r]
            
            # V: j in 0..2, k in 4..7
            for j in range(m_sub):
                for k in range(p_sub):
                    idx_v_sub = j * p_sub + k
                    idx_v_full = j * p + (k + p_sub)
                    V_full[idx_v_full, r2] = best_V[idx_v_sub, r]
            
            # W: k in 4..7, i in 0..1
            for k in range(p_sub):
                for i in range(n_sub):
                    idx_w_sub = k * n_sub + i
                    idx_w_full = (k + p_sub) * n + i
                    W_full[idx_w_full, r2] = best_W[idx_w_sub, r]
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
