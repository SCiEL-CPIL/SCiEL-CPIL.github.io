import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Target: Find rank for 3x4x4, then double it.
    # Schoolbook for 3x4x4 is 48.
    # Schoolbook for 3x4x8 is 96.
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Build tensor for 3x4x4
    n_sub = 3
    m_sub = 4
    p_sub = 4
    
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0
                
    best_rank_sub = 48
    best_decomp_sub = None
    
    # Try to find lower rank for 3x4x4
    for rank_sub in [47, 46, 45, 44]:
        if time.time() > deadline - 20:
            break
            
        for seed in range(20):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(seed)
            U = np.random.randn(n_sub * m_sub, rank_sub) * 0.1
            V = np.random.randn(m_sub * p_sub, rank_sub) * 0.1
            W = np.random.randn(p_sub * n_sub, rank_sub) * 0.1
            
            # Gradient Descent
            lr = 0.01
            converged = False
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                    
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T_sub - S
                norm = np.linalg.norm(residual)
                
                if norm < 1e-6:
                    converged = True
                    break
                    
                grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W)
                grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W)
                grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                S_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(S_r, T_sub, atol=1e-6):
                    best_rank_sub = rank_sub
                    best_decomp_sub = (U_r, V_r, W_r)
                    break # Found a solution for this rank, try lower rank?
                    # We break inner loop, continue to next lower rank in outer loop?
                    # The loop is `for rank_sub in ...`, so yes, it will try lower rank.
                    # But we should probably stop if we find a very low rank?
                    # Let's just take the first one found in descending order of rank?
                    # No, we want minimal rank. So we should try 44 first?
                    # The list is [47, 46, 45, 44]. It tries 47 first.
                    # If 47 works, it sets best_rank_sub=47.
                    # Then it tries 46. If 46 works, best_rank_sub=46.
                    # This is correct.

    # Construct final result
    if best_decomp_sub is not None:
        U_sub, V_sub, W_sub = best_decomp_sub
        rank_sub = best_rank_sub
        total_rank = 2 * rank_sub
        
        # Build U_total (12 x total_rank)
        U_total = np.zeros((n * m, total_rank))
        U_total[:, :rank_sub] = U_sub
        U_total[:, rank_sub:] = U_sub
        
        # Build V_total (32 x total_rank)
        V_total = np.zeros((m * p, total_rank))
        # First half of V_total corresponds to B1 (k=0..3)
        # V_sub has shape (16, rank_sub). Indices j*4+k.
        # V_total indices j*8+k.
        for j in range(m_sub):
            for k in range(p_sub):
                V_total[j * p + k, :rank_sub] = V_sub[j * p_sub + k, :]
                V_total[j * p + (k + p_sub), rank_sub:] = V_sub[j * p_sub + k, :]
                
        # Build W_total (24 x total_rank)
        W_total = np.zeros((p * n, total_rank))
        # W_sub has shape (16, rank_sub). Indices k*3+i.
        # W_total indices k*3+i.
        for k in range(p_sub):
            for i in range(n_sub):
                W_total[k * n + i, :rank_sub] = W_sub[k * p_sub + i, :]
                W_total[(k + p_sub) * n + i, rank_sub:] = W_sub[k * p_sub + i, :]
                
        return {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    else:
        # Fallback to schoolbook 96
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}
