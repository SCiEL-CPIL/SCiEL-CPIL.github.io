import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback to schoolbook
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = rank_schoolbook
    
    # Strategy: Find rank for 3x4x4, then double it.
    # Target ranks for 3x4x4: 47, 46, 45, 44, 43, 42, 41, 40
    # Schoolbook for 3x4x4 is 48.
    
    n_sub = 3
    m_sub = 4
    p_sub = 4
    
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0
                
    # Precompute gradients helper
    def compute_gradients(residual, U, V, W):
        grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W)
        grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W)
        grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V)
        return grad_U, grad_V, grad_W

    def try_rank_sub(rank_sub, max_seeds=10):
        if time.time() > deadline - 10:
            return None
        
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(n_sub * m_sub, rank_sub) * 0.1
            V = np.random.randn(m_sub * p_sub, rank_sub) * 0.1
            W = np.random.randn(p_sub * n_sub, rank_sub) * 0.1
            
            # Adam Optimizer
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            b1 = 0.9; b2 = 0.999; eps = 1e-8
            lr = 0.01
            t = 0
            
            converged = False
            # Limit steps to keep it fast
            for step in range(2000):
                if time.time() > deadline - 10:
                    return None
                
                t += 1
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T_sub - S
                norm = np.linalg.norm(residual)
                
                if norm < 1e-5:
                    converged = True
                    break
                
                grad_U, grad_V, grad_W = compute_gradients(residual, U, V, W)
                
                # Update Adam moments
                mU = b1 * mU + (1 - b1) * grad_U
                vU = b2 * vU + (1 - b2) * (grad_U ** 2)
                mU_hat = mU / (1 - b1 ** t)
                vU_hat = vU / (1 - b2 ** t)
                
                mV = b1 * mV + (1 - b1) * grad_V
                vV = b2 * vV + (1 - b2) * (grad_V ** 2)
                mV_hat = mV / (1 - b1 ** t)
                vV_hat = vV / (1 - b2 ** t)
                
                mW = b1 * mW + (1 - b1) * grad_W
                vW = b2 * vW + (1 - b2) * (grad_W ** 2)
                mW_hat = mW / (1 - b1 ** t)
                vW_hat = vW / (1 - b2 ** t)
                
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                S_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(S_r, T_sub, atol=1e-6):
                    return (U_r, V_r, W_r)
                    
        return None

    # Try ranks in descending order of likelihood (closest to schoolbook first? or lowest first?)
    # Lowest first is better for score, but harder.
    # Let's try 40, 44, 48-1=47.
    target_ranks = [40, 44, 47]
    
    for rank_sub in target_ranks:
        if time.time() > deadline - 5:
            break
            
        decomp_sub = try_rank_sub(rank_sub, max_seeds=5)
        if decomp_sub is not None:
            U_sub, V_sub, W_sub = decomp_sub
            total_rank = 2 * rank_sub
            
            # Build U_total (12 x total_rank)
            U_total = np.zeros((n * m, total_rank))
            U_total[:, :rank_sub] = U_sub
            U_total[:, rank_sub:] = U_sub
            
            # Build V_total (32 x total_rank)
            V_total = np.zeros((m * p, total_rank))
            for j in range(m_sub):
                for k in range(p_sub):
                    V_total[j * p + k, :rank_sub] = V_sub[j * p_sub + k, :]
                    V_total[j * p + (k + p_sub), rank_sub:] = V_sub[j * p_sub + k, :]
                    
            # Build W_total (24 x total_rank)
            W_total = np.zeros((p * n, total_rank))
            for k in range(p_sub):
                for i in range(n_sub):
                    W_total[k * n + i, :rank_sub] = W_sub[k * p_sub + i, :]
                    W_total[(k + p_sub) * n + i, rank_sub:] = W_sub[k * p_sub + i, :]
                    
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }

    return best_result
