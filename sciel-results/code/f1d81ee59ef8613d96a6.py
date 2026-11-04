"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    deadline = time.time() + 140
    
    def compute_residual_fast(U, V, W, n, m, p):
        """Compute residual using vectorized operations."""
        r = U.shape[1]
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape and use einsum
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return T - 1.0
    
    def residual_norm_fast(U, V, W, n, m, p):
        res = compute_residual_fast(U, V, W, n, m, p)
        return np.sum(res**2)
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    np.random.seed(42)
    
    for rank in range(80, 96):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            lr = 0.05
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                    
                res = compute_residual_fast(U, V, W, n, m, p)
                rn = np.sum(res**2)
                
                if rn < 1e-12:
                    break
                
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                
                # Gradients
                # d/dU[i,j,r] = V[j,k,r] * W[k,i,r] * res[i,j,k]
                dU = np.einsum('jkr,kir,ijk->ijr', V3, W3, res)
                dV = np.einsum('ijr,kir,ijk->jkr', U3, W3, res)
                dW = np.einsum('ijr,jkr,ijk->kir', U3, V3, res)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.999
            
            rn = residual_norm_fast(U, V, W, n, m, p)
            if rn < 1e-10 and rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    def to_list(mat):
        return [[float(round(mat[i][j], 10)) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
