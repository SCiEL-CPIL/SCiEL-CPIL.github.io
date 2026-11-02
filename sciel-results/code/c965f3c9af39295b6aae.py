# EVOLVE-BLOCK-START

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomposition():
    # ... (keep existing)
    pass

def _find_rank_16_1x4x8():
    n, m, p = 1, 4, 8
    rank = 16
    
    def loss(params):
        U = params[:m*rank].reshape(m, rank)
        V = params[m*rank : m*rank + m*p*rank].reshape(m, p, rank)
        W = params[m*rank + m*p*rank :].reshape(p, rank)
        
        # Condition: sum_r W[k,r] U[j,r] V[j,k,r] = delta[j,k] ? No.
        # T[j,k] = 1 for all j,k? No.
        # C[k] = sum_j A[j] B[j,k].
        # Coeff of A[j] B[j',k'] is delta[j,j'] delta[k,k']? No.
        # It is delta[j,j'] if k=k'? No.
        # A[j] multiplies B[j,k].
        # So coeff of A[j] B[j',k'] is 1 if j=j' and k=k'? No.
        # If j!=j', coeff is 0.
        # If j=j', coeff is 1 if k=k'? No, B[j,k] is the term.
        # So coeff of A[j] B[j',k'] is 1 if j=j' and k=k'?
        # Wait. C[k] = sum_j A[j] B[j,k].
        # Term A[j] B[j',k'] appears with coeff 1 if j=j' and k=k'?
        # Yes.
        # So T[j,j',k'] = 1 if j=j' and k=k'? No.
        # T is 3-indexed? No, U is 1D, V is 2D, W is 1D.
        # The tensor is T[j, j', k']? No.
        # The contraction is sum_r W[k,r] U[j,r] V[j',k',r].
        # We want this to be 1 if j=j' and k=k'? No.
        # We want C[k] = sum_j A[j] B[j,k].
        # So the coeff of A[j] B[j',k'] in the sum should be 1 if j=j' and k=k'?
        # Yes.
        # So T[j, j', k'] = 1 if j=j' and k=k', else 0.
        # But V is indexed by j', k'.
        # So we want sum_r W[k,r] U[j,r] V[j',k',r] = delta[j,j'] delta[k,k']?
        # No, if j!=j', it should be 0.
        # If j=j', it should be delta[k,k']?
        # Yes.
        
        T_computed = np.einsum('kr,jr,jkr->jjk', W, U, V) # Wrong indices
        # V is (m, p, rank) -> V[j', k', r]
        # U is (m, rank) -> U[j, r]
        # W is (p, rank) -> W[k, r]
        # Result index: j, j', k.
        # We want Result[j, j', k] = 1 if j=j' and k=k? No.
        # We want C[k] = sum_j A[j] B[j,k].
        # So the term A[j] B[j',k'] contributes to C[k] if k=k'.
        # And it should be A[j] B[j,k].
        # So coeff is 1 if j=j' and k=k'.
        # So T_target[j, j', k] = 1 if j=j' and k=k? No.
        # T_target[j, j', k] = 1 if j=j' and k=k?
        # Yes.
        
        T_target = np.zeros((m, m, p))
        for j in range(m):
            for k in range(p):
                T_target[j, j, k] = 1
                
        # Compute T_computed[j, j', k] = sum_r U[j,r] V[j',k,r] W[k,r]
        T_computed = np.einsum('jr,jkr,kr->jjk', U, V, W)
        
        error = np.sum((T_computed - T_target)**2)
        return error

    # Random init
    np.random.seed(123)
    best_params = None
    best_loss = float('inf')
    
    for _ in range(5):
        x0 = np.random.randn(m*rank + m*p*rank + p*rank) * 0.5
        res = minimize(loss, x0, method='BFGS', options={'maxiter': 50})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
    if best_loss < 1e-6:
        U = best_params[:m*rank].reshape(m, rank).tolist()
        V = best_params[m*rank : m*rank + m*p*rank].reshape(m, p, rank).tolist()
        W = best_params[m*rank + m*p*rank :].reshape(p, rank).tolist()
        return U, V, W
        
    return None

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 4, 8
    
    # Try to find rank 32 decomposition
    # Use two rank 16 decompositions for 1x4x8
    U_1x4x8, V_1x4x8, W_1x4x8 = _find_rank_16_1x4x8()
    
    if U_1x4x8 is not None:
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: i=0
        for r in range(16):
            for j in range(m):
                U[0 * m + j][r] = U_1x4x8[j][r]
                V[j * p + k][r] = V_1x4x8[j][k][r] # Wait, V is m x p x rank
                # V_1x4x8[j][k][r]
            for k in range(p):
                W[k * n + 0][r] = W_1x4x8[k][r]
                
        # Block 2: i=1
        for r in range(16):
            for j in range(m):
                U[1 * m + j][16 + r] = U_1x4x8[j][r]
                V[j * p + k][16 + r] = V_1x4x8[j][k][r]
            for k in range(p):
                W[k * n + 1][16 + r] = W_1x4x8[k][r]
                
        # Verify
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T_expected = np.ones((n, m, p))
        
        if np.allclose(T_computed, T_expected, atol=1e-6):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to Rank 56
    # ... (existing code)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
