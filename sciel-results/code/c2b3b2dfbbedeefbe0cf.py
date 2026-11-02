"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != 1 or len(V[r]) != 1 or len(W[r]) != 1:
            return False
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T should equal 1 if i,j,k all valid, else 0
    # Actually the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, the matrix multiplication tensor is: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, max_restarts=5, time_limit=15.0):
    """Try to find a rank-`rank` decomposition via numerical optimization using scipy."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def unpack(x):
        U = x[:dim_u * rank].reshape((dim_u, rank))
        V = x[dim_u * rank : dim_u * rank + dim_v * rank].reshape((dim_v, rank))
        W = x[dim_u * rank + dim_v * rank :].reshape((dim_w, rank))
        return U, V, W

    def loss_and_grad(x):
        U, V, W = unpack(x)
        
        U_3d = U.reshape((n, m, rank))
        V_3d = V.reshape((m, p, rank))
        W_3d = W.reshape((p, n, rank))
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        residual = T - 1.0
        loss = np.sum(residual ** 2)
        
        dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', residual, V_3d, W_3d)
        dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', residual, U_3d, W_3d)
        dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', residual, U_3d, V_3d)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        return loss, grad

    best_result = None
    best_loss = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - t0 > time_limit:
            break
            
        x0 = np.random.uniform(-0.5, 0.5, size=(dim_u + dim_v + dim_w) * rank)
        
        try:
            res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x
                
                if best_loss < 1e-10:
                    U, V, W = unpack(best_result)
                    U_snap = np.round(U * 2) / 2.0
                    V_snap = np.round(V * 2) / 2.0
                    W_snap = np.round(W * 2) / 2.0
                    
                    if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                         return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                         
        except Exception:
            pass

    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try to find a lower-rank decomposition
    # Known lower bounds: for 3x3x6, the rank is at least max(3*3, 3*6, 6*3) = 18
    # Try ranks from 53 down to 45
    # We allocate time per rank attempt
    ranks_to_try = list(range(53, 44, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on remaining time
        time_remaining = deadline - time.time()
        time_per_attempt = min(15.0, time_remaining / max(1, len(ranks_to_try)))
        
        result = _try_rank(rank, max_restarts=3, time_limit=time_per_attempt)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
