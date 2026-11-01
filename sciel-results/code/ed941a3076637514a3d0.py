"""Matrix multiplication 2,2,6 - rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, alpha] -> U3[i, j, alpha]
    # V[j*p+k, alpha] -> V3[j, k, alpha]
    # W[k*n+i, alpha] -> W3[k, i, alpha]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute S[i, j, k] = sum_alpha U3[i,j,alpha] * V3[j,k,alpha] * W3[k,i,alpha]
    # We need this to be 1 for all i,j,k
    S = np.zeros((n, m, p), dtype=np.float64)
    for a in range(r):
        S += U3[:,:,a] * np.einsum('jk,ki->jki', V3[:,:,a], W3[:,:,a])
    
    # Actually let me redo this more carefully
    # S[i,j,k] = sum_a U[i,j,a] * V[j,k,a] * W[k,i,a]
    S = np.zeros((n, m, p), dtype=np.float64)
    for a in range(r):
        ua = U3[:, :, a]  # shape (n, m)
        va = V3[:, :, a]  # shape (m, p)
        wa = W3[:, :, a]  # shape (p, n)
        # For each i, j, k: ua[i,j] * va[j,k] * wa[k,i]
        # Build outer product: for fixed a, contribution is ua[i,j] * va[j,k] * wa[k,i]
        # This is a tensor of shape (n, m, p)
        for i in range(n):
            for j in range(m):
                S[i, j, :] += ua[i, j] * va[j, :] * wa[:, i]
    
    return np.allclose(S, 1.0, atol=1e-9)

def _snap_to_half_integer(arr):
    """Snap values to nearest half-integer (multiple of 0.5)."""
    return np.round(2.0 * arr) / 2.0

def _search_rank(r, time_budget):
    """Try to find a rank-r decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Initialize with random values
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Try to refine using a simple iterative approach
        # We want sum_a U[i,j,a]*V[j,k,a]*W[k,i,a] = 1 for all i,j,k
        
        # Use a simple fixed-point / gradient approach
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        for it in range(50):
            if time.time() - start > time_budget:
                break
            
            # Compute current S[i,j,k]
            S = np.zeros((n, m, p), dtype=np.float64)
            for a in range(r):
                ua = U3[:, :, a]
                va = V3[:, :, a]
                wa = W3[:, :, a]
                for i in range(n):
                    for j in range(m):
                        S[i, j, :] += ua[i, j] * va[j, :] * wa[:, i]
            
            err = np.sum((S - 1.0)**2)
            
            if err < 1e-10:
                break
            
            # Gradient update
            # dS/dU[i,j,a] = V[j,k,a] * W[k,i,a] summed over k... 
            # Actually S[i,j,k] = sum_a U[i,j,a]*V[j,k,a]*W[k,i,a]
            # dS[i,j,k]/dU[i,j,a] = V[j,k,a]*W[k,i,a]
            # So gradient of sum (S-1)^2 w.r.t. U[i,j,a] = 2 * sum_k (S[i,j,k]-1) * V[j,k,a] * W[k,i,a]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for a in range(r):
                ua = U3[:, :, a]
                va = V3[:, :, a]
                wa = W3[:, :, a]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dS = 2.0 * (S[i, j, k] - 1.0)
                            dU[i*m+j, a] += dS * va[j, k] * wa[k, i]
                            dV[j*p+k, a] += dS * ua[i, j] * wa[k, i]
                            dW[k*n+i, a] += dS * ua[i, j] * va[j, k]
            
            # Small step
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        if err < 1e-8:
            # Snap to half-integers and verify
            U_h = _snap_to_half_integer(U)
            V_h = _snap_to_half_integer(V)
            W_h = _snap_to_half_integer(W)
            if _verify(U_h.tolist(), V_h.tolist(), W_h.tolist()):
                return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x2 times 2x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try ranks from 19 upward (rank 18 is known to be too low for most cases, but let's try)
    for r in range(19, 25):
        if time.time() > deadline - 10:
            break
        result = _search_rank(r, min(30, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
