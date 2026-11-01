"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    deadline = time.time() + 100  # 100s budget, leave margin
    
    # Try ranks from schoolbook-1 down
    # We only try a few ranks quickly to avoid wasting time
    for rank in range(schoolbook_rank - 1, 6, -1):
        if time.time() > deadline:
            break
        result = _try_rank_fast(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
            
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_fast(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition of T using a fast random search."""
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # Limit tries to keep it fast
    num_tries = 10
    
    for trial in range(num_tries):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Simple gradient descent with few iterations
        lr = 0.05
        for it in range(100):
            if time.time() > deadline:
                return None
            tensor = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = (tensor - T)
            dU = np.einsum('abd,bc,dc->ac', residual, V, W)
            dV = np.einsum('abd,ac,dc->bc', residual, U, W)
            dW = np.einsum('abd,ac,bc->dc', residual, U, V)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if np.sum(residual**2) < 1e-10:
                break
        
        if np.sum((np.einsum('ac,bc,dc->abd', U, V, W) - T)**2) < 1e-10:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify(T, U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)
# EVOLVE-BLOCK-END
