import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def verify_decomposition(U, V, W, n=2, m=2, p=12):
    """Verify that the decomposition is correct."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # The decomposition should satisfy:
    # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # This is the matrix multiplication tensor.
    
    # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k)
    # And T = sum_t U_t ⊗ V_t ⊗ W_t
    # So T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    # Wait, the indices are:
    # U has shape (n*m, rank): indexed by (i,j) → i*m+j
    # V has shape (m*p, rank): indexed by (j,k) → j*p+k
    # W has shape (p*n, rank): indexed by (k,i) → k*n+i
    
    # The tensor element T[i,j,k] should equal sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # And T[i,j,k] = 1 for all i,j,k (since it's the matrix multiplication tensor)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook: rank 48
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist()):
        # Schoolbook works
        pass
    else:
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    # Try Strassen-based: rank 42
    if time.time() < deadline:
        try:
            U42, V42, W42 = strassen_decomposition(n, m, p)
            if verify_decomposition(U42.tolist(), V42.tolist(), W42.tolist()):
                best_rank = 42
                best_U, best_V, best_W = U42, V42, W42
        except Exception:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def strassen_decomposition(n, m, p):
    """Try to construct a rank-42 decomposition using Strassen for 2x2x2 blocks."""
    # This is a placeholder - need correct Strassen formulas
    raise NotImplementedError
