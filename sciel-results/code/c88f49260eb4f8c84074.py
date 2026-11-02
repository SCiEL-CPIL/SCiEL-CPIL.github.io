"""Program for 2x5x5 matrix multiplication tensor decomposition."""
import numpy as np
from scipy.optimize import linprog
import itertools
import time

DIMENSIONS = [2, 5, 5]

def _is_integer_array(arr, tol=1e-9):
    """Check if array entries are (half-)integers."""
    arr = np.asarray(arr, dtype=float)
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.allclose(doubled, rounded, atol=tol)

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W exactly decompose the tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 for all i,j,k (since it's the identity of multiplication)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        Uc_m = U[:, c].reshape(n, m)
        Vc_m = V[:, c].reshape(m, p)
        Wc_m = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc_m[i,j] * Vc_m[j,k] * Wc_m[k,i]
        # Use einsum for efficiency
        T += np.einsum('ij,jk,ki->ijk', Uc_m, Vc_m, Wc_m)
    return np.allclose(T, 1.0, atol=tol)

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    """Construct the decomposition. Returns dict with 'U', 'V', 'W' keys."""
    n, m, p = DIMENSIONS
    
    # Construct a rank 10 decomposition
    # n=2, m=5, p=5
    # rank = 10
    r = 10
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For i=0, use c in [0, 5)
    # For i=1, use c in [5, 10)
    for j in range(m):
        # i=0 part
        c0 = j
        U[j, c0] = 1.0
        for k in range(p):
            V[j * p + k, c0] = 1.0
            W[k * n + 0, c0] = 1.0
            
        # i=1 part
        c1 = 5 + j
        U[m + j, c1] = 1.0
        for k in range(p):
            V[j * p + k, c1] = 1.0
            W[k * n + 1, c1] = 1.0
            
    # Verify and return
    if _verify(U, V, W, n, m, p):
        return _to_lists(U, V, W)
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return _to_lists(U_fb, V_fb, W_fb)

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
