"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B."""
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i*m+j, j*p+k, k*n+i] != 1:
                    return False
    required = set((i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if T[a,b,c] != 0 and (a,b,c) not in required:
                    return False
    return True

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

def _find_sub_decomp(n, m, p, rank, seed=42, iters=300):
    """Find rank decomposition of n×m × m×p multiplication tensor via ALS."""
    np.random.seed(seed)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(iters):
        M = np.zeros((rank, m*p*p*n))
        for r in range(rank):
            M[r, :] = np.outer(V[:,r], W[:,r]).flatten()
        U_new, _, _, _ = np.linalg.lstsq(M.T, T.reshape(n*m, -1).T, rcond=None)
        U = U_new.T
        
        M = np.zeros((rank, n*m*p*n))
        for r in range(rank):
            M[r, :] = np.outer(U[:,r], W[:,r]).flatten()
        V_new, _, _, _ = np.linalg.lstsq(M.T, T.reshape(m*p, -1).T, rcond=None)
        V = V_new.T
        
        M = np.zeros((rank, n*m*m*p))
        for r in range(rank):
            M[r, :] = np.outer(U[:,r], V[:,r]).flatten()
        W_new, _, _, _ = np.linalg.lstsq(M.T, T.reshape(p*n, -1).T, rcond=None)
        W = W_new.T
    
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    if np.max(np.abs(T_recon - T)) > 1e-6:
        return None
    return U, V, W

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10 using computed sub-decomposition."""
    if n != 2 or m != 3 or p != 10:
        return None
    
    # Find rank-11 decomposition for 2x3 x 3x2 sub-problem
    sub = _find_sub_decomp(2, 3, 2, 11, seed=42, iters=300)
    if sub is None:
        # Try different seeds
        for seed in [1, 7, 13, 99, 123]:
            sub = _find_sub_decomp(2, 3, 2, 11, seed=seed, iters=300)
            if sub is not None:
                break
    if sub is None:
        return None
    
    U_sub, V_sub, W_sub = sub
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for t in range(5):
        k0 = 2*t
        k1 = 2*t + 1
        for r in range(11):
            col = t*11 + r
            # U_sub has shape (6, 11), indices a = i*3+j
            U[:, col] += U_sub[:, r]
            # V_sub has shape (6, 11), indices b = j*2+k (k in {0,1})
            # Map to full: b_full = j*10 + (k0 + k)
            for j in range(3):
                for k in range(2):
                    b_sub = j*2 + k
                    b_full = j*10 + k0 + k
                    V[b_full, col] += V_sub[b_sub, r]
            # W_sub has shape (4, 11), indices c = k*2+i (k in {0,1})
            # Map to full: c_full = (k0+k)*2 + i
            for k in range(2):
                for i in range(2):
                    c_sub = k*2 + i
                    c_full = (k0 + k)*2 + i
                    W[c_full, col] += W_sub[c_sub, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if res55 is not None and time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
