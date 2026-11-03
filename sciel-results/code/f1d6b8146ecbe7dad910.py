"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 130  # seconds, with safety margin

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a flat array."""
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    return tensor

def _check_decomposition(U, V, W, tensor, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    r = U.shape[1]
    T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if not np.allclose(T_computed, tensor, atol=tol, rtol=0):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W

def _modinv(a, p):
    """Modular inverse of a mod p."""
    return pow(int(a) % p, -1, p)

def _gaussian_elim_gf(A, b, p):
    """
    Solve A @ x = b over GF(p).
    A: (m, n) matrix, b: (m,) vector
    Returns x: (n,) solution, or None if inconsistent.
    Sets free variables to 0.
    """
    m, n = A.shape
    Aug = np.zeros((m, n + 1), dtype=np.int64)
    Aug[:, :n] = A % p
    Aug[:, n] = b % p
    
    rank = 0
    for col in range(n):
        if rank >= m:
            break
        
        # Find pivot
        pivot_row = -1
        for row in range(rank, m):
            if Aug[row, col] % p != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap
        if pivot_row != rank:
            Aug[[rank, pivot_row]] = Aug[[pivot_row, rank]]
        
        # Normalize
        inv = _modinv(Aug[rank, col], p)
        Aug[rank, :] = (Aug[rank, :] * inv) % p
        
        # Eliminate
        for row in range(m):
            if row == rank:
                continue
            factor = Aug[row, col] % p
            if factor != 0:
                Aug[row, :] = (Aug[row, :] - factor * Aug[rank, :]) % p
        
        rank += 1
    
    # Check consistency
    for row in range(rank, m):
        if np.any(Aug[row, :n] % p != 0) and Aug[row, n] % p != 0:
            return None
    
    # Extract solution
    x = np.zeros(n, dtype=np.int64)
    for col in range(n):
        for row in range(m):
            if Aug[row, col] % p != 0:
                x[col] = Aug[row, n] % p
                break
    
    return x

def _gf_min_rank(n, m, p, max_rank=50, max_time=100):
    """
    Find minimum rank decomposition over small finite fields.
    """
    start_time = time.time()
    
    # Precompute tensor non-zero pattern
    tensor = _build_tensor(n, m, p)
    
    # For each (a,b), precompute t_ab[c] = T[a,b,c]
    t_ab = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                t_ab[a, b, c] = 1
    
    for field_p in [2, 3, 5, 7]:
        if time.time() - start_time > max_time:
            break
        
        for rank in range(28, min(max_rank + 1, 55)):
            if time.time() - start_time > max_time:
                break
            
            max_attempts = 10 if rank < 40 else 5
            
            for attempt in range(max_attempts):
                if time.time() - start_time > max_time:
                    break
                
                rng = np.random.default_rng(rank * 10000 + attempt * 7 + field_p * 1000 + 42)
                
                # Random sparse W over GF(field_p)
                W = np.zeros((p*n, rank), dtype=np.int64)
                for c in range(rank):
                    nz_count = int(rng.integers(1, 3))
                    nz_indices = rng.choice(p*n, size=nz_count, replace=False)
                    for idx in nz_indices:
                        W[idx, c] = int(rng.integers(1, field_p))
                
                # For each (a,b), solve W @ x = t_ab over GF(field_p)
                # W is (21, rank), t_ab is (21,)
                # x is (rank,)
                
                X = np.zeros((n*m, m*p, rank), dtype=np.int64)
                consistent = True
                
                for a in range(n*m):
                    for b in range(m*p):
                        x_sol = _gaussian_elim_gf(W, t_ab[a, b, :], field_p)
                        if x_sol is None:
                            consistent = False
                            break
                        X[a, b, :] = x_sol
                    
                    if not consistent:
                        break
                
                if not consistent:
                    continue
                
                # Check if X[a,b,r] = U[a,r] * V[b,r] for some U, V
                # For each r, X[:,:,r] should be outer(U[:,r], V[:,r])
                
                U = np.zeros((n*m, rank), dtype=np.int64)
                V = np.zeros((m*p, rank), dtype=np.int64)
                valid = True
                
                for r in range(rank):
                    X_r = X[:, :, r]  # (n*m, m*p)
                    
                    # Find a non-zero entry
                    nz = np.argwhere(X_r % field_p != 0)
                    if len(nz) == 0:
                        continue
                    
                    a0, b0 = nz[0]
                    val0 = int(X_r[a0, b0] % field_p)
                    
                    # Set U[a0,r] = 1, V[b0,r] = val0
                    U[a0, r] = 1
                    V[b0, r] = val0
                    
                    # Verify
                    for a in range(n*m):
                        for b in range(m*p):
                            expected = int((U[a, r] * V[b, r]) % field_p)
                            actual = int(X_r[a, b] % field_p)
                            if expected != actual:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    # Verify over reals
                    U_f = U.astype(np.float64)
                    V_f = V.astype(np.float64)
                    W_f = W.astype(np.float64)
                    
                    T_comp = np.einsum('ar,br,cr->abc', U_f, V_f, W_f, optimize=True)
                    if np.allclose(T_comp, tensor, atol=1e-9):
                        return U_f, V_f, W_f, rank
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try finite-field approach
    if time.time() - start_time < TIME_LIMIT - 10:
        result = _gf_min_rank(n, m, p, max_rank=55, max_time=TIME_LIMIT - 20)
        if result is not None:
            U_found, V_found, W_found, rank_found = result
            if rank_found < best_rank:
                best_rank = rank_found
                best_U, best_V, best_W = U_found, V_found, W_found
    
    # Convert to lists for return
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
