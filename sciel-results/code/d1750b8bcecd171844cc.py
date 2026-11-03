"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    except:
        return False
        
    return np.allclose(result, 1.0, atol=1e-9)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U3[i,j,c]) * Fraction(V3[j,k,c]) * Fraction(W3[k,i,c])
                if s != Fraction(1):
                    return False
    return True

def _try_round_half(U, V, W, n, m, p):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _try_round_int(U, V, W, n, m, p):
    """Try to round entries to integers and verify."""
    U_r = [[round(x) for x in row] for row in U]
    V_r = [[round(x) for x in row] for row in V]
    W_r = [[round(x) for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_decompose_vectorized(n, m, p, rank, max_iters=300, tol=1e-10, seed=42):
    """Vectorized ALS for tensor decomposition - much faster."""
    rng = np.random.default_rng(seed)
    
    U3 = rng.standard_normal((n, m, rank))
    V3 = rng.standard_normal((m, p, rank))
    W3 = rng.standard_normal((p, n, rank))
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Update U3: for each (i,j), solve A[i,j] @ U3[i,j,:] = T[i,j,:]
        # A[i,j,k,c] = V3[j,k,c] * W3[k,i,c]
        # Vectorize: A has shape (n, m, p, rank)
        # V3[j,k,c]: need to broadcast to (n, m, p, rank)
        # W3[k,i,c]: need to broadcast to (n, m, p, rank)
        
        # V3: (m, p, rank) -> (1, m, p, rank)
        # W3: (p, n, rank) -> (p, n, 1, rank) but we need (n, 1, p, rank)
        # W3.T: (n, p, rank) -> (n, 1, p, rank)
        
        V3_b = V3[np.newaxis, :, :, :]  # (1, m, p, rank)
        W3_t = W3.transpose(1, 0, 2)    # (n, p, rank)
        W3_b = W3_t[:, np.newaxis, :, :]  # (n, 1, p, rank)
        
        A = V3_b * W3_b  # (n, m, p, rank)
        
        # Solve for each (i,j): A[i,j] (p x rank) @ U3[i,j] (rank,) = T[i,j] (p,)
        # Use lstsq in batch
        A_T = np.transpose(A, (0, 1, 3, 2))  # (n, m, rank, p)
        
        for i in range(n):
            for j in range(m):
                try:
                    U3[i, j, :] = np.linalg.lstsq(A_T[i, j], T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V3: for each (j,k), solve B[j,k] @ V3[j,k,:] = T[:,j,k]
        # B[j,k,i,c] = U3[i,j,c] * W3[k,i,c]
        # U3: (n, m, rank) -> (n, m, 1, rank) for broadcasting
        # W3: (p, n, rank) -> (p, n, 1, rank)
        
        U3_b = U3[:, :, np.newaxis, :]  # (n, m, 1, rank)
        W3_b2 = W3[np.newaxis, :, :, :]  # (1, n, p, rank)
        
        # We need B[j,k,i,c] = U3[i,j,c] * W3[k,i,c]
        # U3[i,j,c]: (n, m, rank) -> (n, m, 1, rank)
        # W3[k,i,c]: (p, n, rank) -> (1, n, p, rank)
        # Result: (n, m, p, rank) but we want (m, p, n, rank)
        
        # Let's compute differently
        B = np.zeros((m, p, n, rank), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                # U3[:, j, :] is (n, rank)
                # W3[k, :, :] is (n, rank)
                B[j, k, :, :] = U3[:, j, :] * W3[k, :, :]
        
        for j in range(m):
            for k in range(p):
                try:
                    V3[j, k, :] = np.linalg.lstsq(B[j, k, :, :].T, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W3: for each (k,i), solve C[k,i] @ W3[k,i,:] = T[i,:,k]
        # C[k,i,j,c] = U3[i,j,c] * V3[j,k,c]
        
        C = np.zeros((p, n, m, rank), dtype=np.float64)
        for k in range(p):
            for i in range(n):
                # U3[i, :, :] is (m, rank)
                # V3[:, k, :] is (m, rank)
                C[k, i, :, :] = U3[i, :, :] * V3[:, k, :]
        
        for k in range(p):
            for i in range(n):
                try:
                    W3[k, i, :] = np.linalg.lstsq(C[k, i, :, :].T, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        res = np.linalg.norm(recon - T)
        if res < tol:
            break
    
    return U3.reshape(n*m, rank), V3.reshape(m*p, rank), W3.reshape(p*n, rank)

def _structured_decomposition(n, m, p):
    """Try known structured decompositions for matrix multiplication."""
    # For 2x4 * 4x7, we can try block-based approaches
    # One approach: treat as multiple 2x4 * 4x2 and 2x4 * 4x1 subproblems
    
    # Try a simple structured approach using block diagonals
    # The schoolbook rank is 2*4*7 = 56
    # We want to find lower rank
    
    # Approach: Use the fact that we can share computations across blocks
    # For 2x4 * 4x7, we can think of it as:
    # A (2x4) * B (4x7) = C (2x7)
    
    # One structured decomposition:
    # Split B into 7 columns, each 4x1
    # A * B_col for each column
    # This gives 7 * (2*4) = 56 operations (schoolbook)
    
    # Better: Use Strassen-like ideas
    # For 2x2 * 2x2, Strassen uses 7 mults
    # For 2x4 * 4x2, we can potentially use fewer
    
    # Let's try a direct structured approach
    # We'll define U, V, W with a specific pattern
    
    # Actually, let's just try random ALS with many restarts and good rounding
    return None

def _try_rank_with_als(n, m, p, rank, n_trials=10, max_iters=200):
    """Try to find a valid decomposition at given rank using ALS."""
    for trial in range(n_trials):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        seed = hash((n, m, p, rank, trial)) % (2**31)
        U, V, W = _als_decompose_vectorized(n, m, p, rank, max_iters=max_iters, tol=1e-10, seed=seed)
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Try half-integer rounding
        rounded = _try_round_half(U_list, V_list, W_list, n, m, p)
        if rounded is not None:
            return rounded
        
        # Try integer rounding
        rounded = _try_round_int(U_list, V_list, W_list, n, m, p)
        if rounded is not None:
            return rounded
        
        # Try mixed: round some to half-integers, others to integers
        # This is more complex; skip for now
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Try ranks from 55 down to 28
    for rank in range(rank_school - 1, 27, -1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        result = _try_rank_with_als(n, m, p, rank, n_trials=5, max_iters=150)
        if result is not None:
            U_r, V_r, W_r = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
            return best_result
    
    return best_result
# EVOLVE-BLOCK-END
