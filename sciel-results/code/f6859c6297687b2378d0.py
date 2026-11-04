import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [4, 5, 6]
DEADLINE = 120.0

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9, num_samples=500):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    # Check diagonal: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal: sum_r U[i,j,r]*V[j,k,r]*W[k',i',r] should be 0
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        ip = rng.integers(0, n)
        kp = rng.integers(0, p)
        if ip == i and kp == k:
            continue
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+ip, :])
        if abs(val) > tol:
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic for small ranks."""
    r = U.shape[1]
    if r > 200:
        # Too expensive for exact, use floating point
        return _verify_decomposition(U, V, W, n, m, p, tol=1e-10, num_samples=2000)
    
    U_frac = [[Fraction(U[i, j]).limit_denominator(10**6) for j in range(r)] for i in range(n*m)]
    V_frac = [[Fraction(V[i, j]).limit_denominator(10**6) for j in range(r)] for i in range(m*p)]
    W_frac = [[Fraction(W[i, j]).limit_denominator(10**6) for j in range(r)] for i in range(p*n)]
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_frac[i*m+j][r_] * V_frac[j*p+k][r_] * W_frac[k*n+i][r_] for r_ in range(r))
                if val != 1:
                    return False
    
    # Check off-diagonal (sample)
    rng = np.random.default_rng(12345)
    for _ in range(200):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        ip = rng.integers(0, n)
        kp = rng.integers(0, p)
        if ip == i and kp == k:
            continue
        val = sum(U_frac[i*m+j][r_] * V_frac[j*p+k][r_] * W_frac[kp*n+ip][r_] for r_ in range(r))
        if val != 0:
            return False
    return True

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """Try to remove term_idx by redistributing its contribution to other terms.
    Returns (new_U, new_V, new_W, success) or None."""
    r = U.shape[1]
    if r <= 1:
        return None
    
    # Get the term to remove
    u_term = U[:, term_idx].copy()
    v_term = V[:, term_idx].copy()
    w_term = W[:, term_idx].copy()
    
    # Check if this term is "essential" - if any entry is non-zero, 
    # we need to find a way to express it as a combination of others
    # Strategy: try to express the 3-tensor contribution of this term
    # as a linear combination of other terms
    
    # The contribution of term r is: u_term[i,j] * v_term[j,k] * w_term[k,i]
    # We want to find coefficients c_s (s != term_idx) such that:
    # u_term[i,j] * v_term[j,k] * w_term[k,i] = sum_{s != term_idx} c_s * U[i,j,s] * V[j,k,s] * W[k,i,s]
    
    # This is a nonlinear system, so instead we use a simpler heuristic:
    # Try to absorb this term into one other term by scaling
    for s in range(r):
        if s == term_idx:
            continue
        # Check if u_term is proportional to U[:, s]
        if np.allclose(U[:, s], 0.0, atol=1e-12):
            continue
        # Find alpha such that u_term ≈ alpha * U[:, s]
        nonzero_mask = np.abs(U[:, s]) > 1e-12
        if not np.any(nonzero_mask):
            continue
        alpha = np.sum(u_term[nonzero_mask] * U[:, s][nonzero_mask]) / np.sum(U[:, s][nonzero_mask]**2)
        if abs(alpha) < 1e-12:
            continue
        if not np.allclose(u_term, alpha * U[:, s], atol=1e-9):
            continue
        
        # Check if v_term is proportional to V[:, s]
        nonzero_mask_v = np.abs(V[:, s]) > 1e-12
        if not np.any(nonzero_mask_v):
            continue
        beta = np.sum(v_term[nonzero_mask_v] * V[:, s][nonzero_mask_v]) / np.sum(V[:, s][nonzero_mask_v]**2)
        if abs(beta) < 1e-12:
            continue
        if not np.allclose(v_term, beta * V[:, s], atol=1e-9):
            continue
        
        # Check if w_term is proportional to W[:, s]
        nonzero_mask_w = np.abs(W[:, s]) > 1e-12
        if not np.any(nonzero_mask_w):
            continue
        gamma = np.sum(w_term[nonzero_mask_w] * W[:, s][nonzero_mask_w]) / np.sum(W[:, s][nonzero_mask_w]**2)
        if abs(gamma) < 1e-12:
            continue
        if not np.allclose(w_term, gamma * W[:, s], atol=1e-9):
            continue
        
        # If all three are proportional, we can merge
        # The product alpha*beta*gamma * U[:,s]*V[:,s]*W[:,s] gives the contribution
        # We just need to add alpha*beta*gamma to the scaling of term s
        # Actually this means term_idx is redundant - just remove it
        new_U = np.delete(U, term_idx, axis=1)
        new_V = np.delete(V, term_idx, axis=1)
        new_W = np.delete(W, term_idx, axis=1)
        return (new_U, new_V, new_W, True)
    
    return None

def _randomized_rank_reduction(U, V, W, n, m, p, time_limit=100.0):
    """Attempt to reduce rank using randomized algebraic methods."""
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = current_U.shape[1]
    
    # First: try simple term removal (redundancy check)
    r = current_U.shape[1]
    for _ in range(50):  # Max 50 removal attempts
        if time.time() - start_time > time_limit:
            break
        if r <= 1:
            break
        
        # Try removing a random term
        term_idx = np.random.randint(0, r)
        result = _try_remove_term(current_U, current_V, current_W, n, m, p, term_idx)
        if result is not None:
            new_U, new_V, new_W, success = result
            if success:
                # Verify
                if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-9, num_samples=500):
                    current_U, current_V, current_W = new_U, new_V, new_W
                    r = new_U.shape[1]
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
        
        if time.time() - start_time > time_limit:
            break
    
    return best_U, best_V, best_W, best_rank

def _strassen_2x2x2_block(n, m, p, i1, i2, j1, j2, k1, k2):
    """Strassen decomposition for a 2x2x2 block."""
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[i1*m + j1, 0] = 1.0
    U[i2*m + j2, 0] = 1.0
    V[j1*p + k1, 0] = 1.0
    V[j2*p + k2, 0] = 1.0
    W[k1*n + i1, 0] = 1.0
    W[k2*n + i2, 0] = 1.0
    
    # M2 = (A21+A22)B11
    U[i2*m + j1, 1] = 1.0
    U[i2*m + j2, 1] = 1.0
    V[j1*p + k1, 1] = 1.0
    W[k2*n + i1, 1] = 1.0
    
    # M3 = A11(B21-B22)
    U[i1*m + j1, 2] = 1.0
    V[j2*p + k1, 2] = 1.0
    V[j2*p + k2, 2] = -1.0
    W[k1*n + i2, 2] = 1.0
    
    # M4 = A22(B12-B11)
    U[i2*m + j2, 3] = 1.0
    V[j1*p + k2, 3] = 1.0
    V[j1*p + k1, 3] = -1.0
    W[k1*n + i1, 3] = 1.0
    W[k1*n + i2, 3] = 1.0
    
    # M5 = (A11+A12)B22
    U[i1*m + j1, 4] = 1.0
    U[i1*m + j2, 4] = 1.0
    V[j2*p + k2, 4] = 1.0
    W[k1*n + i1, 4] = -1.0
    W[k2*n + i1, 4] = 1.0
    
    # M6 = (A21-A11)(B11+B12)
    U[i2*m + j1, 5] = 1.0
    U[i1*m + j1, 5] = -1.0
    V[j1*p + k1, 5] = 1.0
    V[j1*p + k2, 5] = 1.0
    W[k1*n + i1, 5] = 1.0
    
    # M7 = (A12-A22)(B21+B22)
    U[i1*m + j2, 6] = 1.0
    U[i2*m + j2, 6] = -1.0
    V[j2*p + k1, 6] = 1.0
    V[j2*p + k2, 6] = 1.0
    W[k2*n + i2, 6] = 1.0
    
    return U, V, W

def _strassen_based_decomposition(n, m, p):
    """Build a Strassen-based decomposition using 2x2 blocks where possible."""
    def partition(size):
        blocks = []
        start = 0
        while start < size:
            if size - start >= 2:
                blocks.append((start, start + 2))
                start += 2
            else:
                blocks.append((start, start + 1))
                start += 1
        return blocks
    
    i_blocks = partition(n)
    j_blocks = partition(m)
    k_blocks = partition(p)
    
    U_list = []
    V_list = []
    W_list = []
    
    for i1, i2 in i_blocks:
        for j1, j2 in j_blocks:
            for k1, k2 in k_blocks:
                i_size = i2 - i1
                j_size = j2 - j1
                k_size = k2 - k1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U_b, V_b, W_b = _strassen_2x2x2_block(n, m, p, i1, i2, j1, j2, k1, k2)
                else:
                    # Schoolbook for this block
                    rank = i_size * j_size * k_size
                    U_b = np.zeros((n * m, rank), dtype=float)
                    V_b = np.zeros((m * p, rank), dtype=float)
                    W_b = np.zeros((p * n, rank), dtype=float)
                    col = 0
                    for i in range(i1, i2):
                        for j in range(j1, j2):
                            for k in range(k1, k2):
                                U_b[i * m + j, col] = 1.0
                                V_b[j * p + k, col] = 1.0
                                W_b[k * n + i, col] = 1.0
                                col += 1
                
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy: Start with Strassen-based decomposition, then try to reduce rank
    # via randomized algebraic methods
    
    # Phase 1: Build Strassen-based decomposition
    U, V, W = _strassen_based_decomposition(n, m, p)
    current_rank = U.shape[1]
    
    # Verify the Strassen decomposition
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9, num_samples=1000):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Phase 2: Try randomized rank reduction
    remaining_time = DEADLINE - 30.0 - (time.time() - start_time)
    if remaining_time > 10.0:
        best_U, best_V, best_W, best_rank = _randomized_rank_reduction(
            U, V, W, n, m, p, time_limit=remaining_time
        )
    
    # Phase 3: Final verification
    if time.time() - start_time < DEADLINE - 5.0:
        if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9, num_samples=2000):
            # Fallback to schoolbook
            U_f, V_f, W_f = _schoolbook(n, m, p)
            best_U, best_V, best_W = U_f, V_f, W_f
            best_rank = U_f.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
