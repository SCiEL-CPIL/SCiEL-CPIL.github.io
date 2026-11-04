import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_structured_init(n, m, p, rank, seed=0):
    """Build a structured initialization using cyclic group patterns."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use a cyclic pattern: for each column r, assign non-zero entries
    # based on modular arithmetic to create structure
    for r in range(rank):
        # Choose a subset of positions to activate
        # Use a pattern based on r mod some period
        period = max(1, rank // 3)
        
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # Activate if (i + j + r) % period < period // 2
                if (i + j + r) % period < period // 2:
                    U[a, r] = rng.randn() * 0.5
        
        for j in range(m):
            for k in range(p):
                b = j * p + k
                if (j + k + r) % period < period // 2:
                    V[b, r] = rng.randn() * 0.5
        
        for k in range(p):
            for i in range(n):
                g = k * n + i
                if (k + i + r) % period < period // 2:
                    W[g, r] = rng.randn() * 0.5
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check all 1-positions exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _round_to_rational(M, denom=4):
    """Round matrix entries to nearest 1/denom."""
    return np.round(M * denom) / denom

def _compute_error_fast(U, V, W, n, m, p, zero_sample_size=300):
    """Compute error efficiently using vectorized operations."""
    rank = U.shape[1]
    
    # Compute all 1-positions
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; g = k*n+i
                val = np.dot(U[a, :], V[b, :] * W[g, :])
                error += (val - 1.0) ** 2
    
    # Sample zero constraints
    np.random.seed(42)
    for _ in range(zero_sample_size):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        g = np.random.randint(0, p*n)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            error += val ** 2
    
    return error

def _polish_decomposition(U, V, W, n, m, p, max_time=15):
    """Final high-precision polish pass with tighter tolerance."""
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    
    # Pre-compute index mappings
    a_to_bg = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = [(j * p + k, k * n + i) for k in range(p)]
            a_to_bg[a] = pairs

    b_to_ag = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = [(i * m + j, k * n + i) for i in range(n)]
            b_to_ag[b] = pairs

    g_to_ab = {}
    for k in range(p):
        for i in range(n):
            g = k * n + i
            pairs = [(i * m + j, j * p + k) for j in range(m)]
            g_to_ab[g] = pairs

    # Run polish iterations with decreasing regularization
    for polish_iter in range(20):
        if time.time() > deadline:
            break
        
        reg = 1e-8 * (0.5 ** polish_iter)
        
        # Update U
        G_V = V.T @ V
        G_W = W.T @ W
        A_mat = G_V * G_W + reg * np.eye(rank)
        
        for a in range(n * m):
            b_vec = np.zeros(rank)
            for b, g in a_to_bg[a]:
                b_vec += V[b, :] * W[g, :]
            try:
                U[a, :] = np.linalg.solve(A_mat, b_vec)
            except:
                pass

        # Update V
        G_U = U.T @ U
        A_mat = G_U * G_W + reg * np.eye(rank)
        
        for b in range(m * p):
            b_vec = np.zeros(rank)
            for a, g in b_to_ag[b]:
                b_vec += U[a, :] * W[g, :]
            try:
                V[b, :] = np.linalg.solve(A_mat, b_vec)
            except:
                pass

        # Update W
        G_V = V.T @ V
        A_mat = G_U * G_V + reg * np.eye(rank)
        
        for g in range(p * n):
            b_vec = np.zeros(rank)
            for a, b in g_to_ab[g]:
                b_vec += U[a, :] * V[b, :]
            try:
                W[g, :] = np.linalg.solve(A_mat, b_vec)
            except:
                pass
        
        # Check if we can round
        if polish_iter % 4 == 3:
            error = _compute_error_fast(U, V, W, n, m, p, zero_sample_size=500)
            if error < 1e-8:
                # Try different rounding schemes
                for denom in [2, 4]:
                    U_rounded = np.round(U * denom) / denom
                    V_rounded = np.round(V * denom) / denom
                    W_rounded = np.round(W * denom) / denom
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        return U_rounded, V_rounded, W_rounded
    
    return None

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=100):
    """
    Fast Alternating Least Squares using pre-computed Gram matrices.
    Uses structured initializations and efficient error computation.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Pre-compute index mappings once
    a_to_bg = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = [(j * p + k, k * n + i) for k in range(p)]
            a_to_bg[a] = pairs

    b_to_ag = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = [(i * m + j, k * n + i) for i in range(n)]
            b_to_ag[b] = pairs

    g_to_ab = {}
    for k in range(p):
        for i in range(n):
            g = k * n + i
            pairs = [(i * m + j, j * p + k) for j in range(m)]
            g_to_ab[g] = pairs

    # Try a few promising ranks close to schoolbook
    # Focus on ranks that might be achievable
    ranks_to_try = []
    for r in range(initial_rank - 1, max(50, initial_rank - 30), -1):
        ranks_to_try.append(r)
        if len(ranks_to_try) >= 15:
            break

    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try structured and random initializations
        for init_type in range(4):
            if time.time() > deadline:
                break
            
            if init_type == 0:
                # Structured cyclic initialization
                U, V, W = _build_structured_init(n, m, p, target_rank, seed=target_rank % 100)
            elif init_type == 1:
                # Sparse random
                rng = np.random.RandomState(target_rank)
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Random subset of rows
                    u_rows = rng.choice(n * m, size=max(1, n*m//4), replace=False)
                    v_rows = rng.choice(m * p, size=max(1, m*p//4), replace=False)
                    w_rows = rng.choice(p * n, size=max(1, p*n//4), replace=False)
                    U[u_rows, r] = rng.randn(len(u_rows)) * 0.5
                    V[v_rows, r] = rng.randn(len(v_rows)) * 0.5
                    W[w_rows, r] = rng.randn(len(w_rows)) * 0.5
            else:
                # Random with scaling
                np.random.seed(init_type * 1000 + target_rank)
                scale = 0.3 / np.sqrt(target_rank)
                U = np.random.randn(n * m, target_rank) * scale
                V = np.random.randn(m * p, target_rank) * scale
                W = np.random.randn(p * n, target_rank) * scale
            
            # Alternating least squares with early termination
            prev_error = float('inf')
            stuck_count = 0
            for iteration in range(60):
                if time.time() > deadline:
                    break
                
                # Update U
                G_V = V.T @ V
                G_W = W.T @ W
                A_mat = G_V * G_W + 1e-6 * np.eye(target_rank)
                
                for a in range(n * m):
                    b_vec = np.zeros(target_rank)
                    for b, g in a_to_bg[a]:
                        b_vec += V[b, :] * W[g, :]
                    try:
                        U[a, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update V
                G_U = U.T @ U
                A_mat = G_U * G_W + 1e-6 * np.eye(target_rank)
                
                for b in range(m * p):
                    b_vec = np.zeros(target_rank)
                    for a, g in b_to_ag[b]:
                        b_vec += U[a, :] * W[g, :]
                    try:
                        V[b, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update W
                A_mat = G_U * G_V + 1e-6 * np.eye(target_rank)
                
                for g in range(p * n):
                    b_vec = np.zeros(target_rank)
                    for a, b in g_to_ab[g]:
                        b_vec += U[a, :] * V[b, :]
                    try:
                        W[g, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check convergence every 3 iterations
                if iteration % 3 == 2:
                    error = _compute_error_fast(U, V, W, n, m, p, zero_sample_size=200)
                    
                    if error < 1e-6:
                        # Try rounding
                        for denom in [2, 4]:
                            U_rounded = np.round(U * denom) / denom
                            V_rounded = np.round(V * denom) / denom
                            W_rounded = np.round(W * denom) / denom
                            
                            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                                return U_rounded, V_rounded, W_rounded
                    
                    # Early stopping if stuck
                    if abs(prev_error - error) < 1e-12:
                        stuck_count += 1
                        if stuck_count >= 3:
                            break
                    else:
                        stuck_count = 0
                    prev_error = error
            
            # Try polish if we made progress
            if prev_error < 1.0 and time.time() < deadline - 10:
                polished = _polish_decomposition(U, V, W, n, m, p, max_time=10)
                if polished is not None:
                    return polished
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find a better decomposition with time budget
    remaining = max(0, deadline - time.time())
    if remaining > 5:
        result = _optimize_decomposition_fast(n, m, p, initial_rank=best_rank, max_time=remaining - 5)
        
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
