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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check all 1's
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        g = rng.randint(0, p*n)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val) > tol:
                return False
    return True

def _compute_loss(U, V, W, n, m, p, zero_samples=500):
    """Compute the total loss efficiently."""
    rank = U.shape[1]
    # 1-constraints
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                loss += (val - 1.0) ** 2
    # Zero constraints (sampled)
    rng = np.random.RandomState(42)
    for _ in range(zero_samples):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        g = rng.randint(0, p*n)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            loss += val ** 2
    return loss

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=140):
    """
    Fast ALS optimization exploiting sparse tensor structure.
    Key insight: for each (a,b,g), the constraint is only active when indices match.
    We precompute which (b,g) pairs are consistent with each a, and vice versa.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Precompute consistency mappings
    # For each a (i*m+j), which (b,g) are consistent?
    # b = j*p+k, g = k*n+i for k in range(p)
    a_to_bg = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = []
            for k in range(p):
                b = j * p + k
                g = k * n + i
                pairs.append((b, g))
            a_to_bg[a] = pairs
    
    # For each b (j*p+k), which (a,g) are consistent?
    # a = i*m+j, g = k*n+i for i in range(n)
    b_to_ag = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = []
            for i in range(n):
                a = i * m + j
                g = k * n + i
                pairs.append((a, g))
            b_to_ag[b] = pairs
    
    # For each g (k*n+i), which (a,b) are consistent?
    # a = i*m+j, b = j*p+k for j in range(m)
    g_to_ab = {}
    for k in range(p):
        for i in range(n):
            g = k * n + i
            pairs = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                pairs.append((a, b))
            g_to_ab[g] = pairs
    
    best_result = None
    
    # Try decreasing ranks from initial_rank-1 down
    for target_rank in range(initial_rank - 1, max(20, initial_rank - 60), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for seed in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            
            U = np.random.randn(n * m, target_rank) * 0.05
            V = np.random.randn(m * p, target_rank) * 0.05
            W = np.random.randn(p * n, target_rank) * 0.05
            
            # ALS iterations
            for iteration in range(30):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve least squares over consistent (b,g) pairs
                for a in range(n * m):
                    pairs = a_to_bg[a]
                    # Design matrix: each row is V[b,:] * W[g,:]
                    # Target: 1.0 for each pair
                    if len(pairs) == 0:
                        continue
                    M_mat = np.array([V[b, :] * W[g, :] for b, g in pairs])
                    t_vec = np.ones(len(pairs))
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V: for each b, solve least squares over consistent (a,g) pairs
                for b in range(m * p):
                    pairs = b_to_ag[b]
                    if len(pairs) == 0:
                        continue
                    M_mat = np.array([U[a, :] * W[g, :] for a, g in pairs])
                    t_vec = np.ones(len(pairs))
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W: for each g, solve least squares over consistent (a,b) pairs
                for g in range(p * n):
                    pairs = g_to_ab[g]
                    if len(pairs) == 0:
                        continue
                    M_mat = np.array([U[a, :] * V[b, :] for a, b in pairs])
                    t_vec = np.ones(len(pairs))
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        W[g, :] = sol
                    except:
                        pass
                
                # Check convergence every 5 iterations
                if iteration % 5 == 4:
                    loss = _compute_loss(U, V, W, n, m, p, zero_samples=300)
                    if loss < 1e-12:
                        if _verify_decomposition(U, V, W, n, m, p):
                            best_result = (U, V, W)
                            return best_result
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find a better decomposition
    result = _optimize_decomposition_fast(n, m, p, initial_rank=best_rank, max_time=140)
    
    if result is not None:
        U, V, W = result
        if U.shape[1] < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    # Final safety check
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
