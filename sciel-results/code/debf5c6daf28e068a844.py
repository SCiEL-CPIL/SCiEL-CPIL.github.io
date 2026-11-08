"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(result, ntest=3, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rng = random.Random(seed)
    for t in range(ntest):
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        Ctrue = [[sum(A[i][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)]
                 for i in range(n)]
        Ccomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(len(U[0])):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-9 * max(1, abs(Ctrue[i][k])):
                    return False
    return True

def _compute_residual(U, V, W, rank, n, m, p):
    """Compute max residual of the tensor decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_strassen_like_block():
    """Try to construct a decomposition using Strassen-like block structure.
    
    For 3x4x8, we can try to split the 8 columns into two groups of 4.
    C[:, 0:4] = A @ B[:, 0:4] and C[:, 4:8] = A @ B[:, 4:8]
    
    If we can find a rank-r decomposition for 3x4x4, then total rank would be 2r.
    Schoolbook for 3x4x4 is 48, so 2*48=96 (no improvement).
    
    But if we can find a better 3x4x4 tensor, we win.
    """
    n, m, p = DIMENSIONS
    # This is a placeholder - we don't have a known better 3x4x4 tensor
    return None

def _random_structured_init(rank, seed, n, m, p):
    """Initialize with structured sparsity based on block patterns."""
    rng = random.Random(seed)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Try a pattern where each rank-1 term covers a subset of (i,j,k)
    # with some sharing
    for r in range(rank):
        # Each term picks a random subset of i, j, k indices
        # with some probability of sharing
        if rng.random() < 0.7:
            # Standard term: single (i,j,k)
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j][r] = 1.0
            V[j*p+k][r] = 1.0
            W[k*n+i][r] = 1.0
        else:
            # Try a "shared" term that might cover multiple (i,j,k)
            # This is heuristic - won't usually work but worth trying
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k1 = rng.randint(0, p-1)
            k2 = rng.randint(0, p-1)
            if k1 != k2:
                U[i*m+j][r] = 1.0
                V[j*p+k1][r] = 1.0
                V[j*p+k2][r] = 1.0
                W[k1*n+i][r] = 1.0
                W[k2*n+i][r] = 1.0
    return U, V, W

def _local_search(U, V, W, rank, seed, max_iters=200, n=3, m=4, p=8):
    """Local hill-climbing to reduce residual."""
    rng = random.Random(seed)
    current_err = _compute_residual(U, V, W, rank, n, m, p)
    
    for iteration in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        improved = False
        # Try random single-entry modifications
        for _ in range(30):
            mat_idx = rng.choice(['U', 'V', 'W'])
            if mat_idx == 'U':
                row = rng.randint(0, n*m-1)
                col = rng.randint(0, rank-1)
                old_val = U[row][col]
                delta = rng.choice([-1.0, 1.0, 0.5, -0.5])
                U[row][col] = old_val + delta
            elif mat_idx == 'V':
                row = rng.randint(0, m*p-1)
                col = rng.randint(0, rank-1)
                old_val = V[row][col]
                delta = rng.choice([-1.0, 1.0, 0.5, -0.5])
                V[row][col] = old_val + delta
            else:
                row = rng.randint(0, p*n-1)
                col = rng.randint(0, rank-1)
                old_val = W[row][col]
                delta = rng.choice([-1.0, 1.0, 0.5, -0.5])
                W[row][col] = old_val + delta
            
            new_err = _compute_residual(U, V, W, rank, n, m, p)
            if new_err < current_err - 1e-12:
                current_err = new_err
                improved = True
            else:
                # Revert
                if mat_idx == 'U':
                    U[row][col] = old_val
                elif mat_idx == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
        
        if not improved:
            # Try a more aggressive move: reset a random column
            col = rng.randint(0, rank-1)
            for i in range(n*m):
                U[i][col] = 0.0
            for i in range(m*p):
                V[i][col] = 0.0
            for i in range(p*n):
                W[i][col] = 0.0
            # Initialize this column as a schoolbook term
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j][col] = 1.0
            V[j*p+k][col] = 1.0
            W[k*n+i][col] = 1.0
            current_err = _compute_residual(U, V, W, rank, n, m, p)
    
    return current_err

def _try_rank_reduction(start_rank, end_rank, n, m, p):
    """Try to find a valid decomposition with rank in [end_rank, start_rank)."""
    best_result = None
    best_rank = start_rank
    
    for rank in range(end_rank, start_rank):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            U, V, W = _random_structured_init(rank, rank * 1000 + seed, n, m, p)
            err = _local_search(U, V, W, rank, seed * 10000 + rank, max_iters=100, n=n, m=m, p=p)
            
            if err < 1e-9:
                result = {"U": U, "V": V, "W": W}
                if _validate_decomposition(result, ntest=3, seed=rank + seed):
                    best_result = result
                    best_rank = rank
                    break
        
        if best_result and best_rank == rank:
            break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    best = _build_schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to reduce rank from schoolbook
    # Start from a reasonable lower bound
    # Theoretical lower bound for matrix mult is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But in practice, for these small dimensions, we're likely stuck near schoolbook
    
    # Try a few rank values below schoolbook
    target_ranks = [80, 75, 70, 65, 60]
    
    for target in target_ranks:
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        if target >= best_rank:
            continue
        
        result, rank = _try_rank_reduction(best_rank, target, n, m, p)
        if result:
            best = result
            best_rank = rank
            if best_rank <= 60:
                break
    
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
