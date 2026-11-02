"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Return list of (i, j, k) triples that must equal 1."""
    n, m, p = DIMENSIONS
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    return triples

def _verify(U, V, W, rank):
    """Check exact decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = s + U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, rank):
    """Fast verification using float, then exact if close."""
    n, m, p = DIMENSIONS
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Uf[i * m + j], Vf[j * p + k] * Wf[k * n + i]))
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _structured_init(rank, seed):
    """Create a structured initial decomposition with sparse entries."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # Start from schoolbook-like and randomly zero out / merge
    # Assign each (i,j,k) to a random rank index
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = rng.randrange(rank)
                # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to contribute
                # Set all three to 1 for now; local search will fix
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _local_search(U, V, W, rank, max_iter, seed, deadline):
    """Local search: randomly pick a matrix entry and a value, accept if improves residual."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Value set: small integers and half-integers
    values = [0, 1, -1, 2, -2, 0.5, -0.5]
    
    def residual():
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    total += (s - 1.0) ** 2
        return total
    
    best_res = residual()
    no_improve = 0
    iter_count = 0
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        # Pick random matrix, row, column
        mat = rng.choice(['U', 'V', 'W'])
        if mat == 'U':
            row = rng.randrange(n * m)
        elif mat == 'V':
            row = rng.randrange(m * p)
        else:
            row = rng.randrange(p * n)
        col = rng.randrange(rank)
        old_val = U[row][col] if mat == 'U' else (V[row][col] if mat == 'V' else W[row][col])
        new_val = rng.choice(values)
        if new_val == old_val:
            continue
        # Apply
        if mat == 'U':
            U[row][col] = new_val
        elif mat == 'V':
            V[row][col] = new_val
        else:
            W[row][col] = new_val
        
        res = residual()
        if res < best_res - 1e-12:
            best_res = res
            no_improve = 0
        else:
            # Revert
            if mat == 'U':
                U[row][col] = old_val
            elif mat == 'V':
                V[row][col] = old_val
            else:
                W[row][col] = old_val
            no_improve += 1
            if no_improve > 500:
                break
        iter_count += 1
    return U, V, W, best_res

def _try_rank(rank, deadline):
    """Try to find a decomposition of given rank."""
    n, m, p = DIMENSIONS
    if time.time() > deadline:
        return None
    best_result = None
    best_res = float('inf')
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        seed = trial * 1000 + 42
        U, V, W = _structured_init(rank, seed)
        # Local search with limited iterations
        max_iter = 2000
        U2, V2, W2, res = _local_search(U, V, W, rank, max_iter, seed, deadline)
        if res < best_res:
            best_res = res
            best_result = (U2, V2, W2)
        if res < 1e-6:
            # Exact check
            if _verify(U2, V2, W2, rank):
                return {"U": U2, "V": V2, "W": W2, "rank": rank}
    # Final exact check on best
    if best_result is not None and best_res < 1e-4:
        U2, V2, W2 = best_result
        if _verify(U2, V2, W2, rank):
            return {"U": U2, "V": V2, "W": W2, "rank": rank}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    # Start from schoolbook
    school = _schoolbook()
    best = school
    # Try lower ranks
    for rank in range(18, 24):
        if time.time() > deadline:
            break
        result = _try_rank(rank, deadline)
        if result is not None:
            best = result
            break
    # If we found something better, return it; else schoolbook
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
