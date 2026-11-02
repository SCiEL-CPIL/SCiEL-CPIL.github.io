"""Search-based solver for rank-minimal exact tensor decomposition of 3x5x6 matmul."""
import time
import numpy as np
import itertools
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_rank = n * m * p
    best_result = None

    # Schoolbook baseline (always valid)
    best_result = _schoolbook(n, m, p, best_rank)

    # Verify the baseline
    if _verify(n, m, p, best_result):
        pass  # baseline is valid

    # Search for better (lower rank) decompositions
    # Strategy: Use structured random search with half-integer entries
    # and verification via tensor contraction

    def try_random_rank(r, num_trials, time_budget):
        """Try to find a decomposition of rank r via random search."""
        nonlocal best_rank, best_result
        start = time.time()
        for trial in range(num_trials):
            if time.time() > time_budget or time.time() > deadline:
                break
            # Random half-integer factors with structured patterns
            U, V, W = _generate_structured_factors(n, m, p, r)
            # Check if this decomposes the tensor
            if _check_decomp_fast(n, m, p, U, V, W, r):
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if r < best_rank:
                    best_rank = r
                    best_result = result
                return True
        return False

    def _generate_structured_factors(n, m, p, r):
        """Generate structured random factors with half-integer entries."""
        # Use a mix of random and structured patterns
        scale = 1.0
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Generate random half-integer values
        vals = [0, 0.5, 1, -0.5, -1, 0.0, 2.0, -2.0, 0.25, -0.25, 1.5, -1.5]
        
        for c in range(r):
            # Each column c represents one term in the decomposition
            # U[:, c] has entries for (i, j) pairs
            # V[:, c] has entries for (j, k) pairs
            # W[:, c] has entries for (k, i) pairs
            
            # Strategy: make each term "sparse" with a few non-zero entries
            # This mimics the structure of known efficient algorithms
            
            # For U: pick a few (i,j) pairs to be non-zero
            num_nonzero_u = random.randint(1, min(4, n*m))
            indices_u = random.sample(range(n*m), num_nonzero_u)
            for idx in indices_u:
                U[idx, c] = random.choice(vals)
            
            # For V: pick a few (j,k) pairs
            num_nonzero_v = random.randint(1, min(4, m*p))
            indices_v = random.sample(range(m*p), num_nonzero_v)
            for idx in indices_v:
                V[idx, c] = random.choice(vals)
            
            # For W: pick a few (k,i) pairs
            num_nonzero_w = random.randint(1, min(4, p*n))
            indices_w = random.sample(range(p*n), num_nonzero_w)
            for idx in indices_w:
                W[idx, c] = random.choice(vals)
        
        return U, V, W

    def _check_decomp_fast(n, m, p, U, V, W, r):
        """Check if U,V,W of rank r exactly decompose the matmul tensor using numpy."""
        # For all (i,j,k): sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] == 1
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks from a lower bound upward
    # For 3x5x6: schoolbook is 90
    # Try to find something better
    
    # Prioritize ranks that are likely to work
    # Based on known results, try a range
    
    # First, try some structured approaches for lower ranks
    # Then fall back to random search
    
    # Try ranks in a smart order
    # Lower bound: at least max(n, m, p) = 6, but realistically much higher
    # Try from 30 to 89 (below schoolbook)
    
    ranks_to_try = list(range(30, 90))
    random.shuffle(ranks_to_try)
    # But prioritize lower ranks first
    ranks_to_try.sort()
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue  # no point trying ranks >= current best
        # Try a few random trials for this rank
        num_trials = max(100, int((deadline - time.time()) * 10))
        num_trials = min(num_trials, 500)
        try_random_rank(r, num_trials, deadline - 5)
        if best_rank <= r:
            break  # found one at or below this rank, stop
    
    # Also try a direct construction for specific known patterns
    # For example, try to exploit the fact that n=3 is small
    
    # Return best
    return best_result

def _schoolbook(n, m, p, rank):
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
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, result):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
