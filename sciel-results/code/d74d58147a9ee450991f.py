"""Matrix multiplication tensor decomposition for 3x5x6."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 x 5x6 matrix multiplication.
    Schoolbook rank is 90. We attempt to find a lower-rank exact decomposition
    using a fast vectorized search over half-integer entries, plus a structured
    Strassen-like embedding attempt.
    """
    n, m, p = DIMENSIONS
    nm = n * m      # 15
    mp = m * p      # 30
    pn = p * n      # 18
    schoolbook_rank = n * m * p  # 90

    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def schoolbook():
        """Return the schoolbook rank-90 decomposition."""
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = schoolbook()
    best_rank = schoolbook_rank

    def verify_np(U, V, W, rank):
        """
        Vectorized verification: check that for all (i,j,k),
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
        Returns True if exact (within tight tolerance).
        """
        # Reshape U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)

        # For each (i, j, k): sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Build a tensor of shape (n, m, p) by summing over c
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    def try_random(rank, max_iters, rng):
        """Random search for a valid decomposition of given rank."""
        half_ints = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            U = rng.choice(half_ints, size=(nm, rank))
            V = rng.choice(half_ints, size=(mp, rank))
            W = rng.choice(half_ints, size=(pn, rank))
            if verify_np(U, V, W, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Strategy 1: Try rank 89 with a large number of random trials
    # The probability of finding a rank-89 solution randomly is extremely low,
    # but we try with vectorized verification which is fast.
    rng = np.random.RandomState(42)
    
    # First, do a quick sanity check: verify schoolbook works
    U_sb = np.array(best["U"])
    V_sb = np.array(best["V"])
    W_sb = np.array(best["W"])
    assert verify_np(U_sb, V_sb, W_sb, 90), "Schoolbook verification failed!"

    # Try rank 89
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        max_iters_89 = int((deadline - 20 - elapsed) / 0.005)  # ~5ms per iteration estimate
        max_iters_89 = min(max_iters_89, 200000)
        max_iters_89 = max(max_iters_89, 1000)
        print(f"Trying rank 89 with up to {max_iters_89} iterations...")
        result = try_random(89, max_iters_89, rng)
        if result is not None:
            print("Found rank 89 solution!")
            best = result
            best_rank = 89
            return best

    # Strategy 2: Try a structured approach based on splitting the problem
    # For 3x5x6, consider decomposing as (3x5x6) = (3x3x6) + (3x2x6) or similar
    # Actually, let's try a more principled approach:
    # If we can find a rank-89 solution by modifying one column of schoolbook,
    # we can do a targeted search.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Targeted search: start from schoolbook and perturb
        # Try to find a linear dependency among the 90 schoolbook terms
        # that allows merging two terms into one (reducing rank by 1)
        
        # Alternative: try rank 89 with a smarter initialization
        # Use schoolbook columns but with one column being a sum/difference
        print("Trying structured rank-89 search...")
        
        # Generate random half-integer matrices but bias toward sparse
        for attempt in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Start with mostly zeros, a few non-zeros
            U = np.zeros((nm, 89))
            V = np.zeros((mp, 89))
            W = np.zeros((pn, 89))
            
            # Place ones at random positions
            num_ones = rng.randint(100, 200)
            rows_u = rng.randint(0, nm, size=num_ones)
            cols_u = rng.randint(0, 89, size=num_ones)
            U[rows_u, cols_u] = 1.0
            
            rows_v = rng.randint(0, mp, size=num_ones)
            cols_v = rng.randint(0, 89, size=num_ones)
            V[rows_v, cols_v] = 1.0
            
            rows_w = rng.randint(0, pn, size=num_ones)
            cols_w = rng.randint(0, 89, size=num_ones)
            W[rows_w, cols_w] = 1.0
            
            if verify_np(U, V, W, 89):
                print("Found structured rank 89 solution!")
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 89
                return best

    # If we found a better solution, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        return best
    
    return best
# EVOLVE-BLOCK-END
