"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dict of (i,j,k) -> 1."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _eval_decomposition(U, V, W, n, m, p, T):
    """Check if decomposition exactly matches tensor T. Returns True if exact."""
    rank = len(U)
    # For each (i,j,k) in T, check sum_c U[i,c]*V[j,c]*W[k,c] == 1
    # For each (i,j,k) not in T, check sum == 0
    # Use dict for efficiency
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute all (i,j,k) values
    # T has n*m*p = 105 entries
    for (a, b, c) in T:
        val = 0.0
        for col in range(rank):
            val += U[a, col] * V[b, col] * W[c, col]
        if abs(val - 1) > 1e-9:
            return False
    # Check zeros: for each (a,b,c) not in T, the value must be 0
    # There are (n*m)*(m*p)*(p*n) - n*m*p = 15*35*21 - 105 = 11025 - 105 = 10920 zeros
    # This is expensive, so we'll check a sample or use a smarter method
    # Actually, let's check all to be safe, but this is 10920 * rank operations
    # With rank ~100, that's ~1 million ops, should be fast enough
    T_set = set(T.keys())
    all_a = list(range(n * m))
    all_b = list(range(m * p))
    all_c = list(range(p * n))
    for a in all_a:
        for b in all_b:
            for c in all_c:
                if (a, b, c) in T_set:
                    continue
                val = 0.0
                for col in range(rank):
                    val += U[a, col] * V[b, col] * W[c, col]
                if abs(val) > 1e-9:
                    return False
    return True

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    T = _build_tensor(n, m, p)
    return _eval_decomposition(U, V, W, n, m, p, T)

def _construct_schoolbook():
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
    return {"U": U, "V": V, "W": W}

def _random_search(deadline, best_rank):
    """
    Randomized search: start from schoolbook, try random rank-1 term modifications
    to reduce rank while maintaining exactness.
    
    Strategy: represent the decomposition as a list of (u, v, w) rank-1 terms.
    Each term is a triple of vectors. Try operations:
    1. Remove a term and check if remaining terms still span (unlikely to work directly)
    2. Replace two terms with one (merge) - check if possible
    3. Add a new random term and remove an existing one
    
    Actually, a better approach: use the structure of the multiplication tensor.
    The tensor T[i*m+j, j*p+k, k*n+i] = 1.
    
    A rank-1 decomposition term (u, v, w) contributes u[a]*v[b]*w[c] to T[a,b,c].
    For the multiplication tensor, the "natural" rank-1 terms are the 105 schoolbook terms.
    
    To reduce rank, we need to find linear combinations. This is equivalent to finding
    a lower-rank decomposition of the tensor.
    
    Let's use a different approach: formulate as a linear algebra problem.
    Fix U and V, then W is determined (if the system is consistent).
    
    Actually, let's use a greedy approach:
    1. Start with schoolbook (105 terms).
    2. Try to eliminate one term at a time by adjusting others.
    
    For a more effective search, let's use the following:
    The tensor has a specific structure. Let's try to find a decomposition
    with rank < 105 by using the fact that some entries can be "shared".
    
    Let me try a direct approach: use numpy to solve for a rank-r decomposition
    by fixing r and trying to find U, V, W that work.
    """
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_set = set(T.keys())
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank by removing columns
    # For each column, check if it can be removed (i.e., if the remaining columns still span)
    # This is a combinatorial problem, so use random orderings
    
    current_rank = best_rank
    U = [row[:] for row in best_U]
    V = [row[:] for row in best_V]
    W = [row[:] for row in best_W]
    
    # Try to remove columns one by one in random order
    order = list(range(current_rank))
    random.shuffle(order)
    
    for col_idx in order:
        if time.time() > deadline:
            break
        if col_idx >= current_rank:
            continue
        # Try removing column col_idx
        new_U = [row[:col_idx] + row[col_idx+1:] for row in U]
        new_V = [row[:col_idx] + row[col_idx+1:] for row in V]
        new_W = [row[:col_idx] + row[col_idx+1:] for row in W]
        if _eval_decomposition(new_U, new_V, new_W, n, m, p, T):
            U = new_U
            V = new_V
            W = new_W
            current_rank -= 1
            # Re-shuffle remaining
            order = list(range(current_rank))
            random.shuffle(order)
    
    if current_rank < best_rank:
        best_U = U
        best_V = V
        best_W = W
        best_rank = current_rank
    
    return {"U": best_U, "V": best_V, "W": best_W, "rank": best_rank}

def _strassen_block_decomp(n, m, p):
    """
    Try to use a block-Strassen approach.
    Pad to powers of 2 and use Strassen recursively.
    
    For 3x5 * 5x7:
    Pad A to 4x6, B to 6x8.
    But this gives rank 7 * (2*3*4) = 168 at base case.
    
    Alternative: don't pad, use the actual dimensions.
    Split A (3x5) into top 2 rows (2x5) and bottom 1 row (1x5).
    Split B (5x7) into left 2 cols... no, B is 5x7, we split rows.
    Split B into top 2 rows (2x7) and bottom 3 rows (3x7).
    
    This doesn't align for Strassen.
    
    Let's try: pad A to 4x5 (add 1 zero row), B to 5x8 (add 1 zero col).
    Then A is 4x5, B is 5x8, C is 4x8.
    Split A into 2x2 blocks of 2x2.5... doesn't work cleanly.
    
    Let's try a different approach: use the known result that for m=5,
    we can decompose the 5-dimensional middle space.
    
    Actually, let's just try to find a decomposition with rank < 105
    using a more systematic search.
    """
    return None

def _greedy_merge_search(deadline, initial_rank):
    """
    Greedy search: try to find pairs of rank-1 terms that can be merged
    into a single term (or a pair that can be replaced by a single term).
    
    Two rank-1 terms u1⊗v1⊗w1 and u2⊗v2⊗w2 can be "merged" if their sum
    is itself a rank-1 tensor, i.e., if (u1⊗v1⊗w1 + u2⊗v2⊗w2) has rank 1.
    This happens when u1, u2 are linearly dependent AND v1, v2 are linearly dependent
    AND w1, w2 are linearly dependent.
    
    In our case, the schoolbook terms have u, v, w as unit vectors, so no two
    can be merged this way.
    
    Alternative: try to find a set of k terms that can be replaced by k-1 terms.
    This is the tensor rank minimization problem, which is NP-hard.
    
    Let's use a different approach: fix the structure of U, V, W to have
    a specific pattern and solve for the coefficients.
    """
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_set = set(T.keys())
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a rank-r decomposition by solving a linear system.
    # For a given r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all (a,b,c) in T: sum_k U[a,k]*V[b,k]*W[c,k] = 1
    # and for all (a,b,c) not in T: sum_k U[a,k]*V[b,k]*W[c,k] = 0
    
    # This is a system of bilinear equations, which is hard.
    # Let's try a heuristic: fix U and V randomly, then solve for W.
    
    best_result = None
    best_rank = initial_rank
    
    # Start from schoolbook and try small perturbations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Create a random rank-r decomposition attempt
        r = initial_rank - random.randint(1, 5)
        if r < 1:
            continue
        
        # Initialize U, V, W with random integer values in {-1, 0, 1}
        # with the constraint that the schoolbook terms are "close"
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        W = np.zeros((pn, r), dtype=float)
        
        # Initialize with schoolbook-like structure
        for col in range(r):
            # Pick a random (i,j,k) to be the "primary" term for this column
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            U[i*m+j, col] = random.choice([-1, 1])
            V[j*p+k, col] = random.choice([-1, 1])
            W[k*n+i, col] = random.choice([-1, 1])
        
        # Now try to adjust to make it work
        # This is a non-linear problem, so use a simple iterative approach
        # For now, just check if this random guess works
        if _eval_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, T):
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": r}
            break
    
    if best_result is None:
        # Fall back to schoolbook
        return _construct_schoolbook()
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook()
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Verify schoolbook
    if not _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        return schoolbook
    
    best_rank = n * m * p  # 105
    best_result = schoolbook
    
    # Try greedy column removal
    if time.time() < deadline:
        result = _random_search(deadline, best_rank)
        if result["rank"] < best_rank:
            best_rank = result["rank"]
            best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try Strassen-based approach if time permits
    if time.time() < deadline:
        # Try padding to 4x6 and 6x8 with Strassen
        # Actually, let's try a direct approach:
        # For the 3x5 * 5x7 multiplication, try to find a rank-99 decomposition
        # by using the structure of the tensor.
        
        # One approach: use the fact that the tensor can be written as
        # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        # and try to find linear dependencies.
        
        # Let's try a more targeted search: fix the first n*m*p - r terms
        # and solve for the remaining r terms.
        pass
    
    return best_result
# EVOLVE-BLOCK-END
