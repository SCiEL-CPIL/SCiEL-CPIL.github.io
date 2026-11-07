"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from sympy import isprime, factorint

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_parity_vectors(lo, hi, primes):
    """For each integer in [lo, hi], compute parity vector of prime exponents."""
    nprimes = len(primes)
    vectors = {}
    for m in range(lo, hi + 1):
        vec = [0] * nprimes
        for i, p in enumerate(primes):
            if m % p == 0:
                cnt = 0
                while m % p == 0:
                    m //= p
                    cnt += 1
                vec[i] = cnt % 2
        # Check if remaining m > 1 (has large prime factor)
        # We track this separately - if remaining > 1, it contributes to "other" parity
        if m > 1:
            vec.append(1)  # has odd exponent for some prime not in our list
        else:
            vec.append(0)
        vectors[210 + (210 - lo) + (m - lo)] = vec  # placeholder, fix below
    return vectors

def _build_parity_vectors_fixed(lo, hi, primes):
    """For each integer in [lo, hi], compute parity vector of prime exponents."""
    nprimes = len(primes)
    vectors = {}
    for m in range(lo, hi + 1):
        vec = [0] * nprimes
        remaining = m
        for i, p in enumerate(primes):
            if remaining % p == 0:
                cnt = 0
                while remaining % p == 0:
                    remaining //= p
                    cnt += 1
                vec[i] = cnt % 2
        # If remaining > 1, it has odd exponent for at least one prime not in list
        # But we need to be careful: remaining could be p^2 for large p, which is even
        # For simplicity, if remaining > 1, we mark it as having an "other" odd factor
        # This is conservative - it may miss some solutions but won't give false positives
        if remaining > 1:
            vec.append(1)
        else:
            vec.append(0)
        vectors[m] = vec
    return vectors

def _gaussian_elimination(mat, target, ncols):
    """
    Solve mat * x = target over GF(2).
    mat is list of lists (rows), each row is a list of 0/1 of length ncols.
    Returns a solution x (list of 0/1) or None if no solution.
    We want to find x such that sum(x_i * col_i) = target mod 2.
    Here mat[i] is the parity vector for column i.
    """
    nrows = len(mat)
    if nrows == 0:
        return None
    
    # Build augmented matrix: each row is a parity vector + target
    # We're solving: for each bit position j, sum_i (mat[i][j] * x_i) = target[j]
    # This is a system where columns of mat are the vectors, and we want linear combination = target
    
    # Actually, let me reformulate. We have vectors v_1, ..., v_n in GF(2)^d.
    # We want x_1, ..., x_n in {0,1} such that sum x_i * v_i = target.
    # This is equivalent to: for each coordinate j: sum_i x_i * v_i[j] = target[j]
    # So the matrix has d rows and n columns, and we solve M*x = target.
    
    d = len(target)
    # M is d x n: M[j][i] = mat[i][j]
    M = [[mat[i][j] for i in range(nrows)] for j in range(d)]
    
    # Gaussian elimination on M * x = target
    # Augmented matrix: d rows, n+1 columns
    aug = [row[:] + [target[j]] for j, row in enumerate(M)]
    
    rank = 0
    for col in range(nrows):
        # Find pivot row
        pivot = -1
        for row in range(rank, d):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        # Eliminate
        for row in range(d):
            if row != rank and aug[row][col] == 1:
                for c in range(col, nrows + 1):
                    aug[row][c] ^= aug[rank][c]
        rank += 1
    
    # Check consistency
    for row in range(rank, d):
        if aug[row][nrows] == 1:
            return None  # Inconsistent
    
    # Find a solution (set free variables to 0)
    x = [0] * nrows
    for i in range(rank):
        # Find the column where this row has its pivot
        for col in range(nrows):
            if aug[i][col] == 1:
                x[col] = aug[i][nrows]
                break
    return x

def _find_solution_for_max(M, max_size=100):
    """
    Try to find up to max_size distinct integers in (210, M] whose product
    has the required parity to make product*210 a perfect square.
    Returns list of integers or None.
    """
    if M <= 210:
        return None
    
    # Primes to consider: 2, 3, 5, 7 (from 210) plus small primes up to ~M
    # For the parity to work, we need:
    # - For primes 2,3,5,7: product must have odd exponent (to complement 210's odd exponents)
    # - For all other primes: product must have even exponent
    # 
    # Target parity vector for primes [2,3,5,7]: [1,1,1,1]
    # For "other" primes: 0 (even)
    
    # Use primes up to a reasonable bound. If a number has a prime factor > bound,
    # we need that factor to appear an even number of times total.
    # For simplicity, use primes up to sqrt(M) or so, and track large factors separately.
    
    # Actually, let's use a simpler approach: just use primes 2,3,5,7 and track
    # whether the number has any other prime factor with odd exponent.
    # Target: [1,1,1,1,0] where last is "has other odd prime factor"
    
    primes = [2, 3, 5, 7]
    target = [1, 1, 1, 1, 0]
    
    lo = 211
    hi = M
    vectors = _build_parity_vectors_fixed(lo, hi, primes)
    
    if not vectors:
        return None
    
    # Get all vectors
    keys = sorted(vectors.keys())
    mat = [vectors[k] for k in keys]
    nrows = len(mat)
    
    if nrows == 0:
        return None
    
    # Try to find a solution with at most max_size elements
    # First, find any solution
    x = _gaussian_elimination(mat, target, nrows)
    if x is None:
        return None
    
    # Count selected elements
    selected = [keys[i] for i in range(nrows) if x[i] == 1]
    
    if len(selected) > max_size:
        # Try to find a smaller solution by zeroing out some free variables
        # This is complex; for now, if we have a solution, use it
        # But we need at most 100, so if we have more, we need to reduce
        # For now, return None if too many
        return None
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with safety margin
    
    # Binary search on the maximum value
    # Lower bound: 211 (smallest possible)
    # Upper bound: 10210
    
    lo_bound = 211
    hi_bound = 10210
    
    best_solution = None
    
    # First, try to find any solution with a moderate maximum
    # Then binary search to minimize
    
    # Start with a reasonable guess
    # The seed gives 4*210 = 840, so max is 840, raw value is 630
    # We want to do much better
    
    # Binary search
    lo = lo_bound
    hi = hi_bound
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        solution = _find_solution_for_max(mid)
        
        if solution is not None:
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is None:
        # Fallback: use the seed solution
        return [4 * 210]
    
    # Verify the solution
    product = 210
    for x in best_solution:
        product *= x
    
    if not _is_perfect_square(product):
        # Fallback
        return [4 * 210]
    
    # Ensure all are Python ints and in a list
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
