import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _prime_factors(n):
    """Return dict mapping prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _exponent_vector(n, primes):
    """Return tuple of exponents for n over the given primes."""
    factors = _prime_factors(n)
    vec = [0] * len(primes)
    for i, p in enumerate(primes):
        vec[i] = factors.get(p, 0)
    return tuple(vec)

def _gaussian_elimination(vectors):
    """
    Find maximum linearly independent subset using Gaussian elimination.
    Returns the indices of the independent vectors.
    """
    n = len(vectors)
    if n == 0:
        return []
    
    m = len(vectors[0]) if vectors else 0
    
    # Work with rational numbers for exact arithmetic
    # Use a list of (row_vector, original_index)
    rows = [(list(v), i) for i, v in enumerate(vectors)]
    
    independent = []
    pivot_cols = []
    
    for col in range(m):
        # Find a row with non-zero entry in this column
        pivot_row = -1
        for r in range(len(rows)):
            if rows[r][0][col] != 0:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap to top
        rows[0], rows[pivot_row] = rows[pivot_row], rows[0]
        
        # Eliminate this column from other rows
        for r in range(1, len(rows)):
            factor = rows[r][0][col] / rows[0][0][col]
            for c in range(col, m):
                rows[r][0][c] -= factor * rows[0][0][c]
        
        independent.append(rows[0][1])
        pivot_cols.append(col)
        
        # Remove pivot row
        rows = rows[1:]
        
        if not rows:
            break
    
    return independent

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    limit = 40
    primes = [n for n in range(2, limit + 1) if _is_prime(n)]
    
    # Generate all candidate numbers 2..40
    candidates = list(range(2, limit + 1))
    
    # Convert each candidate to its exponent vector
    vectors = []
    for num in candidates:
        vec = _exponent_vector(num, primes)
        vectors.append((num, vec))
    
    # Use Gaussian elimination to find maximum independent set
    # We process vectors in a strategic order to maximize the set size
    
    # Strategy: Try different orderings and keep the best result
    best_result = []
    best_size = 0
    
    # Ordering 1: By value
    ordered = sorted(vectors, key=lambda x: x[0])
    vecs_only = [v[1] for v in ordered]
    idxs = _gaussian_elimination(vecs_only)
    result = [ordered[i][0] for i in idxs]
    if len(result) > best_size:
        best_size = len(result)
        best_result = result
    
    # Ordering 2: By number of prime factors (sparser first)
    def _num_factors(vec):
        return sum(1 for x in vec if x != 0)
    
    ordered2 = sorted(vectors, key=lambda x: (_num_factors(x[1]), x[0]))
    vecs_only2 = [v[1] for v in ordered2]
    idxs2 = _gaussian_elimination(vecs_only2)
    result2 = [ordered2[i][0] for i in idxs2]
    if len(result2) > best_size:
        best_size = len(result2)
        best_result = result2
    
    # Ordering 3: Primes first, then composites
    def _sort_key3(x):
        num, vec = x
        if _is_prime(num):
            return (0, num)
        else:
            return (1, _num_factors(vec), num)
    
    ordered3 = sorted(vectors, key=_sort_key3)
    vecs_only3 = [v[1] for v in ordered3]
    idxs3 = _gaussian_elimination(vecs_only3)
    result3 = [ordered3[i][0] for i in idxs3]
    if len(result3) > best_size:
        best_size = len(result3)
        best_result = result3
    
    # Verify the best result
    def _check_distinct_products(nums):
        products = set()
        k = len(nums)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    if not _check_distinct_products(best_result):
        # Fallback: just use primes
        best_result = primes
    
    # Ensure types are Python int
    best_result = [int(x) for x in best_result]
    best_result.sort()
    
    return best_result

# EVOLVE-BLOCK-END
