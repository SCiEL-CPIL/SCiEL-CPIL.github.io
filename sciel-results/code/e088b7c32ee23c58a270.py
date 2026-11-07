"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _factorize(n):
    """Return dict of prime->exponent for n."""
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

def _kernel_bits(n, primes):
    """Return a frozenset of primes with odd exponent in n."""
    factors = _factorize(n)
    bits = set()
    for p, e in factors.items():
        if e % 2 == 1:
            bits.add(p)
    return frozenset(bits)

def _xor_sets(a, b):
    return a.symmetric_difference(b)

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _verify(selected, base=210):
    """Check that product(selected) * base is a perfect square."""
    prod = base
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def _find_solution_gf2(integers, target_kernel, primes, max_size=100):
    """
    Find a subset of integers whose kernel XOR equals target_kernel.
    Uses GF(2) Gaussian elimination. Returns list of chosen integers or None.
    """
    n = len(integers)
    if n == 0:
        return None
    
    # Map primes to indices
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Build matrix: each row is the kernel bit vector of integers[i]
    # We want to find a subset whose XOR = target_kernel
    # This is a linear system over GF(2)
    
    # Represent each vector as a Python int (bitmask)
    def kernel_to_int(kern):
        bits = 0
        for p in kern:
            if p in prime_to_idx:
                bits |= (1 << prime_to_idx[p])
        return bits
    
    target_int = kernel_to_int(target_kernel)
    
    # Gaussian elimination to find a solution
    # We have n equations (one per integer) and we want to pick a subset
    # Actually, we have a system: for each prime j, XOR of (x_i has prime j) = target_j
    # This is a system of num_primes equations in n unknowns (x_i in {0,1})
    # We want any solution with at most max_size ones.
    
    # Build augmented matrix: rows = primes, cols = integers + 1 (RHS)
    # A[j][i] = 1 if integers[i] has prime j in its kernel
    # b[j] = 1 if target_kernel has prime j
    
    # Use row-echelon form on the transpose approach
    # Actually, let's do standard GF(2) elimination
    
    # Matrix: each column i represents integer i, with bits for primes
    # We want to solve: sum_i x_i * v_i = target (mod 2)
    
    # Build list of (vector_int, index)
    vecs = []
    for i in range(n):
        kern = _kernel_bits(integers[i], primes)
        vi = kernel_to_int(kern)
        vecs.append((vi, i))
    
    # Gaussian elimination to find basis and express target
    # We'll do elimination on the vectors
    basis = {}  # pivot_bit -> (vector_int, list_of_indices)
    
    # Process vectors
    for vi, idx in vecs:
        v = vi
        indices = [idx]
        while v != 0:
            pivot = v.bit_length() - 1
            if pivot in basis:
                bv, bidx = basis[pivot]
                v ^= bv
                indices = indices + bidx
                # Remove duplicates (each index appears at most twice, so XOR)
                # Actually indices can have duplicates; we need to track parity
                # Let's use a set and toggle
                pass
            else:
                basis[pivot] = (v, indices)
                break
    
    # This approach is getting complex with index tracking. Let me use a different method.
    # Use standard GF(2) Gaussian elimination with tracking.
    
    # Redo: represent the system properly
    # We have n variables x_0..x_{n-1}, and num_primes equations
    # Equation j: sum_i A[j][i] * x_i = b[j] (mod 2)
    
    # Build the matrix as list of rows, where each row is (coeffs_dict, rhs)
    # But num_primes could be large. Let's use the vector representation.
    
    # Alternative: since we just need ANY solution, use the following:
    # Perform Gaussian elimination on the column vectors to find a basis.
    # Then check if target is in the span. If yes, back-substitute.
    
    # Let me use a cleaner approach with lists
    
    # Each vector as a list of 0/1 of length num_primes
    # But num_primes might be up to ~1000, and n up to ~10000. That's 10M entries.
    # Using Python ints as bitmasks is more efficient.
    
    # Redo Gaussian elimination with bitmask vectors
    # basis: dict pivot_bit -> (mask, [indices])
    # We track which original variables are in each basis vector
    
    basis = {}  # pivot -> (mask, set_of_indices)
    
    for i in range(n):
        v = vecs[i][0]
        indices = {i}
        while v != 0:
            pivot = v.bit_length() - 1
            if pivot in basis:
                bm, bi = basis[pivot]
                v ^= bm
                # Symmetric difference of index sets
                indices = indices.symmetric_difference(bi)
            else:
                basis[pivot] = (v, indices)
                break
    
    # Now check if target is in span
    v = target_int
    indices = set()
    solvable = True
    while v != 0:
        pivot = v.bit_length() - 1
        if pivot in basis:
            bm, bi = basis[pivot]
            v ^= bm
            indices = indices.symmetric_difference(bi)
        else:
            solvable = False
            break
    
    if not solvable:
        return None
    
    # indices is the set of variable indices with x_i = 1
    # Convert to list
    sol = sorted(indices)
    
    # Check size constraint
    if len(sol) > max_size:
        # Try to find a smaller solution using free variables
        # For now, if the particular solution is too large, we might need to
        # add null space vectors to reduce size. This is complex.
        # Let's try anyway - for small M, solutions should be small.
        pass
    
    # Extract the actual integers
    result = [integers[i] for i in sol]
    return result if result else None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    base = 210
    base_factors = _factorize(base)
    base_kernel = frozenset(p for p, e in base_factors.items() if e % 2 == 1)
    
    # Collect all primes up to 10210
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    all_primes = sieve(10210)
    
    # Precompute kernels for all integers from 211 to 10210
    # But do it incrementally as we search for M
    
    # Binary search on M
    lo = 211
    hi = 10210
    best_solution = None
    best_M = 10210
    
    # First, check if single integer works
    # x = 210 * k^2, x > 210, x <= 10210
    # 210 * k^2 <= 10210 => k^2 <= 48.6 => k <= 6
    # k=2: x=840, k=3: x=1890, k=4: x=3360, k=5: x=5250, k=6: x=7560
    # Smallest is 840, giving max-210 = 630
    
    # Now search for better solutions with multiple integers
    # Binary search on M
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if mid < 211:
            mid = 211
        
        # Get integers from 211 to mid
        integers = list(range(211, mid + 1))
        if not integers:
            lo = mid + 1
            continue
        
        # Find primes relevant to these integers
        # For efficiency, just use all_primes filtered to those <= mid
        relevant_primes = [p for p in all_primes if p <= mid]
        
        sol = _find_solution_gf2(integers, base_kernel, relevant_primes, max_size=100)
        
        if sol is not None and len(sol) >= 1 and len(sol) <= 100:
            # Verify
            if _verify(sol, base):
                best_solution = sol
                best_M = mid
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # If we found a solution, use it
    if best_solution is not None:
        # Ensure it's a list of Python ints
        result = [int(x) for x in best_solution]
        return result
    
    # Fallback: single integer 840
    return [840]
# EVOLVE-BLOCK-END
