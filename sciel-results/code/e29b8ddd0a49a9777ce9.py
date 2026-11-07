"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pindex = {p: i for i, p in enumerate(primes)}
    
    # Build candidate list: square-free integers in [2, 40]
    candidates = []
    for n in range(2, 41):
        # Check square-free: no prime squared divides n
        square_free = True
        for p in primes:
            if p * p > n:
                break
            if n % (p * p) == 0:
                square_free = False
                break
        if not square_free:
            continue
        # Build exponent vector
        vec = [0] * len(primes)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        candidates.append((n, vec))
    
    # Greedy Gaussian elimination over Q to find max linearly independent subset
    # We process candidates and maintain a row-reduced echelon form
    # basis: list of (pivot_col, vector_as_Fraction_list)
    basis = []  # list of vectors, each reduced, with known pivot columns
    pivot_cols = []  # pivot column for each basis vector
    selected = []
    
    for n, vec in candidates:
        v = [Fraction(x) for x in vec]
        # Try to reduce v against existing basis
        for bi, bv in enumerate(basis):
            pc = pivot_cols[bi]
            if v[pc] != 0:
                factor = v[pc] / bv[pc]
                for j in range(len(v)):
                    v[j] -= factor * bv[j]
        # After reduction, check if v is zero
        if all(x == 0 for x in v):
            continue
        # Find pivot column (first non-zero)
        pc = None
        for j in range(len(v)):
            if v[j] != 0:
                pc = j
                break
        if pc is None:
            continue
        # Add to basis: normalize and eliminate from other basis vectors
        # Normalize so pivot is 1
        pv = v[pc]
        if pv != 1:
            v = [x / pv for x in v]
        # Eliminate this pivot from existing basis vectors
        for bi, bv in enumerate(basis):
            if bv[pc] != 0:
                factor = bv[pc]
                for j in range(len(bv)):
                    bv[j] -= factor * v[j]
        basis.append(v)
        pivot_cols.append(pc)
        selected.append(n)
    
    # Return as list of Python ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
