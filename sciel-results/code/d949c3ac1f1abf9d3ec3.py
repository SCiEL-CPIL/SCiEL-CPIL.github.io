"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. The condition "all subset products are distinct" is equivalent to the numbers being 
       multiplicatively independent.
    2. For integers, multiplicative independence is equivalent to the linear independence 
       of their prime exponent vectors over the rationals (or integers).
    3. We map each integer n in 2..40 to a vector v_n where v_n[i] is the exponent of 
       the i-th prime in the factorization of n.
    4. We need to find the largest subset of these vectors that is linearly independent.
    5. This is equivalent to finding the rank of the matrix formed by these vectors, 
       and identifying a basis for the column space (or row space, depending on orientation).
    6. Since we want to maximize the count, we just need to find a maximal linearly 
       independent subset. Gaussian elimination can do this efficiently.
    
    Note: The vectors are in R^12 (primes up to 40 are 2,3,5,7,11,13,17,19,23,29,31,37).
    The maximum possible size is 12.
    """
    
    # 1. Generate primes up to 40
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(40)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Map each number 2..40 to its exponent vector
    vectors = []
    nums = list(range(2, 41))
    for n in nums:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        # If temp > 1, it's a prime > 40? No, n <= 40, so temp must be 1.
        vectors.append(v)
        
    # 3. Find a maximal linearly independent subset using Gaussian elimination
    # We treat each vector as a row in a matrix. We want to find which rows are 
    # linearly independent.
    # Actually, it's easier to build a basis incrementally.
    # Basis: list of vectors.
    # We iterate through all numbers. If a vector is independent of the current basis, 
    # we add it to the basis and keep the number.
    
    # To make elimination efficient, we can use fractions or just integers with careful 
    # pivot selection. Since numbers are small, integers are fine.
    # We will maintain the basis in row-echelon form.
    
    basis = [] # List of (original_index, vector)
    selected_indices = []
    
    for idx, v in enumerate(vectors):
        # Try to reduce v using current basis
        current_v = v[:]
        is_independent = False
        
        # Sort basis by pivot column? Or just iterate.
        # For simplicity and speed (small N), we can just do standard elimination.
        # However, to ensure we keep a maximal set, we should check if it adds to the rank.
        
        # Let's use a standard Gaussian elimination approach on the fly.
        # We maintain basis in row echelon form.
        # Each basis vector has a pivot (first non-zero element).
        
        # Check if current_v is zero
        if all(x == 0 for x in current_v):
            continue
            
        # Reduce current_v by existing basis vectors
        # We need to align pivots.
        # Let's keep basis sorted by pivot index for efficient reduction.
        
        # Actually, simpler approach:
        # Keep a list of basis vectors.
        # For each new vector, check if it is a linear combination of the basis.
        # This is hard to do directly with integers without fractions.
        
        # Alternative: Convert to fractions for robust elimination.
        # Or, since the dimension is small (12), we can just check if the rank increases.
        # Rank check: Add v to the set of basis vectors, compute rank. If rank increased, keep it.
        # Computing rank from scratch every time is O(N^4) which is fine for N=39, dim=12.
        
        def get_rank(vecs):
            # Gaussian elimination to find rank
            if not vecs:
                return 0
            matrix = [list(v) for v in vecs]
            rows = len(matrix)
            cols = len(matrix[0])
            
            rank = 0
            for col in range(cols):
                # Find pivot
                pivot_row = None
                for r in range(rank, rows):
                    if matrix[r][col] != 0:
                        pivot_row = r
                        break
                if pivot_row is None:
                    continue
                    
                # Swap pivot row with current rank row
                matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
                
                # Eliminate below
                for r in range(rank + 1, rows):
                    factor = matrix[r][col] / matrix[rank][col]
                    if factor != 0:
                        for c in range(col, cols):
                            matrix[r][c] -= factor * matrix[rank][c]
                            
                rank += 1
            return rank

        # This rank check is a bit slow if done naively with fractions.
        # Let's use a more direct incremental basis construction with fractions.
        
    # Let's restart the selection logic with a cleaner incremental Gaussian Elimination
    # using Fraction for exactness and simplicity.
    
    from fractions import Fraction
    
    basis_vecs = [] # List of vectors in row echelon form
    basis_orig_idx = [] # Corresponding original indices
    
    for idx, v in enumerate(vectors):
        # Convert to Fraction
        fv = [Fraction(x) for x in v]
        
        # Reduce fv using basis_vecs
        # basis_vecs is in row echelon form, so pivot of basis_vecs[i] is at some col
        # We need to know the pivot columns.
        
        # Let's store (pivot_col, vec) for basis
        # Actually, let's just store the vectors and re-derive pivots or store them.
        
        # Efficient approach:
        # Maintain basis in reduced row echelon form (RREF) or just Row Echelon.
        # Let's use a dictionary or list of (pivot, vec).
        
        # Re-implementing incremental basis with pivot tracking
        # basis: list of tuples (pivot_col, vector_as_Fractions)
        # sorted by pivot_col
        
        current = fv[:]
        # Try to eliminate leading terms
        for pivot_col, b_vec in sorted(basis, key=lambda x: x[0]):
            if current[pivot_col] != 0:
                factor = current[pivot_col] / b_vec[pivot_col]
                for j in range(pivot_col, num_primes):
                    current[j] -= factor * b_vec[j]
                    
        # Check if current is zero
        if all(x == 0 for x in current):
            # Dependent, skip
            continue
        else:
            # Find new pivot
            new_pivot = None
            for j in range(num_primes):
                if current[j] != 0:
                    new_pivot = j
                    break
                    
            if new_pivot is not None:
                # Normalize current vector so pivot is 1
                # (Optional, but good for RREF)
                # For just independence check, we don't strictly need normalization, 
                # but it makes future elimination cleaner.
                # Let's normalize.
                inv_pivot = Fraction(1) / current[new_pivot]
                current = [x * inv_pivot for x in current]
                
                # Add to basis
                basis.append((new_pivot, current))
                selected_indices.append(idx)
                
                # Sort basis by pivot column to maintain order for next iterations
                basis.sort(key=lambda x: x[0])
                
    # The selected indices correspond to the numbers in `nums`
    # nums is range(2, 41)
    result = [nums[i] for i in selected_indices]
    
    # Verify validity (just in case, though math guarantees it)
    # All subset products distinct?
    # Since vectors are linearly independent, yes.
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
