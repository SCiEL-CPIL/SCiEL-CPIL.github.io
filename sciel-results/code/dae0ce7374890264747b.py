"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from typing import List, Tuple

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _get_exponent_vector(num: int) -> List[int]:
    """
    Returns the vector of exponents for the prime factorization of num.
    The vector has length NUM_PRIMES.
    """
    vec = [0] * NUM_PRIMES
    x = num
    for i, p in enumerate(PRIMES):
        if x % p == 0:
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            vec[i] = count
    # x should be 1 for all numbers in [2, 40]
    return vec

def _find_max_independent_set(numbers: List[int]) -> List[int]:
    """
    Finds the largest subset of numbers whose exponent vectors are linearly independent.
    Uses a simple Gaussian elimination approach over rationals (implemented with integers).
    Since the dimension is small (12), this is very fast.
    """
    basis = []  # List of (vector, number)
    
    # Sort numbers to have a deterministic order, though for max size it doesn't strictly matter
    # as long as we check all. However, picking primes first is a good heuristic for stability.
    # Actually, any maximal independent set will have the same size (rank of the matrix).
    # We just need to find one.
    
    for num in numbers:
        vec = _get_exponent_vector(num)
        
        # Try to reduce vec by the existing basis
        # We want to see if vec is in the span of the current basis.
        # If it is, skip it. If not, add it to the basis.
        
        # Simple reduction:
        # We'll maintain the basis in row-echelon form.
        # For simplicity, since N is small, we can just check if adding this vector increases the rank.
        # A faster way: Keep a list of basis vectors.
        # To check independence, we can try to express 'vec' as a linear combination of basis vectors.
        # Or, simpler: Add vec to basis, then perform Gaussian elimination on the whole set.
        # But that's O(N^3). N=39, so it's fine.
        
        # Let's use a more efficient incremental approach.
        # Maintain basis in row-echelon form.
        # For each new vector, try to eliminate it using existing basis vectors.
        
        v = vec[:]
        reduced = True
        # This simple loop doesn't guarantee full reduction if basis isn't in strict echelon form.
        # Let's just maintain a list of vectors and check rank at the end? No, we need the subset.
        
        # Better approach:
        # Maintain a list of basis vectors `basis_vecs` and corresponding numbers `basis_nums`.
        # For a new vector `v`:
        # 1. Try to subtract multiples of existing basis vectors to zero out components.
        # 2. If we can reduce `v` to zero, it's dependent.
        # 3. If not, it's independent. Add it to basis.
        
        # To make reduction work well, we should keep the basis in row-echelon form.
        # Let's implement a helper to reduce a vector against a basis.
        
        def reduce_vector(v, basis):
            # basis is a list of (pivot_index, vector)
            # v is the vector to reduce
            for pivot_idx, b_vec in basis:
                if v[pivot_idx] != 0:
                    # We need to make v[pivot_idx] zero.
                    # Since b_vec[pivot_idx] is the pivot (non-zero), we can do:
                    # v = v - (v[pivot_idx] / b_vec[pivot_idx]) * b_vec
                    # To keep integers, we can just check if the ratio is integer?
                    # Actually, linear independence over Q is what matters.
                    # We can use fractions or just float? No, exact arithmetic is better.
                    # But for small integers, float is risky.
                    # Let's use a simpler check:
                    # If we maintain the basis such that each basis vector has a unique pivot
                    # and all other basis vectors have 0 at that pivot, then:
                    # If v has a non-zero component at the pivot of a basis vector,
                    # we can subtract (v[pivot] / b[pivot]) * b.
                    # This requires division.
                    
                    # Alternative: Just check if the new vector is in the span.
                    # Since dimension is 12, we can just keep a list of all accepted vectors
                    # and periodically check the rank of the matrix formed by them.
                    # But we need to know *which* vectors to keep.
                    
                    pass
            return v

        # Let's use a different, simpler strategy for small N:
        # Keep a list of accepted numbers.
        # For each new number, check if the set of vectors (accepted + new) has the same rank
        # as the set of accepted vectors.
        # If rank increases, accept.
        
        # We need a function to calculate the rank of a set of vectors.
        
        def calculate_rank(vectors):
            if not vectors:
                return 0
            # Gaussian elimination
            # Make a copy
            mat = [v[:] for v in vectors]
            rows = len(mat)
            cols = len(mat[0])
            rank = 0
            for col in range(cols):
                # Find pivot
                pivot_row = -1
                for r in range(rank, rows):
                    if mat[r][col] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                # Swap
                mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
                # Eliminate
                for r in range(rows):
                    if r != rank and mat[r][col] != 0:
                        # mat[r] = mat[r] - (mat[r][col] / mat[rank][col]) * mat[rank]
                        # Use fractions to be safe? Or just check if divisible?
                        # Since we are checking independence over Q, we can use floats for rank determination
                        # if we are careful, but integers are better.
                        # Actually, for small integers, we can just use a determinant check for subspaces?
                        # No, Gaussian elimination with rational arithmetic (fractions) is safest.
                        
                        from fractions import Fraction
                        factor = Fraction(mat[r][col], mat[rank][col])
                        for c in range(cols):
                            mat[r][c] = mat[r][c] - factor * mat[rank][c]
                        # Convert back to int if possible? No, keep as Fraction.
                rank += 1
            return rank

        # This is O(N^3 * D) which is fine for N=39, D=12.
        
        current_vectors = [_get_exponent_vector(n) for n in current_numbers]
        current_rank = calculate_rank(current_vectors)
        
        new_vec = _get_exponent_vector(num)
        new_vectors = current_vectors + [new_vec]
        new_rank = calculate_rank(new_vectors)
        
        if new_rank > current_rank:
            current_numbers.append(num)
            current_vectors.append(new_vec)
            current_rank = new_rank

    return current_numbers

def construct():
    deadline = time.time() + 45
    start = time.time()
    
    # All candidate numbers in [2, 40]
    candidates = list(range(2, 41))
    
    # To maximize the chance of finding a basis quickly and robustly,
    # we can sort candidates. Primes are "pure" vectors, so they are good candidates.
    # But any maximal independent set works.
    # Let's just iterate through them.
    
    current_numbers = []
    
    # Optimization: Start with primes, as they are guaranteed independent.
    primes_in_range = [p for p in PRIMES if p <= 40]
    current_numbers.extend(primes_in_range)
    
    # Now try to add composites if possible (though we know max is 12, 
    # and we already have 12 primes, so we can't add more).
    # But for generality, if the range was larger, we would.
    # Here, len(current_numbers) is 12.
    # The rank of the 12 prime vectors is 12.
    # The dimension of the space is 12.
    # So no more vectors can be added.
    
    # However, to be safe and follow the "stronger algorithm" directive, 
    # let's verify the rank and ensure we have a maximal set.
    
    # The set of primes is already a maximal independent set.
    # We can just return it.
    
    # But let's double check if there's any weirdness.
    # The task asks to maximize the number.
    # We proved max is 12.
    # We have 12.
    # So we return the 12 primes.
    
    # Just in case my manual proof missed something (e.g. if a composite vector
    # allows a different configuration with more than 12? No, dimension bound is hard).
    
    # Let's run the general algorithm just to be sure it finds 12.
    
    # Reset and run the general finder
    # This is redundant but ensures the code path is exercised and robust.
    
    # Actually, the general finder above is O(N^4) if we recalculate rank from scratch each time.
    # N=39. 39^4 is ~ 2.3 million operations. Very fast.
    
    # Let's implement the incremental rank check properly to be efficient.
    
    # I will rewrite the construct to use a simple, correct, and fast method:
    # 1. Identify all prime exponent vectors.
    # 2. Use Gaussian elimination to find a basis.
    
    vectors_and_nums = []
    for n in candidates:
        vec = _get_exponent_vector(n)
        vectors_and_nums.append((vec, n))
    
    # Find basis
    basis_vecs = []
    basis_nums = []
    
    # We want to pick vectors that are linearly independent.
    # Iterate through candidates.
    # For each candidate, check if it is in the span of the current basis.
    # If not, add it to the basis.
    
    from fractions import Fraction
    
    # To check if v is in span of basis, we can try to reduce v to zero using basis vectors.
    # We need the basis to be in row-echelon form for this to be straightforward.
    
    # Let's maintain basis in row-echelon form.
    # basis is a list of (pivot_col, vector)
    
    echelon_basis = []
    
    for vec, num in vectors_and_nums:
        v = vec[:]
        # Reduce v using echelon_basis
        # For each basis vector, if v has a non-zero entry at the pivot col,
        # subtract a multiple of the basis vector.
        
        # Note: The pivot column of each basis vector is unique.
        # And for a basis vector with pivot p, all other basis vectors have 0 at p.
        
        is_dependent = True
        for pivot_col, b_vec in echelon_basis:
            if v[pivot_col] != 0:
                # We need to eliminate v[pivot_col]
                # factor = v[pivot_col] / b_vec[pivot_col]
                # v = v - factor * b_vec
                # We can do this with Fractions.
                factor = Fraction(v[pivot_col], b_vec[pivot_col])
                for i in range(NUM_PRIMES):
                    v[i] = v[i] - factor * b_vec[i]
                
                # After reduction, v[pivot_col] should be 0.
                # Check if it's effectively zero.
                if v[pivot_col] != 0:
                    # This shouldn't happen if arithmetic is exact
                    pass
                
        # After processing all basis vectors, if v is all zeros, it's dependent.
        # If not, it's independent.
        
        if all(x == 0 for x in v):
            continue # Dependent
        
        # It's independent. Add to echelon basis.
        # Find the first non-zero column in v.
        pivot_col = -1
        for i in range(NUM_PRIMES):
            if v[i] != 0:
                pivot_col = i
                break
        
        if pivot_col != -1:
            # We need to ensure that this new vector has 0 in all previous pivot columns.
            # Since we reduced it using all existing basis vectors, and their pivots are unique,
            # v should already have 0 at all existing pivot columns.
            # Let's verify.
            # Actually, the reduction step above ensures that for each existing basis vector's pivot,
            # the component in v becomes 0.
            
            # So v is ready to be added.
            # But wait, does v have non-zero components in columns that are pivots of *future* basis vectors?
            # Yes, but that's fine. When those future vectors are added, they will be reduced by this one.
            
            # However, standard Gaussian elimination requires that when we add a new row,
            # we eliminate it against previous rows. We did that.
            # We do NOT eliminate previous rows against the new row.
            # This results in a row-echelon form (upper triangular-like), not reduced row-echelon.
            # This is sufficient for checking independence.
            
            # Store the vector and its number.
            # We need to store the vector as integers or Fractions?
            # For subsequent reductions, Fractions are needed if the pivot is not 1.
            # So we store the vector as a list of Fractions.
            
            v_frac = [Fraction(x) for x in v]
            echelon_basis.append((pivot_col, v_frac))
            basis_nums.append(num)
            
            # Optimization: If we have reached the dimension, we can stop.
            if len(echelon_basis) == NUM_PRIMES:
                break

    return [int(x) for x in basis_nums]

# EVOLVE-BLOCK-END
