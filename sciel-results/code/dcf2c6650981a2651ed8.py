"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    from math import gcd
    from itertools import combinations
    
    # Generate all integers 2..40
    candidates = list(range(2, 41))
    n = len(candidates)
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Factor each number into prime exponent vector
    def factorize(n_val):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_val % p == 0:
                vec[i] += 1
                n_val //= p
        return tuple(vec)
        
    vecs = [factorize(x) for x in candidates]
    
    # We want the largest subset of vectors that are linearly independent over Q.
    # This is equivalent to finding a basis for the span of these vectors.
    # The size of the basis is the rank of the set of vectors.
    # Since the set of primes is included, the rank is exactly num_primes (12).
    # We can find any basis using a greedy algorithm (Gaussian elimination style).
    
    basis = []
    selected_indices = []
    
    # We process vectors in order. 
    # To ensure we get a valid basis, we check if the new vector increases the rank.
    # We maintain the basis in row-echelon form for efficient rank checking.
    # Since dimension is small (12), we can just use a list of vectors and check rank via Gaussian elimination on the fly.
    
    def is_independent(vec, current_basis):
        # Create a matrix with current_basis + [vec]
        # Check if rank increases
        # Since dim is 12, we can represent vectors as tuples of 12 ints.
        # We perform Gaussian elimination to see if vec is in span of current_basis.
        
        # Convert to list of lists for manipulation
        matrix = [list(v) for v in current_basis]
        matrix.append(list(vec))
        
        m = len(matrix)
        n_cols = len(matrix[0]) if m > 0 else 0
        
        if m == 0:
            return True
            
        # Gaussian Elimination
        rank = 0
        for col in range(n_cols):
            # Find pivot
            pivot_row = -1
            for r in range(rank, m):
                if matrix[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
                
            # Swap rows
            matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
            
            # Eliminate below
            for r in range(rank + 1, m):
                if matrix[r][col] != 0:
                    factor = matrix[r][col] / matrix[rank][col]
                    for c in range(col, n_cols):
                        matrix[r][c] -= factor * matrix[rank][c]
            rank += 1
            if rank == m:
                break
                
        # If rank == m, all vectors are independent.
        # Specifically, we added 'vec' to the basis.
        # If the rank of (basis + vec) is rank(basis) + 1, then it's independent.
        # We know rank(basis) is the number of vectors in current_basis (since they were added independently).
        # So we just check if rank of the new matrix is len(current_basis) + 1.
        
        return rank == len(current_basis) + 1

    for i in range(n):
        if is_independent(vecs[i], basis):
            basis.append(vecs[i])
            selected_indices.append(i)
            
    result = [candidates[i] for i in selected_indices]
    
    # Verify all subset products are distinct
    def verify(nums):
        seen = set()
        seen.add(1)  # empty product
        k = len(nums)
        for r in range(1, k + 1):
            for combo in combinations(nums, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in seen:
                    return False
                seen.add(prod)
        return True
        
    if not verify(result):
        # Fallback to primes if greedy failed (should not happen)
        prime_indices = [i for i, v in enumerate(vecs) if sum(v) == 1]
        result = [candidates[i] for i in prime_indices]
        
    return result

# EVOLVE-BLOCK-END
