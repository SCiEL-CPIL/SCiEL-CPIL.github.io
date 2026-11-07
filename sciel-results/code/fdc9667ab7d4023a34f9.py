"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    from math import isqrt
    from itertools import combinations
    import random

    # Factor each number 2..40 into prime-exponent tuple.
    def factor(n):
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

    # Build exponent vectors for 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    vecs = {}
    for n in range(2, 41):
        f = factor(n)
        vecs[n] = tuple(f.get(p, 0) for p in primes)

    def has_distinct_subset_products(selected):
        """Check that all 2^len(selected) subset products are distinct."""
        k = len(selected)
        if k > 20:
            return False  # safety
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask >> i & 1:
                    prod *= selected[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    def find_basis(candidates, vec_map):
        """Find a maximal linearly independent set over Q using Gaussian Elimination."""
        # We treat vectors as rows in a matrix.
        # We want to find the largest subset that is linearly independent.
        # This is equivalent to finding the rank.
        # We use a greedy approach with rational arithmetic to maintain a basis.
        
        basis = []
        # To avoid floating point issues, we use fractions or just keep track of the "pivot" column.
        # Since dimension is small (12), we can use simple elimination.
        
        # Represent vectors as lists of floats for elimination, but keep original ints for verification.
        # Actually, for linear independence over Q, we can just check if the vector is in the span of the basis.
        
        # Simpler: Just try to add vectors. If it increases the rank, add it.
        # Rank calculation via Gaussian Elimination on the current basis + new vector.
        
        def get_rank(matrix):
            """Calculate rank of a list of tuples (vectors) using Gaussian Elimination."""
            if not matrix:
                return 0
            # Copy to avoid mutation
            mat = [list(row) for row in matrix]
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
                pivot_val = mat[rank][col]
                for r in range(rows):
                    if r != rank and mat[r][col] != 0:
                        factor = mat[r][col] / pivot_val
                        for c in range(col, cols):
                            mat[r][c] -= factor * mat[rank][c]
                rank += 1
            return rank

        current_rank = 0
        basis_vecs = []
        
        # Sort candidates to prefer "simpler" vectors first?
        # Or just iterate.
        for n in candidates:
            v = vec_map[n]
            # Check if adding v increases rank
            new_basis = basis_vecs + [list(v)]
            r = get_rank(new_basis)
            if r > current_rank:
                basis_vecs.append(list(v))
                current_rank = r
                basis.append(n)
        return basis

    # 1. Find a basis
    candidates = list(range(2, 41))
    # Shuffle to break ties? No, deterministic is better for reproducibility unless we use random restarts.
    # Let's try a few different orderings.
    
    best_set = []
    
    for seed in range(10):
        random.seed(seed)
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Find basis
        basis = find_basis(shuffled, vecs)
        current_set = basis[:]
        
        # Try to add remaining elements
        remaining = [n for n in shuffled if n not in current_set]
        
        # Multiple passes
        for _ in range(3):
            for n in remaining:
                if n in current_set:
                    continue
                trial = current_set + [n]
                if has_distinct_subset_products(trial):
                    current_set = trial
                    # Remove from remaining? 
                    # It's in current_set now, so the check `if n in current_set` will skip it next time.
            
            # Try swapping?
            # If we can't add, maybe we can swap a bad element for two good ones?
            # This is complex. Let's stick to addition first.
            
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    # Final verification
    if has_distinct_subset_products(best_set):
        return list(best_set)
    else:
        # Fallback to primes
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# EVOLVE-BLOCK-END
