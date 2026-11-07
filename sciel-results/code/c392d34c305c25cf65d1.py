"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    N = 40
    candidates = list(range(2, N + 1))
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map prime to index
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Factorize each candidate into an exponent vector
    vectors = []
    for n in candidates:
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        # temp should be 1 since all primes <= 40 are in the list
        vectors.append(vec)
        
    # Find the maximum linearly independent subset using Gaussian Elimination over Q
    # We maintain a list of basis vectors (as tuples of fractions or just rational numbers)
    # and the corresponding original integers.
    
    # To avoid floating point issues, we can use fractions or just integer arithmetic with pivots.
    # Since the numbers are small, we can use a simple basis extension algorithm.
    # A set of vectors is independent if we can add them one by one without one being a linear combination of previous.
    
    basis = []
    selected_indices = []
    
    # We want to maximize the count. Any basis of the spanned space has size = rank.
    # We just need to find the rank and a corresponding subset.
    
    # We iterate through candidates. If a vector is independent of the current basis, we add it.
    # This greedy approach finds a maximal independent set, which has size equal to the rank.
    
    def is_independent(vec, basis):
        # Check if vec is in the span of basis.
        # We can perform row reduction on [basis + vec] and see if vec becomes zero.
        # Or, simpler: maintain basis in row-echelon form.
        pass

    # Let's use a standard basis building algorithm.
    # We store basis vectors as lists of fractions.
    from fractions import Fraction
    
    basis_vecs = []
    
    for i, vec in enumerate(vectors):
        # Convert to Fraction list for exact arithmetic
        v = [Fraction(x, 1) for x in vec]
        
        # Try to reduce v by existing basis vectors
        # We assume basis_vecs are in row echelon form (pivot at increasing indices)
        # Actually, let's just keep a list and check independence by checking if v can be reduced to zero.
        
        # A more efficient way:
        # Maintain basis in row echelon form.
        # For each new vector, try to eliminate components using existing basis vectors.
        
        # Let's implement a simple check:
        # Is v linearly independent of basis_vecs?
        # We can add v to basis_vecs and check the rank increase.
        # Or, simpler for small N:
        # Just keep a list of accepted vectors.
        # To check if v is independent, we can try to express v as linear combo of basis.
        # This is solving a linear system.
        
        # Given the small size (max 12 basis vectors), we can just do:
        # 1. Add v to a copy of basis.
        # 2. Perform Gaussian elimination on this new set.
        # 3. If the rank increased, v was independent.
        
        # To be faster, we maintain basis in reduced row echelon form (RREF) or at least Row Echelon Form (REF).
        
        # Let's maintain basis in REF.
        # Pivot for basis_vec[j] is at index pivot_idx[j].
        
        # Actually, for N=39 and Dim=12, even O(N * Dim^3) is trivial.
        # Let's just store the accepted vectors and for each new one, check if it's independent.
        
        # Independence check:
        # Create matrix M with rows = basis_vecs + [v].
        # If rank(M) > len(basis_vecs), then v is independent.
        
        def get_rank(vecs):
            if not vecs:
                return 0
            # Copy vectors to list of lists of Fractions
            mat = [list(v) for v in vecs]
            rows = len(mat)
            cols = len(mat[0])
            rank = 0
            col = 0
            for r in range(rows):
                if col >= cols:
                    break
                # Find pivot
                pivot = -1
                for i in range(r, rows):
                    if mat[i][col] != 0:
                        pivot = i
                        break
                if pivot == -1:
                    col += 1
                    continue
                # Swap
                mat[r], mat[pivot] = mat[pivot], mat[r]
                # Eliminate below
                for i in range(r + 1, rows):
                    factor = mat[i][col] / mat[r][col]
                    for j in range(col, cols):
                        mat[i][j] -= factor * mat[r][j]
                rank += 1
                col += 1
            return rank

        current_rank = len(basis_vecs)
        if time.time() > deadline:
            break
            
        # Check if v increases rank
        test_vecs = basis_vecs + [v]
        new_rank = get_rank(test_vecs)
        
        if new_rank > current_rank:
            basis_vecs.append(v)
            selected_indices.append(i)
            
        # Optimization: if we already have 12, we can stop
        if len(basis_vecs) == num_primes:
            break

    result = [candidates[i] for i in selected_indices]
    
    # Ensure types are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
