"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 40
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
            
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Function to get prime factorization vector for a number
    def factorize(x):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while x % p == 0:
                vec[i] += 1
                x //= p
        return tuple(vec)
        
    # We want to find a maximum linearly independent set of vectors in Z^k
    # where k is the number of primes.
    # Since the primes themselves form a basis, the max size is len(primes).
    
    # Greedy selection:
    # Start with empty set.
    # Iterate through numbers 2..n.
    # If the vector of the number is linearly independent of the current set, add it.
    
    selected_vectors = []
    selected_numbers = []
    
    def is_independent(vec, current_vectors):
        # Check if vec is linearly independent of current_vectors
        # We can use Gaussian elimination on the matrix formed by current_vectors + [vec]
        # Or simpler: check if vec can be formed by integer linear combination? 
        # Actually, linear independence over Q is sufficient for distinct subset products.
        # Since we are in Z, linear independence over Q implies linear independence over Z for the basis?
        # Wait, subset products distinct <=> no non-trivial integer combination of exponents is 0.
        # This is exactly linear independence over Q.
        
        # Let's build a matrix with current_vectors as rows and check rank.
        # But simpler: since we are adding one vector at a time, we can just check if it's in the span.
        # However, "span" over Q is hard to check directly with integers without fractions.
        # But note: the vectors are sparse and small.
        
        # Alternative: The set of prime exponent vectors for primes 2..p_k is the identity matrix.
        # Any set of k+1 vectors in k-dimensional space is dependent.
        # So max size is k.
        # The primes themselves are independent.
        # So we can just return the primes.
        
        # To be "stronger" and general, let's implement a simple rank check.
        # Matrix M with rows = current_vectors + [vec].
        # If rank(M) == len(current_vectors) + 1, it's independent.
        
        # We can perform Gaussian elimination on the matrix of integers.
        # Rows are tuples of ints.
        
        matrix = list(current_vectors)
        matrix.append(list(vec))
        
        # Gaussian elimination to count rank
        rank = 0
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        
        # Transpose to work with columns? Or just standard row reduction.
        # Let's do standard row reduction on the matrix.
        # We want to find the number of linearly independent rows.
        
        # Since the dimension is small (12), this is fast.
        # Convert to list of lists for mutation
        mat = [list(row) for row in matrix]
        
        r = 0
        for c in range(cols):
            # Find pivot
            pivot = -1
            for i in range(r, rows):
                if mat[i][c] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue
            # Swap
            mat[r], mat[pivot] = mat[pivot], mat[r]
            # Eliminate below
            for i in range(r + 1, rows):
                factor = mat[i][c] / mat[r][c]
                if factor != 0:
                    for j in range(c, cols):
                        mat[i][j] -= factor * mat[r][j]
            r += 1
            if r == rows:
                break
                
        # The rank is r.
        # If rank == len(current_vectors) + 1, then vec is independent.
        return r == len(current_vectors) + 1

    # Iterate through candidates
    # To maximize the chance of finding a large set, we should prioritize numbers that are "simple"
    # i.e., primes, or prime powers, etc.
    # Actually, the order doesn't matter for the *existence* of a max set, but for greedy,
    # picking primes first ensures we get the full basis.
    # Let's iterate from 2 to n.
    
    for x in range(2, n + 1):
        if time.time() > deadline:
            break
        vec = factorize(x)
        if is_independent(vec, selected_vectors):
            selected_vectors.append(vec)
            selected_numbers.append(x)
            
    # The result should be the list of selected numbers
    # Ensure they are Python ints
    return [int(x) for x in selected_numbers]
# EVOLVE-BLOCK-END
