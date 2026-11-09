"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _get_vector(n):
    """Return the prime exponent vector for n as a numpy array."""
    vec = np.zeros(NUM_PRIMES, dtype=int)
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return vec

def _rank_of_matrix(matrix):
    """Calculate the rank of a list of vectors."""
    if not matrix:
        return 0
    mat = np.array(matrix)
    return np.linalg.matrix_rank(mat)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    candidates = list(range(2, 41))
    vectors = {n: _get_vector(n) for n in candidates}
    
    best_size = 0
    best_set = []
    
    # Strategy 1: Greedy with various orderings
    # We want to maximize the number of linearly independent vectors.
    # Simple greedy: add vector if it increases the rank.
    
    orderings = [
        "ascending",
        "descending",
        "by_prime_count",
        "random"
    ]
    
    def greedy_select(order):
        if order == "ascending":
            ordered_candidates = sorted(candidates)
        elif order == "descending":
            ordered_candidates = sorted(candidates, reverse=True)
        elif order == "by_prime_count":
            # Prioritize numbers with fewer prime factors (primes themselves, prime powers)
            def pc(n):
                v = vectors[n]
                return np.count_nonzero(v)
            ordered_candidates = sorted(candidates, key=pc)
        elif order == "random":
            import random
            ordered_candidates = candidates[:]
            random.shuffle(ordered_candidates)
        else:
            ordered_candidates = candidates
            
        current_set = []
        current_vectors = []
        current_rank = 0
        
        for n in ordered_candidates:
            if time.time() > deadline:
                break
            v = vectors[n]
            # Check if adding v increases rank
            # We can do this by checking if v is in the span of current_vectors
            # Or simply calculate rank of [current_vectors + [v]]
            # Since dim is small (12), this is fast.
            
            if not current_vectors:
                current_set.append(n)
                current_vectors.append(v)
                current_rank = 1
            else:
                # Optimization: if current_rank == NUM_PRIMES, we can't add more
                if current_rank == NUM_PRIMES:
                    break
                
                # Create a new matrix and check rank
                # To be efficient, we can try to see if v is linearly dependent
                # using Gaussian elimination on the augmented matrix, but np.linalg.matrix_rank is fine for small N.
                
                # Check rank
                new_mat = np.vstack([np.array(current_vectors), v])
                new_rank = np.linalg.matrix_rank(new_mat)
                
                if new_rank > current_rank:
                    current_set.append(n)
                    current_vectors.append(v)
                    current_rank = new_rank
        
        return current_set

    # Run greedy with deterministic orderings
    for order in ["ascending", "descending", "by_prime_count"]:
        if time.time() > deadline:
            break
        s = greedy_select(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # If we haven't reached the theoretical max (12), try random restarts
    if best_size < NUM_PRIMES:
        import random
        while time.time() < deadline:
            s = greedy_select("random")
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
                if best_size == NUM_PRIMES:
                    break
            
            # Occasionally try a "swap" improvement on the best set found
            if best_size > 0 and best_size < NUM_PRIMES:
                # Try to replace one element to see if we can add another?
                # This is complex. Simple random restarts are usually sufficient for this small problem.
                pass

    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
