"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _get_prime_exponents(n, primes):
    """Returns a tuple of exponents for each prime in the list."""
    exponents = []
    for p in primes:
        count = 0
        while n % p == 0:
            count += 1
            n //= p
        exponents.append(count)
    return tuple(exponents)

def _has_distinct_subset_products(numbers, primes):
    """
    Checks if the set of numbers has distinct subset products.
    Uses the property that this is equivalent to the subset sums of 
    prime exponent vectors being distinct.
    """
    if not numbers:
        return True
    
    # Get vectors
    vectors = [_get_prime_exponents(n, primes) for n in numbers]
    
    # Start with the empty sum
    sums = {tuple([0] * len(primes))}
    
    for v in vectors:
        new_sums = set()
        for s in sums:
            # Add v to s
            new_s = tuple(s[i] + v[i] for i in range(len(primes)))
            new_sums.add(new_s)
        
        # The new set of sums is the union of old sums and new sums
        # If there's a collision, the size won't double
        if len(sums) + len(new_sums) != len(sums.union(new_sums)):
            return False
        sums = sums.union(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Candidates
    candidates = list(range(2, P['n'] + 1))
    
    # Pre-compute vectors for all candidates
    candidate_vectors = {n: _get_prime_exponents(n, primes) for n in candidates}
    
    # We want to maximize the size.
    # We will try to find the largest subset.
    # Since the max size is small (~16), we can use a search.
    
    best_set = []
    
    # Order candidates: Primes first, then prime powers, then others
    # This helps greedy/backtracking find large sets faster
    def sort_key(n):
        v = candidate_vectors[n]
        # Primes have exactly one non-zero entry, which is 1
        non_zero = sum(1 for x in v if x > 0)
        max_exp = max(v) if v else 0
        # Prioritize primes (non_zero=1, max_exp=1)
        # Then prime powers (non_zero=1, max_exp>1)
        # Then others
        if non_zero == 1:
            return (0, max_exp)
        else:
            return (1, n)
            
    candidates.sort(key=sort_key)
    
    # Backtracking search
    # We try to include candidates.
    # Pruning: if current_size + remaining < best_size, prune.
    
    def backtrack(index, current_set, current_sums):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        # Update best
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # Pruning
        if len(current_set) + (len(candidates) - index) <= len(best_set):
            return
            
        if index == len(candidates):
            return
            
        # Try including candidates[index]
        v = candidate_vectors[candidates[index]]
        
        # Check if adding v keeps sums distinct
        # new_sums = {s + v for s in current_sums}
        # Check collision
        has_collision = False
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + v[i] for i in range(len(primes)))
            if new_s in current_sums:
                has_collision = True
                break
            new_sums.add(new_s)
            
        if not has_collision:
            # Recurse with inclusion
            new_current_sums = current_sums.union(new_sums)
            current_set.append(candidates[index])
            backtrack(index + 1, current_set, new_current_sums)
            current_set.pop()
            
        # Try excluding candidates[index]
        backtrack(index + 1, current_set, current_sums)
        
    # Initial sums: just the zero vector
    initial_sums = {tuple([0] * len(primes))}
    backtrack(0, [], initial_sums)
    
    # Return the best set found
    # Ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
