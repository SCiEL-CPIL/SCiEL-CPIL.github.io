"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    deadline = 300.0
    margin = 5.0  # seconds
    
    # Start with the full set of integers in [0, N]
    # This maximizes the initial pool of candidates.
    T = set(range(N + 1))
    
    # Generate primes up to N
    primes = _primes(N)
    
    for p in primes:
        # Check time limit
        if time.time() > deadline - margin:
            break
            
        current_len = len(T)
        
        # The condition only applies to primes p <= len(T)
        if p > current_len:
            break
            
        # Count the number of elements in T for each residue class mod p
        counts = [0] * p
        for x in T:
            counts[x % p] += 1
            
        # If all residue classes are present, we must remove one to make it admissible.
        # To maximize the size of T, we remove the class with the fewest elements.
        min_count = min(counts)
        
        if min_count > 0:
            # Find the index of the minimum count
            # If there are ties, index() returns the first one
            target_class = counts.index(min_count)
            
            # Remove all elements in T that belong to this class
            # Using set comprehension is efficient for this size
            T = {x for x in T if x % p != target_class}
            
    return list(T)
# EVOLVE-BLOCK-END
