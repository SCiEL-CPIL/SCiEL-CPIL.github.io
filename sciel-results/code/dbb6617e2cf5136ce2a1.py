import time

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def construct():
    # We want to find the largest admissible set S in [0, N].
    # S is admissible if for all primes p <= |S|, S misses a residue mod p.
    # Strategy: Pick a threshold K. Remove residue 0 mod p for all primes p <= K.
    # Let S_K be the resulting set.
    # S_K is admissible if K >= |S_K| (since all primes <= |S_K| are <= K).
    # We want to maximize |S_K|.
    # |S_K| decreases as K increases.
    # We find the smallest K such that K >= |S_K|.
    # This K yields the largest valid S_K.
    
    primes = _primes(N)
    
    # Initialize S with all numbers
    # We will iteratively remove numbers divisible by the next prime.
    # But we need to check the condition K >= |S_K|.
    # Note: The condition depends on |S_K|.
    
    # We can iterate K.
    # For each K, we need |S_K|.
    # S_K is the set of numbers in [0, N] not divisible by any prime <= K.
    
    # Efficient construction:
    # Start with S = [0, N].
    # For each prime p in increasing order:
    #   Remove elements from S that are divisible by p.
    #   Check if p >= |S|.
    #   If yes, return S.
    
    # Why is this correct?
    # If we stop at prime p, we have removed residues for all primes <= p.
    # The set S is admissible for all primes q <= |S| if q <= p.
    # Since we stop when p >= |S|, all primes q <= |S| are <= p.
    # Thus S is admissible.
    # Since |S| decreases as we add more primes, the first time we satisfy p >= |S|
    # gives the largest possible |S| for this "remove 0" strategy.
    # Is "remove 0" optimal?
    # As derived, the size is invariant to the choice of residues (for a fixed set of primes).
    # So "remove 0" is as good as any other.
    
    S = set(range(N + 1))
    
    for p in primes:
        # Remove elements divisible by p
        # Optimization: Iterate over multiples of p
        # But S is a set.
        # We can filter S.
        
        # Check deadline
        if time.time() > _DEADLINE:
            break
        
        # Remove x from S if x % p == 0
        # To speed up, we can just create a new set or filter.
        # Given N=5000, S is small.
        
        S = {x for x in S if x % p != 0}
        
        # Check condition
        if p >= len(S):
            return list(S)
            
    # If we run out of primes (shouldn't happen for N=5000), return S
    return list(S)
# EVOLVE-BLOCK-END
