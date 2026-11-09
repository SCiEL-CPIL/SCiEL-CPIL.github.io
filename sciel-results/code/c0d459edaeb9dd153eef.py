"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Initialize S as a set of all integers in [0, N]
    # Using a list of booleans or a set? 
    # A set allows easy removal, but counting residues requires iteration.
    # A list of active indices might be faster for iteration if we filter.
    # Let's use a list of integers and filter it.
    S = list(range(N + 1))
    
    # Get primes up to N
    primes = _primes(N)
    
    for p in primes:
        # Condition: p <= len(S)
        if p > len(S):
            break
            
        # Count residues mod p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        # Check if any residue class is empty
        if min(counts) == 0:
            continue
            
        # Find the residue class with the minimum count
        min_count = min(counts)
        # If multiple classes have the same min count, pick the first one (arbitrary)
        # Actually, we just need to pick one that minimizes removal.
        # All min_count classes remove the same number of elements.
        target_residue = counts.index(min_count)
        
        # Remove all x in S such that x % p == target_residue
        S = [x for x in S if x % p != target_residue]
        
    return S
# EVOLVE-BLOCK-END
