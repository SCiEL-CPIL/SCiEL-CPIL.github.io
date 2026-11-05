"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Start with the full set [0, 5000]
    # Using a list for efficient iteration and removal via filtering
    # But sets are better for membership? 
    # We need to remove specific elements. 
    # Let's use a list of booleans or a set.
    # A set of integers is fine.
    S = set(range(N + 1))
    
    # Get primes up to N+1 (max possible size)
    primes = _primes(N + 1)
    
    for p in primes:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
            
        current_size = len(S)
        
        # If p is greater than the current size, the condition "for every prime p <= len(tuple)" 
        # implies we don't need to satisfy constraints for this p or larger primes 
        # because len(tuple) < p. 
        # Wait, the condition is: for every prime p <= len(S), some residue class is missed.
        # So if p > len(S), we don't care about p.
        # Since we iterate primes in increasing order, if p > len(S), all subsequent primes are also > len(S).
        # So we can break.
        if p > current_size:
            break
            
        # Count residues mod p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        # Check if any class is empty
        if 0 in counts:
            continue
            
        # All classes are covered. We must remove elements from one class.
        # To maximize the remaining set size, remove the class with the fewest elements.
        min_count = min(counts)
        # Find all residues with this min count. 
        # If there are multiple, it doesn't matter which one we pick for the count reduction, 
        # but it might affect future steps? 
        # Generally, picking any min is fine. Let's pick the first one.
        target_residue = counts.index(min_count)
        
        # Remove all x in S such that x % p == target_residue
        # S = {x for x in S if x % p != target_residue}
        
        # Optimization: iterate and filter
        new_S = set()
        for x in S:
            if x % p != target_residue:
                new_S.add(x)
        S = new_S

    return list(S)
# EVOLVE-BLOCK-END
