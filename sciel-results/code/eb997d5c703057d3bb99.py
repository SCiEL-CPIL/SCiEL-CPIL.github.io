import time
import math

N = 5000

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
    deadline = start_time + 290  # Safety margin for 300s limit
    
    # Start with the full set
    S = set(range(N + 1))
    
    # Precompute primes up to N
    primes = _primes(N)
    
    while time.time() < deadline:
        k = len(S)
        if k == 0:
            break
            
        # Find primes p <= k
        # We need to check if for any such p, all residues are covered.
        # We prefer to fix the smallest such p to reduce k significantly.
        
        target_prime = None
        target_residue = None
        min_count = float('inf')
        
        # Iterate through primes in increasing order
        for p in primes:
            if p > k:
                break
                
            # Check if p is fully covered
            # We need to count occurrences of each residue mod p
            # If any residue has count 0, p is NOT fully covered, so we are "good" for this p.
            # If all residues have count > 0, p IS fully covered, so we must repair.
            
            # Optimization: Instead of checking all primes, we can just find the first one that is fully covered.
            # Since we iterate in increasing order, the first one we find is the smallest.
            
            residue_counts = [0] * p
            for x in S:
                residue_counts[x % p] += 1
            
            # Check if any residue is missing
            if min(residue_counts) > 0:
                # This prime is fully covered. We need to remove one residue class.
                # We want to remove the class with the fewest elements to minimize damage.
                target_prime = p
                min_count = min(residue_counts)
                # Find the residue with min_count
                # There might be multiple, pick the first one
                for r in range(p):
                    if residue_counts[r] == min_count:
                        target_residue = r
                        break
                break # We found the smallest problematic prime, stop searching for others
        
        if target_prime is None:
            # No prime p <= k is fully covered. We are admissible.
            break
            
        # Remove elements in S that are congruent to target_residue mod target_prime
        to_remove = [x for x in S if x % target_prime == target_residue]
        for x in to_remove:
            S.remove(x)
            
    return sorted(S)
