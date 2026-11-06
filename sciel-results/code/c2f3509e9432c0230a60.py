"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from math import isqrt

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)
PRIME_SET = set(PRIMES)

def construct():
    """
    Construct a set of prime residue classes covering 1..L with sum 1/p <= 1.
    
    Strategy: Greedy set cover using bitmask representation for fast coverage counting.
    """
    # We'll work with a target range first, then extend
    initial_L = 1000
    
    # Bitmask for 1..initial_L
    # Bit i represents number i+1 (so bit 0 = 1, bit 1 = 2, ..., bit 999 = 1000)
    
    uncovered = (1 << initial_L) - 1  # All bits set
    
    classes = []
    used_primes = set()
    remaining_budget = Fraction(1, 1)
    
    # Precompute masks for each (p, r) combination
    # For prime p, residue r: mask has bits set where (i+1) % p == r
    # i ranges from 0 to initial_L-1, so number = i+1
    
    # We'll compute these on the fly during greedy steps to save memory
    
    while uncovered:
        best_p = None
        best_r = None
        best_new_coverage = 0
        
        # For each unused prime that fits in budget
        for p in PRIMES:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
            
            # For each residue r in 0..p-1, compute how many uncovered numbers it covers
            # Number n is covered if n % p == r
            # n = i+1 where i is the bit index
            # So we need (i+1) % p == r, i.e., i % p == (r-1) % p
            
            target_residues = []
            for r in range(p):
                # Find which bit positions i satisfy (i+1) % p == r
                # i % p == (r-1) % p
                target = (r - 1) % p
                
                # Count uncovered bits at positions where i % p == target
                count = 0
                i = target
                while i < initial_L:
                    if (uncovered >> i) & 1:
                        count += 1
                    i += p
                
                if count > best_new_coverage:
                    best_new_coverage = count
                    best_p = p
                    best_r = r
            
            # Early termination if we found a perfect cover for this step
            if best_new_coverage == 0:
                continue
        
        if best_p is None:
            break
        
        # Add this class
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        remaining_budget -= Fraction(1, best_p)
        
        # Update uncovered: clear bits where (i+1) % best_p == best_r
        # i % best_p == (best_r - 1) % best_p
        target = (best_r - 1) % best_p
        i = target
        while i < initial_L:
            uncovered &= ~(1 << i)
            i += best_p
    
    # Now we've covered 1..initial_L (hopefully)
    # Check how far we actually covered
    # Find the first gap
    first_gap = None
    for n in range(1, initial_L + 1):
        covered = False
        for p, r in classes:
            if n % p == r:
                covered = True
                break
        if not covered:
            first_gap = n
            break
    
    if first_gap is not None and first_gap > 1:
        # We didn't fully cover 1..initial_L, so L is first_gap - 1
        L = first_gap - 1
    else:
        # We covered 1..initial_L, try to extend
        # Continue adding primes to cover beyond initial_L
        L = initial_L
        
        # Try to extend coverage
        # We'll check further and add primes if beneficial
        check_limit = min(10000, 100000)
        
        while True:
            # Find first gap after L
            next_gap = None
            for n in range(L + 1, check_limit + 1):
                covered = False
                for p, r in classes:
                    if n % p == r:
                        covered = True
                        break
                if not covered:
                    next_gap = n
                    break
            
            if next_gap is None:
                L = check_limit
                break
            
            # Try to add a prime to cover next_gap
            best_p = None
            best_r = None
            best_future_coverage = 0
            
            for p in PRIMES:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if remaining_budget < cost:
                    continue
                
                # To cover next_gap, we need next_gap % p == r
                r = next_gap % p
                
                # Count how many numbers in [next_gap, next_gap + 1000] would be newly covered
                future_cov = 0
                for n in range(next_gap, min(next_gap + 1000, check_limit + 1)):
                    if n % p == r:
                        # Check if already covered
                        already = False
                        for p2, r2 in classes:
                            if n % p2 == r2:
                                already = True
                                break
                        if not already:
                            future_cov += 1
                
                if future_cov > best_future_coverage:
                    best_future_coverage = future_cov
                    best_p = p
                    best_r = r
            
            if best_p is None or best_future_coverage == 0:
                L = next_gap - 1
                break
            
            classes.append([best_p, best_r])
            used_primes.add(best_p)
            remaining_budget -= Fraction(1, best_p)
            L = next_gap - 1  # At least we covered up to here
    
    # Final verification: find the actual L
    # Check from 1 upward until we find a gap
    final_L = 0
    for n in range(1, 100001):
        covered = False
        for p, r in classes:
            if n % p == r:
                covered = True
                break
        if covered:
            final_L = n
        else:
            break
    
    return classes

# EVOLVE-BLOCK-END
