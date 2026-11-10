"""
Find the largest nonempty subset of [1,255] containing 255,
with gcd(a,b) > 1 for all distinct pairs.
"""
import math
import time

def construct():
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Factorize 255 = 3 * 5 * 17
    # We want the largest set S ⊆ [1,255] with 255 ∈ S and gcd(a,b)>1 for all a≠b in S.
    
    # Key insight: 255 = 3*5*17. Any element in S must share a common factor with 255
    # (otherwise gcd with 255 would be 1). So every element must be divisible by 3, 5, or 17.
    # But also, any two elements in S must have gcd > 1.
    
    # The simplest large set: all multiples of 3 in [1,255] that also contain 255.
    # 255 is divisible by 3, so it's included. Size = floor(255/3) = 85.
    # Check: any two multiples of 3 have gcd at least 3 > 1. ✓
    
    # Can we do better? Consider mixing multiples of 3 and multiples of 5.
    # If a is a multiple of 3 (not 5) and b is a multiple of 5 (not 3), gcd(a,b) could be 1.
    # So we can't freely mix. We'd need to remove elements to maintain pairwise gcd > 1.
    
    # Strategy: Try all "prime-based" cliques and combinations.
    # A clique based on prime p: all multiples of p in [1,N]. Size = floor(N/p).
    # For p=3: 85, p=5: 51, p=17: 15.
    
    # But we can also consider sets where every element is divisible by at least one
    # of a set of primes, AND any two elements share at least one prime.
    # This is equivalent to: the set of elements whose prime factors intersect with
    # some "core" set, and pairwise intersection is non-empty.
    
    # The maximum is likely achieved by the multiples of the smallest prime factor.
    # Let's verify: multiples of 3 give 85 elements.
    
    # Can we add any element not divisible by 3?
    # Such an element must be divisible by 5 or 17 (to share factor with 255).
    # If we add a multiple of 5 (not 3), it must have gcd > 1 with every multiple of 3 in the set.
    # A multiple of 3 like 3 has gcd(3, 5) = 1. So we'd have to remove 3.
    # This trade-off is not favorable.
    
    # Let's do a more systematic search:
    # Start with multiples of 3. Try adding elements divisible by 5 or 17 (but not 3),
    # and remove elements from the multiples-of-3 set that conflict.
    
    best_set = set(range(3, N+1, 3))  # All multiples of 3
    best_size = len(best_set)
    
    # Try to improve by swapping: add a non-multiple-of-3 element, remove conflicting ones
    # For each candidate x not divisible by 3 but divisible by 5 or 17:
    candidates_to_add = [x for x in range(1, N+1) if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0) and x != 255]
    
    # Also consider: what if we use a different base?
    # Try multiples of 5: size 51. Worse.
    # Try multiples of 17: size 15. Worse.
    
    # What about a set where every element is divisible by 3 OR (divisible by 5 AND we've removed all multiples of 3 that conflict)?
    # This gets complex. Let's try a greedy approach.
    
    # Actually, let's think about it differently. The problem is finding the maximum
    # pairwise non-coprime set containing 255.
    
    # Known result: For N, the maximum size of a pairwise non-coprime subset of [1,N]
    # containing N is floor(N/p) where p is the smallest prime factor of N, IF N is
    # "sufficiently large" relative to its prime factors. For N=255=3*5*17, this should hold.
    
    # Let's verify by trying to extend the multiples-of-3 set.
    
    # For each candidate x (divisible by 5 or 17, not by 3, x ≠ 255):
    #   Find all y in current_set where gcd(x,y) == 1.
    #   If len(conflicting) < 1 (i.e., no conflicts), we can add x.
    #   If len(conflicting) >= 1, we'd need to remove them. Net gain = 1 - len(conflicting).
    
    current_set = set(range(3, N+1, 3))
    
    for x in candidates_to_add:
        if time.time() > deadline:
            break
        # Find conflicts
        conflicts = [y for y in current_set if math.gcd(x, y) == 1]
        net_gain = 1 - len(conflicts)
        if net_gain > 0:
            # Add x, remove conflicts
            current_set.difference_update(conflicts)
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(best_set)
    
    # Try starting from multiples of 5
    current_set = set(range(5, N+1, 5))
    candidates_to_add = [x for x in range(1, N+1) if x % 5 != 0 and (x % 3 == 0 or x % 17 == 0) and x != 255]
    for x in candidates_to_add:
        if time.time() > deadline:
            break
        conflicts = [y for y in current_set if math.gcd(x, y) == 1]
        net_gain = 1 - len(conflicts)
        if net_gain > 0:
            current_set.difference_update(conflicts)
            current_set.add(x)
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
    
    # Try starting from multiples of 17
    current_set = set(range(17, N+1, 17))
    candidates_to_add = [x for x in range(1, N+1) if x % 17 != 0 and (x % 3 == 0 or x % 5 == 0) and x != 255]
    for x in candidates_to_add:
        if time.time() > deadline:
            break
        conflicts = [y for y in current_set if math.gcd(x, y) == 1]
        net_gain = 1 - len(conflicts)
        if net_gain > 0:
            current_set.difference_update(conflicts)
            current_set.add(x)
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
    
    # Ensure 255 is in the result
    if 255 not in best_set:
        best_set.add(255)
    
    return list(best_set)
