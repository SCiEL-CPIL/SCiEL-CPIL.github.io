import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate primes
    primes = [p for p in range(2, 1001) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    # We determined that using primes >= 13 (163 primes) is a good base set.
    # Sum of reciprocals is ~0.92.
    # We can potentially use 11 if we drop a large prime, but let's stick to the 163 primes >= 13 first.
    # Actually, let's allow the algorithm to pick from all primes, but enforce the sum constraint.
    # To ensure we don't exceed budget, we can sort primes by 1/p (descending) and pick greedily?
    # No, we need to maximize L.
    
    # Let's try the sequential greedy on the set of primes >= 13.
    # If we get stuck or run out of primes, we stop.
    
    # Candidate primes: all primes <= 1000.
    # But we must respect sum(1/p) <= 1.
    # The set {13..997} works.
    # Let's use this set.
    
    candidate_primes = primes
    
    L_max = 10000
    covered = np.zeros(L_max + 1, dtype=bool)
    indices = np.arange(1, L_max + 1)
    
    selected_classes = []
    used_primes = set()
    current_sum = Fraction(0)
    
    # We will iterate x from 1 to L_max
    # For each x, if not covered, we must pick a prime to cover it.
    # Which prime?
    # We want to pick a prime p in candidate_primes \ used_primes
    # such that r = x % p.
    # Among all such p, we prefer one that covers the most OTHER uncovered integers.
    # This is a greedy choice for the "next" step.
    
    for x in range(1, L_max + 1):
        if time.time() > deadline:
            break
            
        if covered[x]:
            continue
            
        best_p = None
        best_r = None
        best_gain = -1
        best_cost = None
        
        for p in candidate_primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if current_sum + cost > 1:
                continue
            
            r = x % p
            
            # Calculate gain: how many uncovered integers y in [1, L_max] are covered by [p, r]?
            # y % p == r
            mask = (indices % p) == r
            # Count uncovered ones
            gain = np.sum(covered[1:L_max+1] & ~mask) # Wait, covered is True if covered.
            # We want count of UNCOVERED y where y%p==r.
            # uncovered_mask = ~covered[1:L_max+1]
            # gain = np.sum(uncovered_mask & mask)
            
            unc_mask = ~covered[1:L_max+1]
            gain = np.sum(unc_mask & mask)
            
            # Tie-breaker: prefer smaller p? Or larger gain?
            # We want to maximize gain.
            if gain > best_gain:
                best_gain = int(gain)
                best_p = p
                best_r = r
                best_cost = cost
        
        if best_p is None:
            break
            
        used_primes.add(best_p)
        current_sum += best_cost
        selected_classes.append([best_p, best_r])
        
        # Update covered
        mask = (indices % best_p) == best_r
        covered[1:L_max+1][mask] = True
        
    # Calculate L
    false_indices = np.where(~covered[1:L_max+1])[0]
    if len(false_indices) == 0:
        L = L_max
    else:
        L = false_indices[0]
        
    return selected_classes
