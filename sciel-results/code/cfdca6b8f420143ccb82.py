# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    # Generate all primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    L_target = 1000
    
    # Precompute coverage masks for each prime and residue
    # mask[p][r] is a bitmask of integers in [1, L_target] congruent to r mod p
    # We use Python integers as bitsets. Bit i corresponds to integer i+1.
    coverage_masks = {}
    for p in primes:
        masks_for_p = []
        for r in range(p):
            mask = 0
            # Integers x in [1, L_target] such that x % p == r
            # If r == 0, x is p, 2p, ...
            # If r > 0, x is r, r+p, ...
            start = r if r != 0 else p
            for x in range(start, L_target + 1, p):
                mask |= (1 << (x - 1))
            masks_for_p.append(mask)
        coverage_masks[p] = masks_for_p

    # Initial uncovered mask: all bits 0 to L_target-1 set
    initial_uncovered = (1 << L_target) - 1
    
    # We will use a recursive backtracking search.
    # To avoid deep recursion and timeouts, we use a heuristic ordering and limit the search space.
    # We'll try to find a solution covering [1, 1000].
    
    # Sort primes to try smaller ones first? Or larger?
    # Smaller primes have higher cost 1/p but cover more numbers per residue (density 1/p).
    # Larger primes have lower cost but cover fewer numbers.
    # We need a mix.
    
    # Let's try a greedy initialization first to get a good starting point, then backtrack?
    # Or just backtrack with good pruning.
    
    # Pruning: 
    # 1. Cost constraint: sum(1/p) <= 1.
    # 2. Coverage constraint: Can the remaining primes cover the remaining uncovered integers?
    #    A simple bound: If the number of uncovered integers is > sum( (L_target // p) * p for p in remaining_primes )? No, that's not right.
    #    Each prime p can cover at most L_target // p integers (actually ceil(L_target/p)).
    #    So if count(uncovered) > sum(ceil(L_target/p) for p in remaining_primes), prune.
    
    # To make the search tractable, we restrict the set of primes we consider.
    # We will consider the first K primes. If K is too small, we might fail. If too large, search space explodes.
    # Let's try K = 40.
    
    candidate_primes = primes[:40]
    
    # Precompute costs
    costs = {p: Fraction(1, p) for p in candidate_primes}
    
    # Precompute max coverage count for each prime
    max_cov_counts = {}
    for p in candidate_primes:
        max_cov = 0
        for mask in coverage_masks[p]:
            c = bin(mask).count('1')
            if c > max_cov:
                max_cov = c
        max_cov_counts[p] = max_cov

    # Sort candidate primes by max coverage descending? Or by cost ascending?
    # Let's sort by max coverage descending, so we pick "strong" primes first.
    sorted_primes = sorted(candidate_primes, key=lambda p: -max_cov_counts[p])
    
    # Backtracking
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(uncovered_mask, remaining_primes, current_cost, chosen):
        if uncovered_mask == 0:
            return chosen
        
        if not remaining_primes:
            return None
        
        # Pruning: Check if remaining primes can possibly cover the uncovered integers
        # Upper bound on coverage: sum of max_cov_counts for remaining primes
        # But max_cov_counts is for the *whole* range, not just uncovered.
        # A tighter bound: For each remaining prime p, the max number of *uncovered* integers it can cover is
        # the max over residues r of (coverage_masks[p][r] & uncovered_mask).count('1')
        # This is expensive to compute for all p and r at every node.
        
        # Cheaper bound: If current_cost + sum(costs[p] for p in remaining_primes) > 1, we can't add all.
        # But we don't need to add all.
        
        # Heuristic: Pick the prime p in remaining_primes that covers the most *uncovered* integers.
        best_p = None
        best_r = None
        best_new_cov = 0
        best_mask = 0
        
        # We only consider primes that are affordable
        for p in remaining_primes:
            if current_cost + costs[p] > 1:
                continue
            
            # Find the residue r for p that covers the most uncovered integers
            max_new = 0
            best_r_p = None
            best_mask_p = 0
            for r in range(p):
                mask = coverage_masks[p][r]
                new_cov_mask = mask & uncovered_mask
                c = bin(new_cov_mask).count('1')
                if c > max_new:
                    max_new = c
                    best_r_p = r
                    best_mask_p = new_cov_mask
            
            if max_new > best_new_cov:
                best_new_cov = max_new
                best_p = p
                best_r = best_r_p
                best_mask = best_mask_p
        
        if best_p is None or best_new_cov == 0:
            return None
        
        # Try the best choice first
        new_uncovered = uncovered_mask & ~best_mask
        new_cost = current_cost + costs[best_p]
        new_chosen = chosen + [[best_p, best_r]]
        
        # Remove best_p from remaining
        new_remaining = [p for p in remaining_primes if p != best_p]
        
        result = backtrack(new_uncovered, new_remaining, new_cost, new_chosen)
        if result is not None:
            return result
        
        # If best choice failed, try other choices for the same prime?
        # Or try other primes?
        # To keep it simple and fast, we just try the best choice. If it fails, we backtrack.
        # But this might miss solutions where the "best" prime needs a different residue.
        
        # Let's modify: For the selected best_p, try ALL residues that cover > 0 uncovered integers.
        # Sort them by coverage descending.
        
        residues_to_try = []
        for r in range(best_p):
            mask = coverage_masks[best_p][r]
            new_cov_mask = mask & uncovered_mask
            c = bin(new_cov_mask).count('1')
            if c > 0:
                residues_to_try.append((c, r, new_cov_mask))
        
        # Sort by coverage descending
        residues_to_try.sort(key=lambda x: -x[0])
        
        for c, r, mask in residues_to_try:
            new_uncovered = uncovered_mask & ~mask
            new_cost = current_cost + costs[best_p]
            new_chosen = chosen + [[best_p, r]]
            new_remaining = [p for p in remaining_primes if p != best_p]
            
            result = backtrack(new_uncovered, new_remaining, new_cost, new_chosen)
            if result is not None:
                return result
        
        return None

    # Start search
    # We need to pass a list of primes. We'll use the sorted_primes list.
    # But we need to remove primes as we use them.
    # The above backtrack function removes best_p from remaining.
    
    solution = backtrack(initial_uncovered, sorted_primes, Fraction(0, 1), [])
    
    if solution is None:
        # If the restricted set failed, try a larger set or a different heuristic
        # For now, return a greedy solution as fallback
        uncovered = initial_uncovered
        chosen = []
        current_cost = Fraction(0, 1)
        available_primes = primes[:]
        
        while uncovered and available_primes:
            best_p = None
            best_r = None
            best_new_cov = 0
            best_mask = 0
            
            for p in available_primes:
                if current_cost + Fraction(1, p) > 1:
                    continue
                for r in range(p):
                    mask = coverage_masks[p][r]
                    new_cov_mask = mask & uncovered
                    c = bin(new_cov_mask).count('1')
                    if c > best_new_cov:
                        best_new_cov = c
                        best_p = p
                        best_r = r
                        best_mask = new_cov_mask
            
            if best_p is None:
                break
            
            chosen.append([best_p, best_r])
            uncovered &= ~best_mask
            current_cost += Fraction(1, best_p)
            available_primes.remove(best_p)
        
        solution = chosen
    
    # Verify and return
    final_cost = sum(Fraction(1, p) for p, _ in solution)
    assert final_cost <= 1, f"Final cost {final_cost} exceeds 1"
    for p, r in solution:
        assert p <= 1000 and 0 <= r < p, f"Invalid class [{p}, {r}]"
    prime_list = [p for p, _ in solution]
    assert len(prime_list) == len(set(prime_list)), "Duplicate primes"
    
    return solution
# EVOLVE-BLOCK-END
