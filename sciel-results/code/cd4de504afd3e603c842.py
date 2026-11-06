"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s margin
    
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N that are > 12
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divs = get_divisors(N)
    candidates = [d for d in all_divs if d > 12]
    candidates.sort()
    n_cands = len(candidates)
    
    # For efficiency, use a dictionary for quick lookup
    cand_set = set(candidates)
    
    best_max = N  # worst case: single factor
    best_factors = [N]
    
    # We want to find subsets of candidates (distinct) with product = N
    # and minimize the maximum element in the subset
    # Subset size: 1 to 12
    
    # Strategy: for each possible max value M (from smallest candidate upward),
    # check if there's a subset of candidates all <= M, distinct, product = N, size 1-12
    # Binary search or linear scan on M
    
    # Better: DFS with pruning, trying to build factors in increasing order
    # and tracking the product so far
    
    # Precompute: for each candidate, its value
    # Use recursive search
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For pruning: given current product p and remaining target N/p,
    # and minimum allowed factor (must be > last chosen, and > 12),
    # and max factors remaining
    
    # Let's do a DFS that picks factors in increasing order
    # State: (start_idx, remaining_product, factors_so_far)
    # We want remaining_product to become 1
    
    # Pruning:
    # 1. remaining_product must be >= 1
    # 2. If remaining_product == 1, we're done
    # 3. The number of factors we can still add is limited by (12 - len(factors_so_far))
    # 4. Each remaining factor must be > last chosen (for distinctness, increasing order)
    # 5. Each remaining factor must be > 12
    # 6. The product of the smallest possible remaining factors must be <= remaining_product
    #    and the product of the largest possible must be >= remaining_product
    
    # For efficiency, precompute prefix products of candidates from each index
    # Actually, let's just do a well-pruned DFS
    
    # To minimize the maximum, we can do:
    # For each candidate max M (in increasing order), check if a valid subset exists
    # with all elements <= M. First such M is optimal.
    
    # But checking "does a subset exist with product N, all <= M, size 1-12" is still hard.
    # Instead, let's do DFS that builds increasing sequences and tracks the max.
    
    # Alternative: just do DFS and keep track of the best (smallest max) found.
    
    def dfs(start_idx, remaining, factors):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid decomposition
            cur_max = factors[-1] if factors else 0
            if cur_max < best_max:
                best_max = cur_max
                best_factors = list(factors)
            return
        
        if len(factors) >= 12:
            return
        
        # Pruning: remaining must be divisible by some combination of candidates
        # The smallest factor we can pick next is candidates[start_idx] (if start_idx < n_cands)
        # Each subsequent factor must be > the previous
        
        # Lower bound on number of factors needed:
        # If we pick the largest possible factors, how many do we need?
        # Upper bound on number of factors we can still pick: 12 - len(factors)
        
        # Pruning: remaining must be >= 1
        # If remaining < candidates[start_idx] and start_idx < n_cands, we can't pick any more
        # But we need remaining == 1, so if remaining > 1 and we can't pick anything, fail
        if remaining > 1:
            if start_idx >= n_cands:
                return
            # The smallest factor we can pick is candidates[start_idx]
            # If candidates[start_idx] > remaining, we can't pick it (would overshoot)
            # But we could potentially skip it... wait, we're going in order, so we either pick or skip
            # Actually in this DFS, we're picking a subsequence, not necessarily consecutive
            # Let me restructure: at each step, try each candidate from start_idx onward
        
        # Try each candidate from start_idx to n_cands-1
        # But we need candidate * (something) = remaining, so candidate must divide remaining
        # And candidate > 12 (already ensured)
        # And candidate > factors[-1] if factors (ensured by start_idx > last chosen index)
        
        # Optimization: for each candidate i from start_idx, if candidates[i] divides remaining
        # and candidates[i] <= remaining (since all factors > 1)
        
        # Also: after picking candidates[i], the remaining product is remaining // candidates[i]
        # and we continue from i+1
        
        # Additional pruning: the product of the smallest (12 - len(factors) - 1) candidates
        # after i must be <= remaining // candidates[i] (if we need more factors)
        # And if we stop after this pick, remaining // candidates[i] must == 1
        
        for i in range(start_idx, n_cands):
            c = candidates[i]
            if c > remaining:
                break  # candidates are sorted, so all subsequent are also > remaining
            if remaining % c != 0:
                continue
            
            new_remaining = remaining // c
            factors.append(c)
            
            # Check if new_remaining == 1
            if new_remaining == 1:
                cur_max = c
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = list(factors)
                factors.pop()
                continue
            
            # Continue DFS
            if len(factors) < 12:
                dfs(i + 1, new_remaining, factors)
            
            factors.pop()
            
            if time.time() > deadline:
                return
        
        # Also consider: what if we don't pick any more factors?
        # That's handled by the remaining == 1 check at the top
        
    # Start DFS
    dfs(0, N, [])
    
    return best_factors

# EVOLVE-BLOCK-END
