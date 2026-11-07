"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Precomputation
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Map each prime to an index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor mask for each number 1..N
    # mask[n] is an integer where bit i is set if primes[i] divides n
    masks = [0] * (N + 1)
    for n in range(1, N + 1):
        m = n
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            # m is a prime factor > sqrt(original n)
            if m in prime_to_idx:
                mask_val |= (1 << prime_to_idx[m])
        masks[n] = mask_val
    
    # Create list of (value, mask) for numbers 1..N
    # We include 1 as well (mask 0)
    candidates = []
    for n in range(1, N + 1):
        candidates.append((n, masks[n]))
        
    # Sort candidates by value descending to improve bound tightness
    candidates.sort(key=lambda x: -x[0])
    
    # Precompute suffix sums of values for the "Sum of all compatible" bound?
    # Actually, the bound depends on the current mask.
    # Bound(current_idx, current_mask, current_val):
    #   Max possible additional value = sum of values of candidates[j] for j >= current_idx
    #   such that (candidates[j][1] & current_mask) == 0
    # This sum is an upper bound because it ignores conflicts among the remaining candidates themselves.
    
    # To speed up the bound calculation, we can't precompute for all masks.
    # But we can compute it on the fly.
    # Number of candidates is 256.
    # In the worst case, we iterate 256 times per node.
    # Depth is at most 256.
    # Branching factor is 2.
    # But pruning is key.
    
    best_sum = 0
    best_selection = []
    
    # We use a recursive B&B function
    # To avoid recursion depth issues and allow time checks, we can use a stack or just recursion with sys.setrecursionlimit
    sys.setrecursionlimit(10000)
    
    def bound(idx, mask):
        """Calculate upper bound on additional value from candidates[idx:] compatible with mask."""
        total = 0
        for j in range(idx, len(candidates)):
            val, cmask = candidates[j]
            if (cmask & mask) == 0:
                total += val
        return total
    
    def search(idx, mask, current_val, current_list):
        nonlocal best_sum, best_selection
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if current_val > best_sum:
            best_sum = current_val
            best_selection = current_list[:]
            
        # If no more candidates, return
        if idx >= len(candidates):
            return
            
        # Calculate upper bound
        # If current_val + bound <= best_sum, prune
        b_val = bound(idx, mask)
        if current_val + b_val <= best_sum:
            return
            
        # Option 1: Include candidates[idx] if compatible
        val, cmask = candidates[idx]
        if (cmask & mask) == 0:
            # Include
            search(idx + 1, mask | cmask, current_val + val, current_list + [val])
            
            # If we just hit the theoretical max possible (sum of all remaining compatible), we could stop, 
            # but the bound function already handles the "all compatible" sum.
            
        # Option 2: Exclude candidates[idx]
        search(idx + 1, mask, current_val, current_list)

    # Initial call
    # Note: The bound calculation inside search is O(N) per node.
    # With N=256, and good pruning, this should be fast.
    # However, Python recursion and loop overhead might be slow.
    # Optimization: We can precompute the list of candidates compatible with a given mask? No, mask changes.
    
    # Let's try to optimize the bound calculation.
    # Instead of summing all compatible, we can use a tighter bound if we sort by value.
    # The current bound is "Sum of all compatible values".
    
    # One potential issue: The search space might still be large if the bound is loose.
    # Let's add a heuristic: Greedy construction first to get a good lower bound (best_sum).
    
    # Greedy: Pick largest compatible number repeatedly
    greedy_mask = 0
    greedy_sum = 0
    greedy_set = []
    for val, cmask in candidates:
        if (cmask & greedy_mask) == 0:
            greedy_mask |= cmask
            greedy_sum += val
            greedy_set.append(val)
    
    best_sum = greedy_sum
    best_selection = greedy_set
    
    # Now run B&B
    # To prevent the time limit from being exceeded by deep unproductive branches,
    # we can add a node counter or rely on the time check.
    
    search(0, 0, 0, [])
    
    # Final result
    # The values in best_selection are the numbers themselves.
    # We need to return a list of ints.
    # Ensure distinct and sorted? The task says "distinct Python integers".
    # The search ensures distinctness because we process each candidate once.
    
    # Convert to list of ints
    result = [int(x) for x in best_selection]
    
    # Safety check: Verify pairwise coprime
    # This is O(K^2) where K is size of result. K is small (< 20).
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy if somehow invalid (should not happen)
        result = [int(x) for x in greedy_set]
        
    return result

# EVOLVE-BLOCK-END
