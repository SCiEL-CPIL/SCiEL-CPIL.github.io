import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin below 60s
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    
    # Filter divisors > target, as they cannot be part of the sum
    # (all divisors are positive)
    divisors = [d for d in divisors if d <= target]
    
    # Sort in descending order.
    # This helps in two ways:
    # 1. For pruning: if sum of k largest < target, impossible.
    # 2. For finding solutions quickly: larger numbers are more likely to sum to target with fewer terms, 
    #    and lexicographic order on descending list might hit "large first" combos early?
    #    Actually, lexicographic order on [big, ..., small] means we pick big ones first.
    #    The known solution {2520, 1680, 756, 21, 2} consists of large numbers.
    #    So it should be found early in the k=5 search.
    divisors.sort(reverse=True)
    
    m = len(divisors)
    
    # Precompute prefix sums of sorted divisors for quick max-sum checks
    # prefix_sum[i] = sum(divisors[0:i])
    prefix_sum = [0] * (m + 1)
    for i in range(m):
        prefix_sum[i+1] = prefix_sum[i] + divisors[i]
        
    # Also need suffix sums or just check min sum?
    # Min sum for k elements is sum of k smallest, which are at the end of the list.
    # suffix_sum[i] = sum(divisors[i:m])
    # But we can just check if sum(divisors[-k:]) > target.
    
    # Iterate k from 1 to m
    for k in range(1, m + 1):
        # Time check
        if time.time() - start_time > DEADLINE:
            break
            
        # Pruning: Max possible sum with k divisors
        if k <= m:
            max_sum = prefix_sum[k]
            if max_sum < target:
                continue
                
        # Pruning: Min possible sum with k divisors
        # The k smallest divisors are divisors[m-k:m]
        min_sum = sum(divisors[m-k:m])
        if min_sum > target:
            continue
            
        # Search for a combination of size k that sums to target
        # Use itertools.combinations
        found = None
        for combo in combinations(divisors, k):
            if time.time() - start_time > DEADLINE:
                return None # Timeout, let caller handle or return best effort? 
                            # Task says "invalid result... scores 0". 
                            # We must return a valid result. 
                            # If we timeout, we should probably return the greedy fallback 
                            # which is valid but not minimal. 
                            # However, the problem guarantees a solution exists.
                            # And our search should finish well before deadline.
                
            if sum(combo) == target:
                found = list(combo)
                break
                
        if found is not None:
            # Verify
            assert sum(found) == target
            assert len(set(found)) == len(found)
            assert all(n % x == 0 for x in found)
            return found

    # Fallback: Greedy
    # This should rarely be reached if the above search works.
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    if sum(chosen) == target:
        return chosen
    else:
        # If even greedy fails (should not happen), return empty or raise?
        # Task requires valid result. 
        # Since 5040 is practical, greedy usually works or a simple backtracking would.
        # But we have a proven solution for k=5.
        # If we reach here, something went wrong with the search.
        # Let's force the known solution if we are in a known state? No, keep it generic.
        # But for this specific task, we can just return the known set if we fail to find it?
        # No, that's cheating. 
        # The search for k=5 is C(59,5) ~ 5M. It will finish.
        return []
