import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 512
    max_size = 48
    
    def is_valid_with_bitset(selected, candidate):
        """Check if adding candidate to selected creates any mean violation.
        Uses bitset DP for subset sums."""
        # Check: for each element a in selected + [candidate],
        # is a the mean of some subset of size >= 2 from the others?
        all_elements = selected + [candidate]
        n_elem = len(all_elements)
        
        for i, a in enumerate(all_elements):
            others = [x for j, x in enumerate(all_elements) if j != i]
            m = len(others)
            if m < 2:
                continue
            target = a  # We need sum(subset) = k * a for some k >= 2
            # Check all k from 2 to m
            # For each k, check if there's a subset of others of size k with sum = k*a
            # Use DP: dp[j][s] = whether we can pick j elements from others with sum s
            # But we need to check for each k separately
            # More efficient: for each k in 2..m, check if subset of size k has sum k*a
            
            # Bitset approach for each k:
            # We want to know if there exists a subset of 'others' of exactly size k with sum = k*a
            # Use DP where dp[j] is a bitset of achievable sums using exactly j elements
            
            max_sum = sum(others)
            if max_sum < 2 * a:
                continue
            
            # For each k from 2 to m, check if sum k*a is achievable with exactly k elements
            # We'll do a single DP pass tracking (count, sum) pairs
            # dp[j] = set of achievable sums with exactly j elements (using bitsets)
            
            # Actually, let's be smarter. We only need to check specific targets.
            # For k=2: need two elements x,y in others with x+y=2a, i.e., y=2a-x
            # For k=3: need three elements with sum=3a, etc.
            
            # Let's use a DP table: dp[j][s] = True if we can pick j elements with sum s
            # But this could be expensive. Let's use bitsets per count.
            
            # dp[j] is a bitset where bit s is set if sum s is achievable with j elements
            dp = [0] * (m + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for val in others:
                # Process in reverse to avoid using same element twice
                for j in range(m, 0, -1):
                    dp[j] |= (dp[j-1] << val)
            
            # Now check: for k=2..m, is bit (k*a) set in dp[k]?
            for k in range(2, m + 1):
                target_sum = k * a
                if target_sum <= max_sum and (dp[k] >> target_sum) & 1:
                    return False
        
        return True
    
    def greedy_build(order, deadline):
        """Greedily build a valid set following the given order."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= max_size:
                break
            if is_valid_with_bitset(selected, x):
                selected.append(x)
        return selected
    
    # Strategy: try multiple random orderings
    best = []
    
    # First, try a few structured orderings
    # Ordering 1: ascending
    order1 = list(range(1, n + 1))
    result = greedy_build(order1, deadline)
    if len(result) > len(best):
        best = result
    
    # Ordering 2: descending
    order2 = list(range(n, 0, -1))
    result = greedy_build(order2, deadline)
    if len(result) > len(best):
        best = result
    
    # Ordering 3: try to pick numbers that are "harder" to be means
    # Numbers near the extremes are harder to be means of other numbers
    # Try: alternate from ends
    order3 = []
    lo, hi = 1, n
    while lo <= hi:
        order3.append(lo)
        if lo != hi:
            order3.append(hi)
        lo += 1
        hi -= 1
    result = greedy_build(order3, deadline)
    if len(result) > len(best):
        best = result
    
    # Now try random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        order = list(range(1, n + 1))
        random.shuffle(order)
        result = greedy_build(order, deadline)
        if len(result) > len(best):
            best = result
            # If we reached max_size, we can stop
            if len(best) >= max_size:
                break
    
    return best[:max_size]
