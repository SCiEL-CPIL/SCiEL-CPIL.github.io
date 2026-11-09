import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    if not nums:
        return 1
    return reduce(_lcm, nums)

def _is_valid(S, deadline=None):
    """
    Check if S is reciprocal-subsum-free.
    Returns True if valid, False if not, None if timeout.
    """
    if not S:
        return True
    
    S_list = list(S)
    # Check smaller x first, as they have larger targets L/x and are more likely to be sums of others
    S_sorted = sorted(S_list)
    
    L = _lcm_all(S_list)
    
    for x in S_sorted:
        if deadline is not None and time.time() > deadline:
            return None
        
        target = L // x
        others = [y for y in S_list if y != x]
        if not others:
            continue
            
        vals = [L // y for y in others]
        
        # Quick check: if sum of all vals < target, it's impossible
        total_sum = sum(vals)
        if total_sum < target:
            continue
            
        # DP for subset sum
        # We only care about sums <= target
        achievable = {0}
        for v in vals:
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Helper to update best
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(best)
            
    # Strategy: Multiple restarts with Greedy Construction + Local Search
    # We will try different random seeds/orders
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        current = set()
        
        # 1. Greedy Construction
        # Pick elements in a random order, but with a bias towards smaller numbers? 
        # Or just random.
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        
        for elem in candidates:
            if time.time() > deadline:
                break
            candidate = current | {elem}
            if _is_valid(candidate, deadline) is True:
                current = candidate
                
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
            
        # 2. Local Search (Hill Climbing)
        # Try to add elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element
            for elem in range(1, 25):
                if elem in current:
                    continue
                candidate = current | {elem}
                if _is_valid(candidate, deadline) is True:
                    current = candidate
                    improved = True
                    break # Restart loop as set changed
                    
            if improved:
                continue
                
            # Try swapping: remove x, add y
            # To avoid infinite loops, only accept if size increases or (size same and random choice for diversity)
            # Since we want max size, we only accept size increases.
            # If we are stuck at size K, we can't increase size with a single swap unless we remove 1 and add 1 (impossible) 
            # Wait, swap keeps size constant. So local search with swaps doesn't increase size.
            # We need a different move: Add 2 elements, remove 1? Or just rely on restarts.
            
            # Let's try a "2-for-1" move: Remove x, add y and z?
            # This is expensive.
            
            # Alternative: If we can't add any single element, we are in a local optimum for this construction path.
            # Break and restart with a different random order.
            break
            
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
            
    # If we have time, do a more exhaustive check on the best set found?
    # Or try to construct from scratch with a specific heuristic:
    # Prioritize adding numbers that are "hard" to be formed by others?
    # Small numbers are easy to form (e.g., 1/2 = 1/3 + 1/6). 
    # Large numbers are hard to form.
    # So maybe greedy from large to small is better?
    
    if time.time() < deadline - 5:
        # Try greedy from 24 down to 1
        current = set()
        for elem in range(24, 0, -1):
            if time.time() > deadline:
                break
            candidate = current | {elem}
            if _is_valid(candidate, deadline) is True:
                current = candidate
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
            
        # Try greedy from 1 to 24
        current = set()
        for elem in range(1, 25):
            if time.time() > deadline:
                break
            candidate = current | {elem}
            if _is_valid(candidate, deadline) is True:
                current = candidate
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
            
    # Final verification
    if best_set:
        if _is_valid(best_set) is True:
            return sorted(best_set)
        else:
            # Fallback: return a known valid small set if best fails (should not happen)
            return [24]
            
    return [24]

# Ensure reproducibility for debugging, but random for search
# random.seed(42) # Uncomment for deterministic debugging
