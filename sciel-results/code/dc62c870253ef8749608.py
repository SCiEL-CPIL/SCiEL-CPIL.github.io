import time
import random

# EVOLVE-BLOCK-START
import sys

MAX_N = 512
MAX_SIZE = 48
MAX_SUM = MAX_SIZE * MAX_N  # 24576
MASK = (1 << (MAX_SUM + 1)) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = []
    
    # Strategies: Ascending, Descending, and Random
    strategies = []
    strategies.append(list(range(1, MAX_N + 1)))
    strategies.append(list(range(MAX_N, 0, -1)))
    
    # Add a few random shuffles
    for _ in range(10):
        s = list(range(1, MAX_N + 1))
        random.shuffle(s)
        strategies.append(s)
        
    # Also try a "greedy with backtracking" or just multiple greedy passes?
    # Greedy is usually sufficient if the order is good.
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in strategy:
            if time.time() > deadline:
                break
                
            if len(current_set) >= MAX_SIZE:
                break
                
            candidate_set = current_set + [x]
            
            if is_valid_set(candidate_set):
                current_set = candidate_set
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == MAX_SIZE:
                break # Found optimal
        
    return best_set

def is_valid_set(S):
    n = len(S)
    if n < 2:
        return True
        
    # Precompute dp for the whole set S
    # dp[c] is a bitset of sums reachable by c elements from S
    dp = [0] * (n + 1)
    dp[0] = 1  # Sum 0 reachable with 0 elements
    
    for u in S:
        # Update in reverse order of counts to avoid using u multiple times
        # We only need up to n elements
        for c in range(n, 0, -1):
            # dp[c] |= (dp[c-1] << u)
            # Apply mask to keep integer size bounded
            dp[c] |= (dp[c-1] << u) & MASK
            
    # Now check each element v
    for v in S:
        U_size = n - 1
        for k in range(2, U_size + 1):
            target = k * v
            if target > MAX_SUM:
                continue
                
            # Check if target is reachable by k elements from S
            # If not, it's definitely not reachable from S\{v}
            if not (dp[k] >> target) & 1:
                continue
                
            # If reachable from S, we must check if it's reachable from S\{v}
            # Run DP for S\{v}
            if not is_sum_reachable_excluding(S, v, k, target):
                continue
            else:
                # It IS reachable from S\{v}, so v is the mean. Invalid.
                return False
                
    return True

def is_sum_reachable_excluding(S, v, k, target):
    # Check if target can be formed by k distinct elements from S excluding v
    # Use bitset DP
    n_ex = len(S) - 1
    if k > n_ex:
        return False
        
    dp = [0] * (n_ex + 1)
    dp[0] = 1
    
    for u in S:
        if u == v:
            continue
        for c in range(n_ex, 0, -1):
            dp[c] |= (dp[c-1] << u) & MASK
            
    return bool((dp[k] >> target) & 1)

# EVOLVE-BLOCK-END
