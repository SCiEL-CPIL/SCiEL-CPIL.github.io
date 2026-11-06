"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors descending to prioritize larger values (helps minimize count)
    divisors.sort(reverse=True)
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    # Use dict for sparse representation
    dp = {0: 0}
    # parent tracking for reconstruction
    parent = {}  # (sum, divisor_index) -> (prev_sum, prev_count)
    
    for i, d in enumerate(divisors):
        if d > target:
            continue
        # Iterate over current states in reverse to avoid using same divisor twice
        # Collect current keys first
        current_states = list(dp.items())
        for s, cnt in current_states:
            new_s = s + d
            if new_s > target:
                continue
            if new_s not in dp or cnt + 1 < dp[new_s]:
                dp[new_s] = cnt + 1
                parent[new_s] = (s, d)
    
    if target not in dp:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct the solution
    chosen = []
    cur = target
    while cur > 0:
        prev_s, d = parent[cur]
        chosen.append(d)
        cur = prev_s
    
    return chosen
# EVOLVE-BLOCK-END
