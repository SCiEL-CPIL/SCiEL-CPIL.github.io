"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

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
    
    # BFS on sum states: level k means exactly k divisors used
    # visited[s] = True if sum s is achievable with some number of divisors
    # parent[s] = (prev_sum, divisor) for reconstruction
    visited = {0: True}
    parent = {}  # s -> (prev_s, d)
    
    # Current frontier of sums achievable with exactly k divisors
    current_frontier = [0]
    k = 0
    
    while current_frontier:
        if time.time() - start_time > DEADLINE:
            break
        
        k += 1
        next_frontier = []
        
        for s in current_frontier:
            for d in divisors:
                if d > target - s:
                    continue  # since divisors are sorted desc, no point continuing
                new_s = s + d
                if new_s > target:
                    continue
                if not visited.get(new_s, False):
                    visited[new_s] = True
                    parent[new_s] = (s, d)
                    next_frontier.append(new_s)
                    if new_s == target:
                        # Found target at level k
                        # Reconstruct the solution
                        chosen = []
                        cur = target
                        while cur > 0:
                            prev_s, d = parent[cur]
                            chosen.append(d)
                            cur = prev_s
                        return chosen
        
        current_frontier = next_frontier
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
# EVOLVE-BLOCK-END
