# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s limit
    
    # We want the largest subset of [1,100] with no subset summing to 200.
    # Use DP: dp[s] = max cardinality of a subset with sum exactly s (s < 200)
    # Process elements one by one, for each decide include/exclude.
    
    # dp[s] = max number of elements to achieve sum s, where s in [0, 199]
    # Initialize: dp[0] = 0, others = -infinity
    NEG = -10**9
    dp = [NEG] * 200
    dp[0] = 0
    
    for x in range(1, 101):
        if time.time() > deadline:
            break
        # Process in reverse to avoid using x multiple times
        new_dp = dp[:]  # option: don't include x
        for s in range(0, 200 - x):
            if dp[s] > NEG:
                new_s = s + x
                if new_s < 200:
                    if dp[s] + 1 > new_dp[new_s]:
                        new_dp[new_s] = dp[s] + 1
        dp = new_dp
    
    # Find max cardinality among all reachable sums < 200
    best_count = 0
    for s in range(200):
        if dp[s] > best_count:
            best_count = dp[s]
    
    # Now reconstruct the subset
    if best_count == 0:
        return [100]
    
    # Reconstruct: go backwards through elements
    # We need to track choices, so redo with tracking
    NEG = -10**9
    # dp[i][s] = max cardinality using elements 1..i with sum s
    # This is 101*200 = 20200 entries, fine
    dp_table = [[NEG] * 200 for _ in range(101)]
    dp_table[0][0] = 0
    
    for i in range(1, 101):
        x = i
        for s in range(200):
            # Don't include x
            dp_table[i][s] = dp_table[i-1][s]
            # Include x
            if s >= x:
                if dp_table[i-1][s-x] > NEG:
                    if dp_table[i-1][s-x] + 1 > dp_table[i][s]:
                        dp_table[i][s] = dp_table[i-1][s-x] + 1
    
    # Find best
    best_count = 0
    best_sum = 0
    for s in range(200):
        if dp_table[100][s] > best_count:
            best_count = dp_table[100][s]
            best_sum = s
    
    # Reconstruct
    selected = []
    s = best_sum
    for i in range(100, 0, -1):
        x = i
        # Check if x was included
        if s >= x and dp_table[i-1][s-x] + 1 == dp_table[i][s]:
            selected.append(x)
            s -= x
    
    if not selected:
        selected = [100]
    
    return selected
# EVOLVE-BLOCK-END
