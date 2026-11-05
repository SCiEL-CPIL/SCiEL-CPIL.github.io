"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all distinct positive divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors in descending order for greedy-like heuristics
    divisors_desc = sorted(divisors, reverse=True)
    divisors_asc = sorted(divisors)

    # Greedy solution first (the seed approach)
    remainder = target
    chosen_greedy = []
    for d in divisors_desc:
        if d <= remainder:
            chosen_greedy.append(d)
            remainder -= d
        if remainder == 0:
            break
    assert remainder == 0, f"Greedy failed, remainder={remainder}"
    best_solution = list(chosen_greedy)
    best_len = len(best_solution)

    if best_len == 1:
        return best_solution

    # DP to find minimum number of divisors summing to target
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor)
    
    for d in divisors:  # iterate over each divisor once (0/1 knapsack style)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] != INF and dp[target] < best_len:
        # Reconstruct the solution
        solution = []
        s = target
        while s > 0:
            prev_s, d = parent[s]
            solution.append(d)
            s = prev_s
        solution.reverse()
        best_solution = solution
        best_len = len(solution)
    
    # Now try to find even shorter solutions using BFS/A* on remainder states
    # with a limit on the number of divisors used
    # Since we want to minimize count, we can do a BFS where each level adds one divisor
    # But the state space is large. Instead, let's try a bounded search.
    
    # Try to find a solution with k divisors where k < best_len
    # We can use a meet-in-the-middle approach for small k
    # Or we can use iterative deepening DFS with pruning
    
    # Let's try iterative deepening: for k = 1 to best_len-1, check if k divisors can sum to target
    # This is expensive, so we'll use a smart pruning strategy
    
    # First, let's see what the DP gave us
    if best_len <= 2:
        # Can't do much better than 2, and 1 is already checked
        pass
    else:
        # Try to find a solution with fewer terms using a different approach
        # We'll use a recursive search with memoization, limiting the depth
        
        # Sort divisors in descending order for better pruning
        divs_desc = sorted(divisors, reverse=True)
        
        # For iterative deepening, we need to be careful about the state space
        # Let's try k = best_len - 1, best_len - 2, ... down to 3
        # Use a set to track visited (remaining_sum, min_divisor_index) states to avoid cycles
        
        found_better = False
        for max_k in range(best_len - 1, 2, -1):
            if time.time() > deadline - 5:
                break
            
            # Try to find a solution with exactly max_k divisors
            # Use DFS with pruning: we need to pick max_k divisors from divs_desc
            # such that they sum to target
            
            # Pruning: the sum of the largest max_k divisors must be >= target
            # and the sum of the smallest max_k divisors must be <= target
            max_possible_sum = sum(divs_desc[:max_k])
            min_possible_sum = sum(divs_desc[-max_k:])
            
            if max_possible_sum < target or min_possible_sum > target:
                continue
            
            # DFS with memoization
            # State: (index_in_divs, remaining_sum, remaining_count)
            # We pick divisors in order to ensure distinctness and avoid permutations
            memo = set()
            found = [False]
            
            def dfs(idx, rem, count):
                if found[0]:
                    return
                if time.time() > deadline - 5:
                    found[0] = True  # signal timeout, treat as found to stop
                    return
                if count == 0:
                    if rem == 0:
                        found[0] = True
                    return
                if rem < 0:
                    return
                if idx >= len(divs_desc):
                    return
                
                # Pruning: even if we take the largest 'count' divisors from idx onward,
                # can we reach 'rem'?
                if count > len(divs_desc) - idx:
                    return
                
                # Upper bound: sum of largest 'count' divisors from idx
                # Lower bound: sum of smallest 'count' divisors from idx
                # For simplicity, just check if rem is achievable
                
                key = (idx, rem, count)
                if key in memo:
                    return
                memo.add(key)
                
                # Option 1: skip divs_desc[idx]
                dfs(idx + 1, rem, count)
                if found[0]:
                    return
                
                # Option 2: take divs_desc[idx]
                if divs_desc[idx] <= rem:
                    dfs(idx + 1, rem - divs_desc[idx], count - 1)
            
            dfs(0, target, max_k)
            
            if found[0] and time.time() < deadline - 5:
                # We found a solution with max_k divisors, but we need to reconstruct it
                # This is tricky with the DFS above. Let's do a different approach.
                # Actually, let's just trust the DP result for now if it's already good.
                pass
    
    # Verify the solution
    assert len(set(best_solution)) == len(best_solution), "Not distinct"
    assert all(5040 % d == 0 for d in best_solution), "Not divisors"
    assert sum(best_solution) == target, f"Sum mismatch: {sum(best_solution)}"
    assert len(best_solution) <= 256, "Too many divisors"
    
    return best_solution
# EVOLVE-BLOCK-END
