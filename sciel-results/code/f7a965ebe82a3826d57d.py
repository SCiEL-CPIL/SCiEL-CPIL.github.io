"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Filter to only those <= target (since all positive, any divisor > target can't be in solution)
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # First: try small k with pruned recursive search
    # k=1
    if target in divisors:
        return [target]
    
    # Try k=2,3,4,5,6 with pruning
    for k in range(2, 7):
        if time.time() > deadline:
            break
        best_result = [None]
        
        def search(start, remaining_k, remaining_sum, chosen):
            if time.time() > deadline:
                return False
            if remaining_k == 0:
                if remaining_sum == 0:
                    best_result[0] = list(chosen)
                    return True
                return False
            if start + remaining_k > m:
                return False
            # Max possible sum from remaining
            max_possible = suffix_sum[start] - suffix_sum[start + remaining_k]
            if max_possible < remaining_sum:
                return False
            # Min possible sum: pick remaining_k smallest from divisors[start:]
            # The smallest remaining_k elements in divisors[start:] are divisors[m-remaining_k:m]
            # but only if m - remaining_k >= start
            if m - remaining_k < start:
                return False
            min_possible = suffix_sum[m - remaining_k]
            if min_possible > remaining_sum:
                return False
            
            for i in range(start, m - remaining_k + 1):
                d = divisors[i]
                if d > remaining_sum:
                    continue
                # Lower bound: remaining_sum - d must be achievable with remaining_k-1 from divisors[i+1:]
                rem_k = remaining_k - 1
                if rem_k == 0:
                    if remaining_sum - d == 0:
                        best_result[0] = list(chosen) + [d]
                        return True
                    continue
                # Check if we can fill remaining_sum - d with rem_k items from divisors[i+1:]
                # Max sum from divisors[i+1:] picking rem_k
                max_rem = suffix_sum[i+1] - suffix_sum[i+1 + rem_k]
                if max_rem < remaining_sum - d:
                    continue
                # Min sum from divisors[i+1:] picking rem_k
                if m - rem_k < i + 1:
                    continue
                min_rem = suffix_sum[m - rem_k]
                if min_rem > remaining_sum - d:
                    continue
                
                chosen.append(d)
                if search(i + 1, rem_k, remaining_sum - d, chosen):
                    return True
                chosen.pop()
        
        search(0, k, target, [])
        if best_result[0] is not None:
            return best_result[0]
    
    # For k >= 7, use DP approach
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [(-1, -1)] * (target + 1)
    dp[0] = 0
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    best_count = dp[target]
    best_solution = None
    
    if best_count < INF:
        s = target
        solution = []
        while s > 0:
            prev_s, d = parent[s]
            solution.append(d)
            s = prev_s
        best_solution = solution
        best_count = len(best_solution)
    
    if best_solution is None:
        # Greedy fallback
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        best_solution = chosen
        best_count = len(chosen)
    
    # Local search: try to reduce solution size by merging pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_solution)
        
        for i, a in enumerate(best_solution):
            for b in best_solution[i+1:]:
                s = a + b
                if s in divisors and s not in current:
                    new_solution = [x for x in best_solution if x != a and x != b]
                    new_solution.append(s)
                    if len(set(new_solution)) == len(new_solution) and sum(new_solution) == target:
                        if len(new_solution) < len(best_solution):
                            best_solution = new_solution
                            best_count = len(best_solution)
                            improved = True
                            break
            if improved:
                break
    
    # Final verification
    assert sum(best_solution) == target
    assert len(set(best_solution)) == len(best_solution)
    assert all(x in divisors for x in best_solution)
    assert len(best_solution) <= 256
    
    return best_solution
# EVOLVE-BLOCK-END
