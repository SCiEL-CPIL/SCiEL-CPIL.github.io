"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Build suffix sums for pruning: suffix_sum[i] = sum of divisors[i:]
    m = len(divisors)
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Try to find minimum number of distinct divisors summing to target
    # Use iterative deepening DFS with pruning
    
    def dfs(remaining, start_idx, chosen, depth):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return chosen[:]
        if len(chosen) >= depth:
            return None
        # Prune: if sum of all remaining divisors < remaining, impossible
        if suffix_sum[start_idx] < remaining:
            return None
        # Prune: even if we take the largest (depth - len(chosen)) divisors from start_idx,
        # can we reach remaining?
        slots_left = depth - len(chosen)
        if slots_left <= 0:
            return None
        # Upper bound: take the largest slots_left divisors from start_idx onward
        # that are <= remaining
        upper = 0
        count = 0
        for i in range(start_idx, m):
            if count >= slots_left:
                break
            if divisors[i] <= remaining:
                upper += divisors[i]
                count += 1
        if upper < remaining:
            return None
        
        for i in range(start_idx, m):
            d = divisors[i]
            if d > remaining:
                continue
            # Prune: if even taking this divisor plus the best (slots_left-1) others can't reach remaining
            new_remaining = remaining - d
            if new_remaining == 0:
                if len(chosen) + 1 <= depth:
                    return chosen + [d]
                continue
            # Check if we can still reach new_remaining with slots_left-1 more divisors
            if suffix_sum[i + 1] < new_remaining:
                continue
            chosen.append(d)
            result = dfs(new_remaining, i + 1, chosen, depth)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    # Try increasing depths
    for depth in range(1, 15):
        result = dfs(target, 0, [], depth)
        if result is not None:
            return result
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen

# EVOLVE-BLOCK-END
