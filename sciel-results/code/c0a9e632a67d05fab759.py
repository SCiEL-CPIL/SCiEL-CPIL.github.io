"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort in descending order to potentially find solutions faster
    divisors.sort(reverse=True)
    m = len(divisors)  # 60 divisors
    
    # Binary search on the number of divisors k
    # Check if we can represent target as sum of exactly k distinct divisors
    
    def can_represent(k):
        """Check if target can be represented as sum of exactly k distinct divisors."""
        # DP: dp[i][j][s] = whether we can pick j divisors from first i divisors to sum to s
        # But this is too large. Use a different approach.
        
        # For small k, we can use meet-in-the-middle or DFS with pruning.
        # For k up to ~10, DFS with pruning should work.
        
        # Actually, let's use a DP approach:
        # dp[j][s] = set of possible states, but that's expensive.
        
        # Better: use the fact that we have only 60 divisors.
        # For a fixed k, we can use a recursive search with memoization.
        
        # State: (index, remaining_count, remaining_sum)
        # We want to know if we can pick remaining_count divisors from divisors[index:] 
        # that sum to remaining_sum.
        
        from functools import lru_cache
        
        # Pruning: the maximum sum we can get from picking r divisors from divisors[i:]
        # is the sum of the r largest divisors in divisors[i:]
        # The minimum sum is the sum of the r smallest divisors in divisors[i:]
        
        # Precompute suffix sums of largest r and smallest r
        # Actually, for each index i and count r, precompute max and min possible sums
        
        # This is getting complex. Let's just do a DFS with pruning for small k.
        
        # Sort divisors in descending order (already done)
        
        @lru_cache(maxsize=None)
        def dfs(idx, count, rem):
            if count == 0:
                return rem == 0
            if idx >= m:
                return False
            if rem < 0:
                return False
            
            # Pruning: if rem > sum of largest 'count' divisors from idx onwards
            # or rem < sum of smallest 'count' divisors from idx onwards, return False
            
            # Quick check: if rem is too large
            if count > 0 and idx + count > m:
                return False
            
            # Try taking divisor idx
            if dfs(idx + 1, count - 1, rem - divisors[idx]):
                return True
            
            # Try skipping divisor idx
            if dfs(idx + 1, count, rem):
                return True
            
            return False
        
        result = dfs(0, k, target)
        dfs.cache_clear()
        return result
    
    # Binary search for minimum k
    # Lower bound: 1 (if target is a divisor)
    # Upper bound: m (all divisors)
    
    # First check if target is a single divisor
    if target in divisors:
        return [target]
    
    # Binary search
    lo, hi = 2, m
    best_k = None
    best_solution = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        if can_represent(mid):
            best_k = mid
            best_solution = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_k is None:
        # Fallback: use the original 0-1 BFS approach
        INF = float('inf')
        dist = [[INF] * (m + 1) for _ in range(target + 1)]
        dq = deque()
        dist[0][0] = 0
        dq.append((0, 0))
        
        while dq:
            s, i = dq.popleft()
            cur_cost = dist[s][i]
            
            if time.time() > deadline:
                break
            
            if s == target:
                continue
            
            if i >= m:
                continue
            
            if cur_cost < dist[s][i + 1]:
                dist[s][i + 1] = cur_cost
                dq.appendleft((s, i + 1))
            
            new_s = s + divisors[i]
            if new_s <= target:
                if cur_cost + 1 < dist[new_s][i + 1]:
                    dist[new_s][i + 1] = cur_cost + 1
                    dq.append((new_s, i + 1))
        
        min_cost = INF
        best_i = -1
        for i in range(m + 1):
            if dist[target][i] < min_cost:
                min_cost = dist[target][i]
                best_i = i
        
        if min_cost == INF:
            remainder = target
            chosen = []
            for d in divisors:
                if d <= remainder:
                    chosen.append(d)
                    remainder -= d
            assert remainder == 0, f"Greedy failed, remainder={remainder}"
            return chosen
        
        s = target
        i = best_i
        chosen = []
        while i > 0:
            prev_i = i - 1
            d = divisors[prev_i]
            if s >= d and dist[s - d][prev_i] == dist[s][i] - 1:
                chosen.append(d)
                s = s - d
            i = prev_i
        
        chosen.reverse()
        assert len(set(chosen)) == len(chosen), "Not distinct"
        assert all(n % d == 0 for d in chosen), "Not all divisors of 5040"
        assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
        assert len(chosen) <= 256
        return chosen
    
    # Now we know the minimum k. Find the actual solution.
    # Use DFS to find one solution with exactly best_k divisors
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def find_solution(idx, count, rem):
        """Return a list of divisors or None."""
        if count == 0:
            return [] if rem == 0 else None
        if idx >= m:
            return None
        if rem < 0:
            return None
        if count > m - idx:
            return None
        
        # Try taking divisor idx
        sub = find_solution(idx + 1, count - 1, rem - divisors[idx])
        if sub is not None:
            return [divisors[idx]] + sub
        
        # Try skipping divisor idx
        sub = find_solution(idx + 1, count, rem)
        if sub is not None:
            return sub
        
        return None
    
    solution = find_solution(0, best_k, target)
    find_solution.cache_clear()
    
    if solution is None:
        # Should not happen, but fallback
        return [target]
    
    # Verify
    assert len(set(solution)) == len(solution), "Not distinct"
    assert all(n % d == 0 for d in solution), "Not all divisors of 5040"
    assert sum(solution) == target, f"Sum mismatch: {sum(solution)} != {target}"
    assert len(solution) <= 256
    
    return solution
# EVOLVE-BLOCK-END
