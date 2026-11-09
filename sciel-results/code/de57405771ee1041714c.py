import time
from functools import lru_cache

def _get_divisors(n):
    """Generate all divisors of n, sorted in descending order."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 5040
    target = 4979
    
    divisors = _get_divisors(n)
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    if num_divs == 0:
        return []
        
    # Prefix sums for max_sum_from
    prefix_sums = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sums[i + 1] = prefix_sums[i] + divisors[i]
    
    # Precompute min_sum_from[start_idx][count] properly
    # divisors is sorted descending, so divisors[start_idx:] has its smallest elements at the end
    # min_sum_from(s, c) = sum of c smallest elements in divisors[s:]
    # = sum of divisors[num_divs-c : num_divs] if num_divs-c >= s, else infeasible
    # Actually the c smallest in divisors[s:] are divisors[max(s, num_divs-c) : num_divs]
    # Wait no. divisors[s:] = [divisors[s], divisors[s+1], ..., divisors[num_divs-1]]
    # sorted descending. The c smallest are the last c: divisors[num_divs-c] to divisors[num_divs-1]
    # but only if num_divs-c >= s. Otherwise infeasible.
    
    # Precompute table: min_table[s][c] for s in range, c in range
    # This is O(num_divs^2) which is 60^2 = 3600, fine.
    INF = float('inf')
    min_table = [[INF] * (num_divs + 1) for _ in range(num_divs + 1)]
    for s in range(num_divs + 1):
        for c in range(num_divs + 1):
            if c == 0:
                min_table[s][c] = 0
            elif s + c > num_divs:
                min_table[s][c] = INF
            else:
                # c smallest from divisors[s:] are divisors[num_divs-c : num_divs]
                # but they must all be >= s, i.e., num_divs-c >= s
                if num_divs - c >= s:
                    min_table[s][c] = prefix_sums[num_divs] - prefix_sums[num_divs - c]
                else:
                    min_table[s][c] = INF
    
    # max_table[s][c] = sum of c largest from divisors[s:] = prefix_sums[s+c] - prefix_sums[s]
    max_table = [[-1] * (num_divs + 1) for _ in range(num_divs + 1)]
    for s in range(num_divs + 1):
        for c in range(num_divs + 1):
            if c == 0:
                max_table[s][c] = 0
            elif s + c > num_divs:
                max_table[s][c] = -1
            else:
                max_table[s][c] = prefix_sums[s + c] - prefix_sums[s]
    
    # Memoization for failed states
    failed = set()
    
    def dfs(start_idx, remaining, count, selected):
        """
        Try to find 'count' more divisors from divisors[start_idx:] summing to 'remaining'.
        Returns list of selected divisors if found, else None.
        """
        if time.time() > deadline:
            return None
        
        if count == 0:
            if remaining == 0:
                return selected[:]
            return None
        
        if remaining < 0 or remaining == 0:
            return None
        
        if num_divs - start_idx < count:
            return None
        
        # Check memo
        state = (start_idx, remaining, count)
        if state in failed:
            return None
        
        # Bounds
        if remaining > max_table[start_idx][count]:
            failed.add(state)
            return None
        
        if remaining < min_table[start_idx][count]:
            failed.add(state)
            return None
        
        # Search
        for i in range(start_idx, num_divs):
            d = divisors[i]
            
            if d > remaining:
                continue
            
            new_remaining = remaining - d
            new_count = count - 1
            new_start = i + 1
            
            if new_count == 0:
                if new_remaining == 0:
                    return selected + [d]
                continue
            
            # Bounds check for next level
            if new_remaining > max_table[new_start][new_count]:
                # d is decreasing, so new_remaining is increasing.
                # max_table[new_start][new_count] is decreasing as i increases (new_start increases).
                # So break.
                break
            if new_remaining < min_table[new_start][new_count]:
                continue
            
            selected.append(d)
            result = dfs(new_start, new_remaining, new_count, selected)
            selected.pop()
            
            if result is not None:
                return result
        
        failed.add(state)
        return None

    best_result = None
    for k in range(1, 257):
        if time.time() > deadline:
            break
        failed.clear()
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = [2520, 1680, 560, 210, 9]
        
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    assert all(n % d == 0 for d in best_result)
    assert len(best_result) <= 256
    
    return best_result
