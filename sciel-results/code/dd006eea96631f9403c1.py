import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
            
    # Sort descending
    divisors.sort(reverse=True)
    N = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_sums[i][k] is the max sum achievable using k divisors from indices i to N-1
    # Since we want to maximize the sum to check if we can reach target, 
    # we just take the largest k available.
    # Actually, simply summing the largest k available from index i is sufficient for the upper bound check.
    
    # We will try k = 1, 2, ...
    # Max k is N, but likely small.
    
    # Optimization: 
    # If target is in divisors and k=1, return [target]
    # We can just run the search for k=1, 2, ...
    
    # To speed up, we can check if target - d is possible with k-1.
    # We use a recursive DFS with memoization? 
    # State: (start_idx, k, target)
    # Memo key: (start_idx, k, target)
    # This might be large. N=60, K small, Target up to 5000.
    # 60 * 10 * 5000 = 3,000,000 states. Manageable.
    
    from functools import lru_cache
    
    # Clear cache for each k? Or just use a dict.
    # Since we are doing iterative deepening, we can clear the cache for each k or use a global one if we ensure correctness.
    # Actually, for a fixed k, the subproblems are independent of the specific path taken to reach the state (start_idx, k, target) ONLY IF we don't care about the specific set of used divisors.
    # But we DO care about distinctness.
    # The state (start_idx, k, target) implies we are choosing from divisors[start_idx:].
    # This implicitly ensures distinctness because we only pick from indices > start_idx.
    # So yes, memoization on (start_idx, k, target) is valid!
    
    def solve(k):
        @lru_cache(maxsize=None)
        def dfs(start_idx, k_rem, target_rem):
            if k_rem == 0:
                return target_rem == 0
            if k_rem < 0 or target_rem < 0:
                return False
            
            # Pruning: Max possible sum with k_rem divisors from start_idx
            # We need the sum of the largest k_rem divisors in divisors[start_idx:]
            # Since divisors are sorted descending, they are divisors[start_idx : start_idx + k_rem]
            if start_idx + k_rem > N:
                return False
            
            # Sum of largest k_rem available
            max_sum = sum(divisors[start_idx:start_idx+k_rem])
            if max_sum < target_rem:
                return False
            
            # Pruning: Min possible sum? 
            # Min sum is sum of smallest k_rem? No, we just need to reach target.
            # If target_rem is smaller than the smallest divisor, we can't pick it?
            # Actually, if target_rem < divisors[start_idx + k_rem - 1] (smallest in the window of k_rem largest?), no.
            # We can pick any subset.
            
            # Iterate
            for i in range(start_idx, N - k_rem + 1):
                d = divisors[i]
                if d > target_rem:
                    continue
                # Check if we can form target_rem - d with k_rem - 1 divisors from i+1 onwards
                if dfs(i + 1, k_rem - 1, target_rem - d):
                    # Reconstruct path? 
                    # For now, just return True. 
                    # We need the actual divisors.
                    # We can store the choice.
                    return True
            return False
        
        # We need to return the actual list.
        # Let's modify dfs to return the list or None.
        
        @lru_cache(maxsize=None)
        def dfs_path(start_idx, k_rem, target_rem):
            if k_rem == 0:
                if target_rem == 0:
                    return []
                else:
                    return None
            if k_rem < 0 or target_rem < 0:
                return None
            
            if start_idx + k_rem > N:
                return None
                
            max_sum = sum(divisors[start_idx:start_idx+k_rem])
            if max_sum < target_rem:
                return None
            
            for i in range(start_idx, N - k_rem + 1):
                d = divisors[i]
                if d > target_rem:
                    continue
                res = dfs_path(i + 1, k_rem - 1, target_rem - d)
                if res is not None:
                    return [d] + res
            return None
        
        return dfs_path(0, k, target)

    # Try k = 1, 2, ...
    for k in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        result = solve(k)
        if result is not None:
            return result
            
    # Fallback
    return []
