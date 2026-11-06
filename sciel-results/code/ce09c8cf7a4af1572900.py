"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = 5040
    target = 4979
    
    # 1. Get all divisors of 5040
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    D = len(divisors)
    
    # Precompute prefix sums for pruning in backtracking
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (D + 1)
    for i in range(D):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    def range_sum(lo, hi):
        """Sum of divisors[lo:hi] using prefix sums"""
        return prefix[hi] - prefix[lo]
    
    def solve_k_backtrack(k, target, divs):
        """
        Find 'k' distinct elements from 'divs' that sum to 'target'.
        Returns the list of elements if found, else None.
        Uses strong pruning with prefix sums.
        """
        if k == 0:
            return [] if target == 0 else None
        if k == 1:
            # Binary search for target in sorted divs
            lo, hi = 0, len(divs) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if divs[mid] == target:
                    return [target]
                elif divs[mid] > target:
                    hi = mid - 1
                else:
                    lo = mid + 1
            return None
            
        if len(divs) < k:
            return None
        
        # Pruning: check if target is achievable with k elements
        max_sum = range_sum(0, k)  # k largest
        min_sum = range_sum(len(divs) - k, len(divs))  # k smallest
        if max_sum < target or min_sum > target:
            return None
        
        # Backtracking with strong pruning
        def backtrack(start, k_rem, target_rem, current_path):
            if k_rem == 0:
                if target_rem == 0:
                    return current_path[:]
                return None
            
            if len(divs) - start < k_rem:
                return None
            
            # Max possible sum: take k_rem largest from start onwards
            max_possible = range_sum(start, start + k_rem)
            if max_possible < target_rem:
                return None
            
            # Min possible sum: take k_rem smallest from start onwards
            # Note: min_possible is global min, but we need local min from start
            # Actually, the smallest k_rem elements available from start onwards are divs[-k_rem:]
            # But if start > len(divs) - k_rem, then we must take from start onwards
            # Let's just use the global min check as a coarse filter, and rely on max_possible for tighter pruning
            min_possible = range_sum(len(divs) - k_rem, len(divs))
            if min_possible > target_rem:
                return None
            
            for i in range(start, len(divs)):
                d = divs[i]
                if d > target_rem:
                    continue
                
                # Prune: can we form target_rem - d with k_rem - 1 elements from i+1 onwards?
                remaining = target_rem - d
                if k_rem - 1 == 0:
                    if remaining == 0:
                        current_path.append(d)
                        result = current_path[:]
                        current_path.pop()
                        return result
                    continue
                
                # Check if remaining is achievable with k_rem-1 elements from i+1
                if i + 1 + (k_rem - 1) > len(divs):
                    break  # Not enough elements left
                
                # Max sum of next k_rem-1 elements from i+1
                max_next = range_sum(i + 1, i + 1 + (k_rem - 1))
                if d + max_next < target_rem:
                    continue
                
                current_path.append(d)
                result = backtrack(i + 1, k_rem - 1, target_rem - d, current_path)
                if result:
                    return result
                current_path.pop()
            return None
            
        return backtrack(0, k, target, [])
    
    def solve_dijkstra(target, divs):
        """
        Model as shortest path on state space of (remainder, mask) is too large.
        Instead, use BFS/Dijkstra on (remainder, count) where we track the set of used divisors?
        No, that's exponential.
        
        Alternative: Use Dijkstra where state is (sum, bitmask) is too large for 60 divisors.
        
        Let's stick to the iterative deepening with backtracking, but optimize it.
        The key insight from the prompt is to use Dijkstra on remainders.
        However, with 60 divisors, the state space of remainders is up to 4979.
        But we need distinct divisors.
        
        Let's try a different approach:
        Iterate k from 1 to D.
        For each k, use backtracking with strong pruning.
        """
        pass
    
    # Iterate k from 1 upwards
    for k in range(1, D + 1):
        if time.time() > deadline:
             # Fallback to a valid solution (not necessarily minimal) if time runs out
             # The seed solution is a valid fallback.
             remainder = target
             chosen = []
             for d in divisors: # divisors is sorted desc
                 if d <= remainder:
                     chosen.append(d)
                     remainder -= d
             if remainder == 0:
                 return chosen
             else:
                 return [] # Should not happen for practical numbers, but safe
        
        result = solve_k_backtrack(k, target, divisors)
        if result:
            return result
            
    # Should not reach here
    return []

# EVOLVE-BLOCK-END
