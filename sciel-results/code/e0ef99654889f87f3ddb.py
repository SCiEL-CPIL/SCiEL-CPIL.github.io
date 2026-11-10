import time
import sys

def construct():
        start_time = time.time()
        deadline = 50.0  # Safety margin
        
        S = set(range(1, 101))
        
        def get_counts(elements):
            dp = [0] * 201
            dp[0] = 1
            # Sort for determinism? Not needed for counts.
            for e in elements:
                # Iterate backwards to avoid using e multiple times
                for s in range(200, e - 1, -1):
                    if dp[s - e] > 0:
                        dp[s] += dp[s - e]
            return dp

        def find_witness(elements, target, dp):
            # dp is for the current set 'elements'
            if target == 0:
                return []
            if not elements or dp[target] == 0:
                return None
            
            # Heuristic: try larger elements first? 
            # Or just iterate.
            # To speed up, we can pick an element that is likely in a solution.
            # Any element works.
            # We'll pick the largest element <= target.
            # This reduces branching depth.
            candidates = [e for e in elements if e <= target]
            if not candidates:
                return None
            
            # Sort candidates descending
            candidates.sort(reverse=True)
            
            for e in candidates:
                # Try including e
                new_elements = elements - {e}
                # We need dp for new_elements? 
                # We don't have it precomputed for all subsets.
                # But we can check if (target - e) is reachable in new_elements.
                # This requires a sub-DP or a quick check.
                # Since N is small, we can just recurse and let the base case handle it?
                # But without pruning, it might be slow.
                
                # Better: Pass a "remaining set" and use a simple recursive search with memoization?
                # Or just use the fact that we have the global DP?
                # The global DP says 'target' is reachable in 'elements'.
                # It doesn't tell us if 'target-e' is reachable in 'elements-{e}'.
                
                # Let's just do a standard backtracking.
                # To make it fast, we can use a simple heuristic:
                # If we pick e, we need to form target-e from elements-{e}.
                
                # We'll implement a simple DFS.
                # Given the small size, this is fine.
                
                sub = find_witness(new_elements, target - e, None) 
                # Note: passing None means we don't have a precomputed DP for the subproblem.
                # This makes find_witness potentially slow if it explores many paths.
                
                if sub is not None:
                    return [e] + sub
            
            return None

        # The above find_witness is inefficient because it doesn't use DP pruning effectively in recursion.
        # Let's rewrite find_witness to use a local DP or just rely on the small search space.
        # Actually, since we call get_counts anyway, we can just use a simple boolean DP for the witness search?
        # No, let's stick to the simple recursive search but add a simple pruning:
        # If sum(remaining) < target, return None.
        
        def find_witness_simple(elements, target):
            if target == 0:
                return []
            if not elements:
                return None
            # Pruning
            if sum(elements) < target:
                return None
            
            # Try each element
            for e in elements:
                if e > target:
                    continue
                new_elements = elements - {e}
                sub = find_witness_simple(new_elements, target - e)
                if sub is not None:
                    return [e] + sub
            return None

        # This simple search might be slow.
        # But we only call it when dp[200] > 0.
        # And we stop at the first witness.
        # With N=100, and target 200, the depth is limited.
        
        dp = get_counts(S)
        
        while dp[200] > 0:
            if time.time() - start_time > deadline:
                break
            
            T = find_witness_simple(S, 200)
            if T is None:
                break # Should not happen
            
            best_x = None
            max_impact = -1
            best_dp = None
            
            for x in T:
                S_temp = S - {x}
                dp_temp = get_counts(S_temp)
                impact = dp_temp[200 - x] if 200 - x >= 0 else 0
                
                if impact > max_impact:
                    max_impact = impact
                    best_x = x
                    best_dp = dp_temp
            
            if best_x is not None:
                S.remove(best_x)
                dp = best_dp
        
        return list(S)
