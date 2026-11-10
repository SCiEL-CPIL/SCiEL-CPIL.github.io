import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Greedy baseline
    def greedy_baseline():
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best_set = greedy_baseline()
    
    # DP to find min cardinality subset summing to 200
    def find_min_conflict(S):
        # S is a set of integers
        if not S:
            return None
        
        # dp[s] = min number of elements to reach sum s
        # parent[s] = (prev_sum, element) to reconstruct
        INF = float('inf')
        dp = [INF] * 201
        parent = [None] * 201
        dp[0] = 0
        
        elements = sorted(S)
        for x in elements:
            # Iterate backwards to ensure each element used at most once
            for s in range(200 - x, -1, -1):
                if dp[s] != INF:
                    new_s = s + x
                    if new_s <= 200 and dp[s] + 1 < dp[new_s]:
                        dp[new_s] = dp[s] + 1
                        parent[new_s] = (s, x)
        
        if dp[200] == INF:
            return None
        
        # Reconstruct the set
        conflict = set()
        s = 200
        while s > 0:
            if parent[s] is None:
                return None # Should not happen if dp[200] != INF
            prev_s, x = parent[s]
            conflict.add(x)
            s = prev_s
        return conflict

    def search(S, current_depth):
        nonlocal best_set
        
        if time.time() > deadline:
            return
        
        # Prune if current size cannot beat best
        if len(S) <= len(best_set):
            return
        
        conflict = find_min_conflict(S)
        
        if conflict is None:
            # Valid set found
            if len(S) > len(best_set):
                best_set = list(S)
            return
        
        # Branch on removing one element from the conflict
        # Heuristic: remove elements that are "more central" or just iterate
        # Sorting by value might help finding large valid sets?
        for x in conflict:
            new_S = S - {x}
            search(new_S, current_depth + 1)
            if time.time() > deadline:
                return

    # Start search with full set
    full_set = set(range(1, 101))
    search(full_set, 0)
    
    return best_set

# EVOLVE-BLOCK-END
