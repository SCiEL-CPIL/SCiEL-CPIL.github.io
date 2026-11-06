import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcds = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Identify bad triples and build conflict lists
    # conflicts[x] is a set of frozensets {a, b} such that {a, b, x} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcds[a][b] == gcds[b][c] == gcds[c][a]:
                    conflicts[a].add(frozenset((b, c)))
                    conflicts[b].add(frozenset((a, c)))
                    conflicts[c].add(frozenset((a, b)))

    def can_add(x, S):
        """Check if x can be added to set S (represented as a set of ints)."""
        for pair in conflicts[x]:
            if pair.issubset(S):
                return False
        return True

    def greedy_add(order, S):
        """Add as many elements from order to S as possible."""
        S = set(S)
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    best_size = 0
    best_set = set()
    
    # Precompute conflict counts for sorting strategies
    conflict_counts = [len(conflicts[i]) for i in range(N + 1)]
    
    # Strategy 1: Deterministic Greedy with least conflicts first
    order_least = sorted(elements, key=lambda x: conflict_counts[x])
    S = greedy_add(order_least, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 2: Deterministic Greedy with most conflicts first
    order_most = sorted(elements, key=lambda x: conflict_counts[x], reverse=True)
    S = greedy_add(order_most, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Strategy 3: Randomized Local Search
    # We will run multiple iterations of: Random Greedy -> Local Search
    # Local Search: Try removing each element, then greedily refill. Keep best.
    
    # Set a random seed for reproducibility if needed, but random is fine for exploration
    # However, to ensure we don't miss the "best" found in a specific run, we just keep the global best.
    
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random order
        current_order = elements[:]
        random.shuffle(current_order)
        
        # Greedy phase
        S = greedy_add(current_order, set())
        
        # Local Search phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_S_in_iter = S
            best_size_in_iter = len(S)
            
            for u in list(S):
                # Try removing u
                S_temp = S - {u}
                # Greedily add elements from the remaining pool
                # To speed this up, we can just iterate through all elements not in S_temp
                # and try to add them.
                # We use a fixed order for refill to be deterministic within the step?
                # Or random? Let's use conflict count order for refill to be consistent.
                remaining = [x for x in elements if x not in S_temp]
                # Sort by conflict count? This is expensive inside the loop.
                # Just use the original list order or random?
                # Let's use the 'order_least' for refill as it's a good heuristic.
                new_S = greedy_add(order_least, S_temp)
                
                if len(new_S) > best_size_in_iter:
                    best_size_in_iter = len(new_S)
                    best_S_in_iter = new_S
                    improved = True
                    
            if improved:
                S = best_S_in_iter
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S

    # Final check: try to add any remaining elements to the best set found
    # This might catch elements that were not added due to ordering
    for x in elements:
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
            
    return list(best_set)
