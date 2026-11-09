import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Identify perfect squares up to 2*256 = 512
    max_sum = 2 * 256
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # 2. Filter out numbers where 2*x is a perfect square (diagonal constraint)
    candidates = []
    for x in range(1, 257):
        if 2 * x not in squares:
            candidates.append(x)
    
    # 3. Build conflict graph among candidates
    # conflict_map[x] = set of y in candidates such that x+y is a square
    conflict_map = {x: set() for x in candidates}
    for i in range(len(candidates)):
        x = candidates[i]
        for j in range(i + 1, len(candidates)):
            y = candidates[j]
            if x + y in squares:
                conflict_map[x].add(y)
                conflict_map[y].add(x)
    
    best_set = set()
    best_size = 0
    
    # 4. Greedy Min-Degree Algorithm with Random Restart
    # We perform multiple iterations to find a larger independent set
    # The min-degree heuristic for Max Independent Set is:
    # Pick vertex with min degree in current graph, add to IS, remove it and neighbors.
    
    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(candidates)
        current_set = set()
        
        while available:
            # Find node with minimum degree in the induced subgraph of 'available'
            min_deg = float('inf')
            min_nodes = []
            
            # Optimization: Instead of iterating all available and counting neighbors in available,
            # we can just pick a random node if we want pure random, but min-degree is better.
            # For N=256, iterating is fast.
            
            # To speed up tie-breaking randomness, we can shuffle or just pick random among mins
            # But finding mins requires counting.
            
            # Let's just pick a random node? No, min-degree is structurally better.
            # Let's optimize the min-degree finding.
            
            # Actually, for a sparse graph, we can just iterate.
            # deg(v) = len(conflict_map[v] & available)
            
            # To add randomness to ties, we can collect all min-degree nodes and pick one randomly.
            
            # Optimization: If we just pick a random node, it's a random greedy.
            # Min-degree is a specific heuristic. Let's stick to min-degree.
            
            # To make it fast, we don't need to recompute degrees from scratch if we are careful,
            # but N is small enough that O(N^2) per step is fine? 
            # Steps ~ N. Total O(N^3) = 256^3 ~ 16M ops per restart. 
            # 1000 restarts = 16B ops. Too slow in Python.
            
            # We need a faster greedy or fewer restarts.
            # Let's reduce restarts or optimize the inner loop.
            
            # Alternative: Just use random order greedy?
            # Random order greedy: Shuffle candidates, pick first valid.
            # This is O(N^2) per restart. 1000 * 256^2 ~ 65M ops. Feasible.
            
            # Let's try Random Order Greedy first. It's simpler and often good.
            # But Min-Degree is usually better for Max IS.
            
            # Let's try a hybrid: 
            # 1. Calculate degrees once.
            # 2. Pick min degree.
            # 3. Update degrees of remaining neighbors.
            
            # Let's implement the dynamic degree update for Min-Degree Greedy.
            
            # Reset for this restart
            # We need a copy of available and a way to get current degree.
            # Current degree of v = number of neighbors of v that are still in available.
            
            # We can maintain a counter for each node.
            # But we need to reset it every restart.
            
            # Let's just use Random Greedy for simplicity and speed, 
            # and rely on many restarts.
            
            # Actually, let's try Min-Degree with a simple O(N) scan per step.
            # N=256. 256 steps. 256*256 = 65k ops per restart.
            # 1000 restarts = 65M ops. This is fine in Python (approx 1-2s).
            
            min_deg = float('inf')
            candidates_for_pick = []
            
            for v in available:
                deg = len(conflict_map[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_pick = [v]
                elif deg == min_deg:
                    candidates_for_pick.append(v)
            
            # Pick randomly among min-degree nodes to add diversity
            v = random.choice(candidates_for_pick)
            current_set.add(v)
            
            # Remove v and its neighbors from available
            # Neighbors to remove are those in conflict_map[v] that are in available
            to_remove = {v}
            to_remove.update(conflict_map[v] & available)
            available.difference_update(to_remove)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
