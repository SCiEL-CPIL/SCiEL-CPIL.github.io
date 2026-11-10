import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline with 5s safety margin
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] with gcd(x, 255) > 1
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Build adjacency: adj[i] is set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy maximum clique: repeatedly pick vertex with max degree in remaining subgraph
    def greedy_clique(adj, n):
        remaining = set(range(n))
        clique = []
        while remaining:
            # Pick vertex with max degree within remaining
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.append(best_v)
            # Keep only neighbors of best_v in remaining
            remaining = remaining & adj[best_v]
        return clique
    
    best_clique = greedy_clique(adj, n)
    best_size = len(best_clique)
    
    # Try multiple random orderings to improve
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Shuffle candidate order and do greedy
        indices = list(range(n))
        random.shuffle(indices)
        
        remaining = set(indices)
        clique = []
        while remaining:
            # Among remaining, pick the one with highest degree in remaining
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.append(best_v)
            remaining = remaining & adj[best_v]
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
    
    # Also try the simple multiples-of-3 baseline (size 85)
    mult3 = list(range(3, 256, 3))
    if len(mult3) > best_size:
        # Convert to indices
        best_clique = [candidates.index(x) for x in mult3]
        best_size = len(best_clique)
    
    # Try local improvement: try to add vertices one at a time
    if time.time() < deadline:
        current_set = set(best_clique)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                if v in current_set:
                    continue
                # Check if v is connected to all vertices in current_set
                if all(u in adj[v] for u in current_set):
                    current_set.add(v)
                    improved = True
                    break
            if improved:
                best_clique = list(current_set)
                best_size = len(best_clique)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result (it should be, but verify)
    if 255 not in result:
        # This shouldn't happen since 255 is connected to all candidates
        result.append(255)
    
    return result
