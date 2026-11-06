import math
import random

def construct():
    N = 255
    # Candidates must share a factor with 255
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    
    # Build adjacency matrix or list
    # 127 vertices, so O(V^2) is fine
    V = len(candidates)
    adj = [[False] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True

    # Greedy Max Clique Heuristic
    # Try multiple random orders and greedy selections
    best_clique = []
    
    # Also keep the known good solution: multiples of 3
    multiples_of_3 = [x for x in candidates if x % 3 == 0]
    best_clique = multiples_of_3
    
    # Try to improve with local search / greedy
    # Simple greedy: pick vertex with max degree, add to clique, remove non-neighbors
    # Repeat for different starting points or random permutations
    
    for _ in range(100): # 100 iterations
        # Randomly shuffle candidates? Or just greedy on current graph?
        # Let's try a simple greedy on a random subset or random order
        order = list(range(V))
        random.shuffle(order)
        
        clique = []
        available = set(order)
        
        while available:
            # Pick vertex with max degree in available subgraph
            # To speed up, just pick first? No, max degree is better.
            best_v = -1
            max_deg = -1
            for v in available:
                deg = sum(1 for u in available if adj[v][u])
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            clique.append(best_v)
            # Remove non-neighbors of best_v from available
            neighbors = {u for u in available if adj[best_v][u]}
            available = neighbors
            
        # Convert indices to values
        clique_vals = [candidates[v] for v in clique]
        
        if len(clique_vals) > len(best_clique):
            best_clique = clique_vals
            
    return best_clique
