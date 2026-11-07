import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    n = 15
    vertices = list(range(n))
    
    # Precompute all pairs
    all_pairs = list(itertools.combinations(vertices, 2))
    
    def get_neighbors(edges):
        neighbors = {i: set() for i in vertices}
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        return neighbors

    def get_degrees(neighbors):
        return {i: len(neighbors[i]) for i in vertices}

    def is_valid_addition(u, v, neighbors, degrees):
        nu = neighbors[u]
        nv = neighbors[v]
        
        for s in nu:
            for t in nv:
                if s == t:
                    continue
                if degrees[s] == degrees[t]:
                    return False
        return True

    best_edges = set()
    best_count = 0
    
    start_time = time.time()
    # Strict deadline to avoid timeout
    deadline = start_time + 50.0
    
    # Random restarts
    while time.time() < deadline:
        edges = set()
        
        # Greedy Phase
        while time.time() < deadline - 1.0: # Leave 1s for final processing
            neighbors = get_neighbors(edges)
            degrees = get_degrees(neighbors)
            
            added_any = False
            # Shuffle to break symmetry
            pairs_to_try = list(all_pairs)
            random.shuffle(pairs_to_try)
            
            for u, v in pairs_to_try:
                if (u, v) in edges:
                    continue
                
                if is_valid_addition(u, v, neighbors, degrees):
                    edges.add((u, v))
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added_any = True
                    if len(edges) > best_count:
                        best_edges = set(edges)
                        best_count = len(edges)
            
            if not added_any:
                # Stuck. Kick.
                if len(edges) == 0:
                    break
                
                # Remove 1 random edge
                edge_to_remove = random.choice(list(edges))
                edges.remove(edge_to_remove)
                # Next iteration will recompute neighbors/degrees
                pass
        
        # If we have time, we can try to refine the best_edges found so far
        # But restarting from scratch is often better for exploration.
        # Let's just continue the outer loop.

    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
