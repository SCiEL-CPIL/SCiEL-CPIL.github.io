"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency sets
_ADJ = [set() for _ in range(P['n'])]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)


def _distinct_degrees(vertices):
    """Count distinct degrees in the induced subgraph on vertices."""
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in vset if u != v and u in _ADJ[v])
        degrees.append(deg)
    return len(set(degrees))


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    min_vertices = P['minimum_vertices']
    all_vertices = list(range(n))
    
    # Best result so far
    best_vertices = list(range(9))
    best_score = _distinct_degrees(best_vertices)
    
    # Try to find a better solution via local search
    # Start from the seed and try to improve by swapping vertices
    
    def try_improve(vertices, deadline):
        """Try to improve the current vertex set by swapping one vertex out for another."""
        current_vertices = list(vertices)
        current_score = _distinct_degrees(current_vertices)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current_vertices)
            
            # Try removing each vertex and adding each non-member
            for i, v_out in enumerate(current_vertices):
                if time.time() >= deadline:
                    break
                for v_in in all_vertices:
                    if v_in in current_set:
                        continue
                    
                    # Try the swap
                    new_vertices = list(current_vertices)
                    new_vertices[i] = v_in
                    new_score = _distinct_degrees(new_vertices)
                    
                    if new_score > current_score:
                        current_vertices = new_vertices
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return current_vertices, current_score
    
    # Run local search from the seed
    improved_vertices, improved_score = try_improve(best_vertices, deadline)
    if improved_score > best_score:
        best_vertices = improved_vertices
        best_score = improved_score
    
    # Try a few random restarts if time permits
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline - 2 and attempts < 50:
        attempts += 1
        # Random 8-vertex subset
        random_vertices = random.sample(all_vertices, min_vertices)
        rv, rs = try_improve(random_vertices, deadline)
        if rs > best_score:
            best_vertices = rv
            best_score = rs
    
    # Also try larger subsets (9, 10, 11 vertices) from the best found
    for size in range(9, 13):
        if time.time() >= deadline - 1:
            break
        # Try to expand the best_vertices to size
        current = list(best_vertices)
        if len(current) < size:
            # Add random vertices
            available = [v for v in all_vertices if v not in current]
            random.shuffle(available)
            current.extend(available[:size - len(current)])
        
        improved_vertices, improved_score = try_improve(current, deadline)
        if improved_score > best_score:
            best_vertices = improved_vertices
            best_score = improved_score
    
    # Ensure we return exactly the right type
    return list(best_vertices)
# EVOLVE-BLOCK-END
