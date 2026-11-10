import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    def get_220():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points)
        
    best_points = get_220()
    best_k = len(best_points)
    
    # Maximum clique heuristic on the graph of 220 norm-2 vectors
    # Two points u, v form a valid pair if ||u - v||^2 >= max_norm_sq
    # For norm-2 vectors: ||u - v||^2 = 4 - 2*u.v
    # We need 4 - 2*u.v >= 2, i.e., u.v <= 1
    # Since u.v is integer, u.v <= 1 means u.v in {..., -2, -1, 0, 1}
    # But u.v = 1 means they share exactly one coordinate with same sign
    # u.v = 0 means they share no coordinates or share with opposite signs
    # u.v = -1 means they share exactly one coordinate with opposite signs
    
    # Actually, let me reconsider. The condition is:
    # min_{i!=j} ||p_i - p_j||^2 >= max_i ||p_i||^2
    # For all norm-2 vectors, max norm sq = 2
    # So we need ||u - v||^2 >= 2 for all pairs
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 - 2*u.v >= 2
    # So u.v <= 1
    
    # Build adjacency: edge between u and v if u.v <= 1
    n = len(best_points)
    # Precompute dot products
    dots = best_points @ best_points.T
    
    # Adjacency matrix: True if u.v <= 1 (valid pair)
    adj = dots <= 1
    
    # Maximum clique heuristic: greedy with random restarts
    def max_clique_greedy(adj, n, time_limit=580):
        best_size = 0
        best_clique = None
        
        # Try multiple random orderings
        for trial in range(200):
            if time.time() - start_time > time_limit:
                break
            
            order = np.random.permutation(n)
            clique = []
            for idx in order:
                # Check if idx can join clique
                can_join = True
                for c in clique:
                    if not adj[idx, c]:
                        can_join = False
                        break
                if can_join:
                    clique.append(idx)
            
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique.copy()
        
        return best_clique
    
    # Run maximum clique heuristic
    clique = max_clique_greedy(adj, n, time_limit=580)
    if clique is not None and len(clique) > best_k:
        best_points = best_points[clique]
        best_k = len(best_points)
    
    # Also try a local search improvement
    if time.time() - start_time < 590:
        # Try to add points back or swap
        selected_set = set(range(n)) if best_k == n else set()
        if best_k < n:
            selected_set = set(np.where(np.isin(np.arange(n), best_points @ best_points.T <= 1))[0])
            # This is getting complicated, let's just use the clique result
        
    return np.around(best_points).astype(np.int64)
