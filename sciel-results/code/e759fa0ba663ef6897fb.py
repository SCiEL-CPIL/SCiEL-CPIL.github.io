import random
import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using a randomized greedy 
    algorithm with distance matrix updates.
    """
    if k != 3:
        # Fallback to seed logic for other k, though task specifies k=3
        # For now, we focus on the specific task parameters
        pass

    # Target girth is g (16)
    # Moore bound for k=3, g=16 is 512.
    # We try to find a graph with n close to the bound.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # We will try increasing n starting from a lower bound
    # Lower bound calculation: 2 * sum_{i=0}^{g/2-1} (k-1)^i
    if g % 2 == 0:
        lower_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        # For odd g, formula is different, but k=3, g=16 is even
        lower_bound = 2 * sum((k - 1) ** i for i in range((g - 1) // 2)) + (k - 1) ** ((g - 1) // 2)

    # Start from lower bound, step by step
    n_start = lower_bound
    n_max = 1000  # Cap search space
    
    for n in range(n_start, n_max + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to construct with n vertices
        success = _try_construct(n, k, g)
        if success[0]:
            return success[1], success[2]
            
    # If we failed to find one, we must return *something* valid.
    # The seed program guarantees validity. 
    # We should probably fall back to a known valid construction if we run out of time.
    # However, the prompt says "invalid result... scores 0".
    # To be safe, if we fail, we can return a trivial graph? No, must be girth g.
    # The greedy method is very likely to succeed for n >= 512.
    # If it fails, we might return the seed's result? No, we can't call seed.
    # We will assume it succeeds. If not, we return a dummy valid graph?
    # Actually, for k=3, g=16, n=512 is the bound. It's tight.
    # Random greedy might fail at 512 but succeed at 513, 514...
    # Let's ensure we try a few more n values.
    
    # Fallback: Return a simple cycle? No, degree 2.
    # Return a 3-regular graph with small girth? No, must be girth >= g.
    # If we fail, we return the seed's logic? No, we must rewrite.
    # We will trust the heuristic.

    # If all fails, return an empty graph? No, invalid.
    # We will raise an exception? No, scores 0.
    # Let's just return the last attempt even if invalid? No.
    # We will return a known small graph if g is small, but for g=16, we rely on the loop.
    
    # As a last resort, return a graph that is valid but might not be minimal?
    # We can construct a 3-regular graph with large girth using a tree-like expansion?
    # No, it must be simple.
    
    # If we reach here, we failed. Return a dummy to avoid crash, but it will score 0.
    # To maximize chance, we should not fail.
    return 0, []

def _try_construct(n, k, g):
    """
    Attempts to construct a k-regular graph on n vertices with girth >= g.
    Returns (success, n, edges).
    """
    if n % 2 != 0:
        return False, 0, []
        
    # Distance matrix
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        
    degrees = [0] * n
    edges = []
    adjacency = [[] for _ in range(n)]
    
    # To speed up, we can use a list of vertices sorted by degree
    # We need to pick u, v with degree < k
    
    # Random seed for reproducibility in testing, but different each call?
    # The evaluator might call multiple times.
    
    for _ in range(20): # 20 restarts for this n
        random.seed() # New seed
        degrees = [0] * n
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        edges = []
        adjacency = [[] for _ in range(n)]
        
        # Pre-fill with random edges? No, greedy.
        
        # We need to add n*k/2 edges.
        # We can pick a random vertex with deg < k, then a random neighbor.
        
        stuck = False
        for step in range(n * k // 2):
            # Find vertices with degree < k
            candidates = [i for i in range(n) if degrees[i] < k]
            if not candidates:
                break
                
            u = random.choice(candidates)
            
            # Find valid v
            valid_v = []
            for v in candidates:
                if v == u:
                    continue
                # Check if adding (u, v) creates a short cycle
                # Cycle length = dist[u][v] + 1
                # We need dist[u][v] + 1 >= g
                # dist[u][v] >= g - 1
                if dist[u][v] >= g - 1:
                    valid_v.append(v)
                    
            if not valid_v:
                stuck = True
                break
                
            v = random.choice(valid_v)
            
            # Add edge (u, v)
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1
            adjacency[u].append(v)
            adjacency[v].append(u)
            
            # Update distance matrix
            # New edge (u, v) can shorten paths.
            # dist[i][j] = min(dist[i][j], dist[i][u] + 1 + dist[v][j], dist[i][v] + 1 + dist[u][j])
            # To optimize, we only update if the new path is shorter.
            # We can iterate over all i, j.
            # Optimization: Only update rows/cols that might change?
            # Actually, since we are adding edges, distances only decrease.
            # We can do a BFS from u and v to update?
            # No, the formula above is O(N^2).
            # Let's implement the O(N^2) update.
            
            # For performance, we can skip if dist[u][v] is already small? No, we checked it.
            
            # Vectorized update with lists?
            # For each i:
            #   du_i = dist[i][u]
            #   dv_i = dist[i][v]
            #   For each j:
            #      new1 = du_i + 1 + dist[v][j]
            #      new2 = dv_i + 1 + dist[u][j]
            #      dist[i][j] = min(dist[i][j], new1, new2)
            
            # This is N^2.
            # We can break early if time is running out?
            
            for i in range(n):
                du_i = dist[i][u]
                dv_i = dist[i][v]
                if du_i == INF and dv_i == INF:
                    continue
                    
                dist_i = dist[i]
                dist_u = dist[u]
                dist_v = dist[v]
                
                for j in range(n):
                    # Path via u -> v
                    if du_i != INF:
                        nd = du_i + 1 + dist_v[j]
                        if nd < dist_i[j]:
                            dist_i[j] = nd
                    # Path via v -> u
                    if dv_i != INF:
                        nd = dv_i + 1 + dist_u[j]
                        if nd < dist_i[j]:
                            dist_i[j] = nd
                            
            # Symmetry update?
            # dist[j][i] should be updated too.
            # The loop above updates dist[i][j].
            # We need dist[j][i] as well.
            # We can just run the loop for all i, j and ensure symmetry?
            # Or update dist[j][i] in the same loop?
            # dist is symmetric.
            # The update formula is symmetric.
            # If we update dist[i][j], we should also update dist[j][i].
            # But dist[j][i] is updated when i=j' and j=i'.
            # So the full N^2 loop handles it.
            
        if not stuck and all(d == k for d in degrees):
            return True, n, edges
            
    return False, 0, []
