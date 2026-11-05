import time
import random

def _construct_polarity_graph(q):
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2, z2 = points[j]
            if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    return n_pts, adj

def _find_indep_set(active, adj, k, deadline):
    """
    Finds an independent set of size k in the induced subgraph of `active`.
    Returns the set of vertices or None.
    """
    rem = set(active)
    
    def _branch(rem, curr, size):
        if time.time() > deadline:
            return None # Timeout, treat as not found (risky, but prevents hang)
        if size == k:
            return curr.copy()
        if not rem:
            return None
        if size + len(rem) < k:
            return None
        
        # Heuristic: Order by degree in rem
        # To speed up, we can just pick the first one, but sorting helps pruning
        # For small n, sorting is fine.
        sorted_rem = sorted(rem, key=lambda v: len(adj[v] & rem))
        v = sorted_rem[0]
        
        # Branch 1: Include v
        # New rem: remove v and its neighbors
        new_rem = rem - adj[v] - {v}
        res = _branch(new_rem, curr | {v}, size + 1)
        if res is not None:
            return res
            
        # Branch 2: Exclude v
        res = _branch(rem - {v}, curr, size)
        if res is not None:
            return res
            
        return None

    return _branch(rem, set(), 0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = 30
    best_edges = []
    # Fallback: 10 disjoint triangles
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    # Try q=7
    if time.time() < deadline - 10:
        try:
            n, adj = _construct_polarity_graph(7)
            active = list(range(n))
            
            # Prune loop
            while time.time() < deadline - 5:
                indep_set = _find_indep_set(active, adj, 11, deadline)
                if indep_set is None:
                    break
                    
                # Find victim: min degree in active
                min_deg = float('inf')
                victim = -1
                active_set = set(active)
                for v in indep_set:
                    deg = len(adj[v] & active_set)
                    if deg < min_deg:
                        min_deg = deg
                        victim = v
                active.remove(victim)
                
            # If we have a valid graph with n > best_n
            if len(active) > best_n:
                # Relabel
                old_to_new = {v: i for i, v in enumerate(active)}
                edges = []
                for i in range(len(active)):
                    u = active[i]
                    for j in range(i + 1, len(active)):
                        v = active[j]
                        if v in adj[u]:
                            edges.append([old_to_new[u], old_to_new[v]])
                best_n = len(active)
                best_edges = edges
        except Exception:
            pass

    # Try q=5 if q=7 didn't improve or if we have time
    if time.time() < deadline - 10 and best_n < 35:
        try:
            n, adj = _construct_polarity_graph(5)
            active = list(range(n))
            
            while time.time() < deadline - 5:
                indep_set = _find_indep_set(active, adj, 11, deadline)
                if indep_set is None:
                    break
                    
                min_deg = float('inf')
                victim = -1
                active_set = set(active)
                for v in indep_set:
                    deg = len(adj[v] & active_set)
                    if deg < min_deg:
                        min_deg = deg
                        victim = v
                active.remove(victim)
                
            if len(active) > best_n:
                old_to_new = {v: i for i, v in enumerate(active)}
                edges = []
                for i in range(len(active)):
                    u = active[i]
                    for j in range(i + 1, len(active)):
                        v = active[j]
                        if v in adj[u]:
                            edges.append([old_to_new[u], old_to_new[v]])
                best_n = len(active)
                best_edges = edges
        except Exception:
            pass

    return {'n': best_n, 'edges': best_edges}
