import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def _has_c4(adj, n):
        """Check if graph has a C4 using common neighbor count."""
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return True
        return False
    
    def _try_greedy(n, target_deg, seed, deadline):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        # Precompute all edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def is_safe(u, v):
            """Check if adding edge (u,v) keeps graph C4-free."""
            # Adding (u,v) creates C4 iff there exist a,b with u~a, a~b, b~v
            # Equivalently: after adding, some pair has >1 common neighbors
            # Pairs affected: (u,z) for z in N(v)\{u}, and (v,z) for z in N(u)\{v}
            # Also (u,v) itself: common neighbors of u and v must be <= 1
            
            # Check (u,v) common neighbors
            if len(adj[u] & adj[v]) > 1:
                return False
            
            # Check pairs (u, z) for z in N(v) where z != u
            for z in adj[v]:
                if z == u:
                    continue
                # After adding (u,v), v becomes common neighbor of u and z if z~v
                # z~v is true by definition (z in N(v))
                # So new common neighbors of (u,z) = old common + {v} if v not already common
                # v is in N(u) after adding, and v is in N(z) (z in N(v))
                # So v is a new common neighbor unless v was already in N(u) (it's not, we're adding)
                # So new count = old count + 1
                if len(adj[u] & adj[z]) >= 1:
                    # Old count >= 1, new count >= 2, unsafe
                    return False
            
            # Check pairs (v, z) for z in N(u) where z != v
            for z in adj[u]:
                if z == v:
                    continue
                if len(adj[v] & adj[z]) >= 1:
                    return False
            
            return True
        
        # Multi-pass greedy: in each pass, find the best edge to add
        # Best = minimizes max(deg[u], deg[v]) among safe edges that help under-degree vertices
        for pass_num in range(n * 3):
            if time.time() > deadline:
                break
            
            best_edge = None
            best_key = None
            
            for u, v in all_edges:
                if v in adj[u]:
                    continue
                
                deg_u = len(adj[u])
                deg_v = len(adj[v])
                
                # Only consider if it helps at least one under-degree vertex
                if deg_u >= target_deg and deg_v >= target_deg:
                    continue
                
                if not is_safe(u, v):
                    continue
                
                # Key: prioritize edges that help the most under-degree vertex
                # Use (max(deg_u, deg_v), min(deg_u, deg_v)) to balance
                key = (max(deg_u, deg_v), min(deg_u, deg_v))
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Early exit if target reached
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
        
        # Final verification
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if not _has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    best_n = 39
    best_edges = []
    
    # Try from highest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        target_deg = max(1, n - 39)
        
        # Try multiple seeds
        max_seeds = 50 if n >= 46 else 100
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            edges = _try_greedy(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid for this n, try higher n
        
        if best_n >= n:
            # Already found something at this level or better
            if best_n > n:
                break
            # Found at this n, try higher
            continue
    
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
