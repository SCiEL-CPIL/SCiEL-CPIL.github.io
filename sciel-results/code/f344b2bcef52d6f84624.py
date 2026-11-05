import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Try n from 47 downwards
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg_target = n - 39
        
        # If target is 0, empty graph works
        if min_deg_target == 0:
            return {'n': n, 'edges': []}
            
        # Multiple restarts with different strategies
        max_restarts = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            result = _try_greedy_with_priority(n, min_deg_target, restart)
            if result is not None:
                return result
                
            # Try local search refinement
            result = _local_search(n, min_deg_target, restart)
            if result is not None:
                return result
    
    # Fallback
    return {'n': 39, 'edges': []}

def _try_greedy_with_priority(n, min_deg_target, seed):
    """Greedy construction prioritizing edges that help meet degree targets."""
    random.seed(seed)
    
    adj = [set() for _ in range(n)]  # Adjacency as sets for fast intersection
    deg = [0] * n
    edges = []
    
    # Generate all potential edges
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Strategy: iteratively pick edges, prioritizing those that help low-degree vertices
    # but also considering C4 constraints
    for iteration in range(len(pairs) * 2):  # Allow multiple passes
        if time.time() > deadline_global:
            return None
        
        if min(deg) >= min_deg_target:
            return {'n': n, 'edges': edges}
        
        # Score each remaining edge by how much it helps the lower-degree endpoint
        candidates = []
        for u, v in pairs:
            if u in adj[v]:  # Already an edge
                continue
            # Score: prioritize edges that connect to low-degree vertices
            score = min(deg[u], deg[v])
            candidates.append((score, u, v))
        
        if not candidates:
            break
        
        # Sort by score descending (prioritize helping low-degree vertices)
        candidates.sort(key=lambda x: -x[0])
        
        # Try to add the best candidate
        added = False
        for score, u, v in candidates:
            if _can_add_edge_fast(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                added = True
                break
        
        if not added:
            # No more edges can be added without creating C4
            break
    
    if min(deg) >= min_deg_target:
        return {'n': n, 'edges': edges}
    return None

def _local_search(n, min_deg_target, seed):
    """Local search: start with greedy result, try edge swaps to improve min degree."""
    random.seed(seed + 10000)
    
    # First, get a greedy solution
    adj_sets = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if _can_add_edge_fast(u, v, adj_sets):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
    
    if min(deg) >= min_deg_target:
        return {'n': n, 'edges': edges}
    
    # Local search: try to add edges to low-degree vertices
    for step in range(5000):
        if time.time() > deadline_global:
            return None
        
        if min(deg) >= min_deg_target:
            return {'n': n, 'edges': edges}
        
        # Find vertex with minimum degree
        min_u = min(range(n), key=lambda i: deg[i])
        
        # Try to add an edge to min_u
        added = False
        for _trial in range(20):
            v = random.randint(0, n-1)
            if v == min_u or v in adj_sets[min_u]:
                continue
            if _can_add_edge_fast(min_u, v, adj_sets):
                adj_sets[min_u].add(v)
                adj_sets[v].add(min_u)
                deg[min_u] += 1
                deg[v] += 1
                edges.append([min_u, v])
                added = True
                break
        
        if not added:
            # Try removing an edge to unblock
            removed = False
            for _trial in range(10):
                x = random.randint(0, n-1)
                if deg[x] <= min_deg_target or not adj_sets[x]:
                    continue
                y = random.choice(list(adj_sets[x]))
                if deg[y] > min_deg_target:
                    adj_sets[x].discard(y)
                    adj_sets[y].discard(x)
                    deg[x] -= 1
                    deg[y] -= 1
                    # Remove from edges list
                    for i, e in enumerate(edges):
                        if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                            edges.pop(i)
                            break
                    removed = True
                    break
            
            if not removed:
                break  # Stuck
    
    if min(deg) >= min_deg_target:
        return {'n': n, 'edges': edges}
    return None

def _can_add_edge_fast(u, v, adj):
    """Check if adding edge (u,v) creates a C4, using set intersection."""
    if u == v:
        return False
    if v in adj[u]:
        return False
    
    # Adding (u,v) creates C4 iff there exists a path u-w-x-v of length 3
    # i.e., exists w in N(u) and x in N(v) such that w-x is an edge
    # Equivalently: for some w in N(u), N(w) intersects N(v)
    # But we need to be careful: the path should not use u or v directly
    
    # More precisely: exists w in N(u), w != v, and x in N(w) ∩ N(v), x != u
    for w in adj[u]:
        if w == v:
            continue
        # Check if N(w) intersects N(v)
        if adj[w] & adj[v]:
            # Found a common neighbor x of w and v
            # Need to verify x != u (but x is in N(v), and u is not in N(v) since we're adding u-v)
            # Actually, u might be in N(v) if there's already an edge... but we checked v not in adj[u]
            # Since the graph is simple and undirected, if u not in adj[v] then v not in adj[u]
            # So x cannot be u because u is not in N(v)
            return False
    
    return True

# Global deadline for time checks
deadline_global = time.time() + 115.0
