import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Grötzsch graph on vertices 0..10
    # It's the Mycielski graph of C5
    # C5: 0-1-2-3-4-0
    # Mycielski adds vertices 5-9 (copies of 0-4) and vertex 10
    edges_set = set()
    
    def add_edge(u, v):
        if u == v:
            return
        if u > v:
            u, v = v, u
        edges_set.add((u, v))
    
    # C5
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski: for each edge (u,w) in C5, add (u, 5+w) and (w, 5+u)
    for i in range(5):
        u = i
        w = (i + 1) % 5
        add_edge(u, 5 + w)
        add_edge(w, 5 + u)
    
    # Connect vertex 10 to all of 5..9
    for u in range(5):
        add_edge(10, 5 + u)
    
    # Now we have 11 vertices, 20 edges
    # Add vertices 11..15 by connecting to independent sets
    
    # Find independent sets in the current graph
    def get_adjacency():
        adj = [set() for _ in range(16)]
        for (u, v) in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    adj = get_adjacency()
    
    # For adding vertex v, we can connect it to any independent set in the current graph
    # (since connecting to an independent set doesn't create triangles involving v)
    # We want to maximize edges, so connect to a large independent set
    
    # Find large independent sets using greedy
    def find_large_independent_set(adj, n_current):
        # Greedy: repeatedly pick vertex with fewest neighbors in remaining set
        remaining = set(range(n_current))
        independent = []
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            independent.append(best_v)
            remaining.remove(best_v)
            for nb in adj[best_v]:
                if nb in remaining:
                    remaining.discard(nb)
        return independent
    
    # Add vertices 11-15
    for v in range(11, 16):
        if time.time() > deadline:
            break
        # Find a large independent set in current graph (vertices 0..v-1)
        n_cur = v
        indep = find_large_independent_set(adj, n_cur)
        # Sort by some heuristic - maybe connect to vertices that are "central"
        # For now, just connect to the independent set
        for u in indep:
            add_edge(v, u)
        adj[v] = set(indep)
        for u in indep:
            adj[u].add(v)
    
    # Now do local search to add more edges
    def is_triangle_free():
        for (u, v) in edges_set:
            # Check if there's a common neighbor
            common = adj[u] & adj[v]
            if common:
                return False
        return True
    
    def is_3_colorable(adj, n):
        # Fast backtracking 3-colorability with MRV and forward checking
        color = [-1] * n
        domains = [set([0, 1, 2]) for _ in range(n)]
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # MRV: pick uncolored vertex with fewest available colors
            best_v = -1
            best_count = float('inf')
            for v in range(idx, n):
                if color[v] == -1:
                    cnt = len(domains[v])
                    if cnt == 0:
                        return False
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                    if cnt == 1:
                        break
            
            if best_v == -1:
                return True
            
            v = best_v
            saved_domain = domains[v].copy()
            for c in list(domains[v]):
                color[v] = c
                # Forward checking: remove c from neighbors
                consistent = True
                saved = []
                for nb in adj[v]:
                    if color[nb] == -1 and c in domains[nb]:
                        saved.append((nb, c))
                        domains[nb].discard(c)
                        if len(domains[nb]) == 0:
                            consistent = False
                            break
                
                if consistent:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                color[v] = -1
                for nb, c in saved:
                    domains[nb].add(c)
            
            domains[v] = saved_domain
            return False
        
        return backtrack(0)
    
    # Local search: try to add edges
    def try_add_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edges_set:
            return False
        # Check triangle freedom: no common neighbor
        if adj[u] & adj[v]:
            return False
        # Add edge
        edges_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        # Check 3-colorability
        if is_3_colorable(adj, 16):
            # Undo
            edges_set.discard((u, v))
            adj[u].discard(v)
            adj[v].discard(u)
            return False
        return True
    
    # Local search iterations
    iterations = 0
    while time.time() < deadline:
        progress = False
        # Try all possible edge additions
        for u in range(16):
            if time.time() > deadline:
                break
            for v in range(u + 1, 16):
                if time.time() > deadline:
                    break
                if (u, v) in edges_set:
                    continue
                if adj[u] & adj[v]:
                    continue
                if try_add_edge(u, v):
                    progress = True
                    iterations += 1
                    if time.time() > deadline:
                        break
        if not progress:
            break
    
    # Convert to list of edges
    result = [[u, v] for (u, v) in edges_set]
    return result
