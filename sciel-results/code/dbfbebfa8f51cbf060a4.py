import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter(adj, n):
    """
    Returns the diameter of the graph. Returns 999 if disconnected.
    """
    max_diam = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        
        if any(d == -1 for d in dist):
            return 999
            
        for i in range(n):
            if i > src:
                d = dist[i]
                if d > max_diam:
                    max_diam = d
                    
    return max_diam

def _build_adj(base_edges, added_edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        if u not in adj[v] and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            return None
    return adj

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set(tuple(e) for e in base_edges)
    
    # Pre-compute bipartite candidates
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                all_candidates.append((u, v))
                
    best_solution = None
    best_count = float('inf')
    
    deadline = time.time() + 85
    
    # Strategy: 
    # 1. Find a valid solution quickly (Greedy Add)
    # 2. Minimize by removing edges (Greedy Remove)
    # 3. Repeat with different random seeds for the Add phase to find better starting points.
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        current_added = []
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        curr_diam = _bfs_diameter(adj, n)
        
        # Phase 1: Add edges to reach diameter <= 4
        while curr_diam > 4:
            if time.time() > deadline - 5:
                break
                
            best_edge = None
            best_new_diam = curr_diam
            
            # Shuffle candidates
            candidates = all_candidates[:]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if v in adj[u]:
                    continue
                    
                # Tentatively add
                adj[u].add(v)
                adj[v].add(u)
                
                new_diam = _bfs_diameter(adj, n)
                
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            current_added.append([u, v])
            curr_diam = best_new_diam
            
            if curr_diam <= 4:
                break
        
        # Phase 2: Remove edges to minimize count
        if curr_diam <= 4:
            # Try to remove edges
            removed = True
            while removed:
                removed = False
                # Shuffle to try different removals
                edges_to_try = current_added[:]
                random.shuffle(edges_to_try)
                
                for i, edge in enumerate(edges_to_try):
                    if time.time() > deadline - 2:
                        break
                    
                    u, v = edge
                    # Remove edge
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    new_diam = _bfs_diameter(adj, n)
                    
                    # Check if still valid
                    if new_diam <= 4:
                        # Keep it removed
                        current_added.remove(edge)
                        removed = True
                        break
                    else:
                        # Restore edge
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Update best
            if len(current_added) < best_count:
                best_count = len(current_added)
                best_solution = [list(e) for e in current_added]
                
    if best_solution is None:
        # Fallback: Return a known valid set if greedy failed
        # A safe fallback is to add many edges, but we need to be valid.
        # Let's try a static pattern if search fails.
        # Pattern: (i, i+11) for i in 0..12
        fallback = [[i, i+11] for i in range(13)]
        # Verify fallback
        adj_f = _build_adj(base_edges, fallback, n)
        if adj_f is not None and _bfs_diameter(adj_f, n) <= 4:
            return [[int(u), int(v)] for u, v in fallback]
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]
