import time
import random
from collections import deque

P = {'n': 16}

def _diameter_and_essential(n, adj):
    """Check if graph has diameter 2 and every edge is essential.
    Returns (is_valid, num_edges) or (False, 0)."""
    # Check diameter is exactly 2
    # First check connectivity and diameter
    def bfs_dist(start):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist
    
    max_dist = 0
    has_dist_2 = False
    for i in range(n):
        d = bfs_dist(i)
        for j in range(n):
            if d[j] == -1:
                return False, 0  # disconnected
            if d[j] > 2:
                return False, 0
            if d[j] == 2:
                has_dist_2 = True
    
    if not has_dist_2:
        return False, 0  # diameter is 1, not 2
    
    # Now check every edge is essential
    # For each edge, remove it and check if diameter becomes > 2 or disconnected
    # This is O(E * n * (n+E)) which for n=16 is manageable
    
    # Collect edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    for (u, v) in edges:
        # Remove edge (u,v)
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Check diameter
        valid = True
        has_dist_2_after = False
        for i in range(n):
            d = bfs_dist(i)
            for j in range(n):
                if d[j] == -1:
                    valid = False
                    break
                if d[j] > 2:
                    valid = False
                    break
                if d[j] == 2:
                    has_dist_2_after = True
            if not valid:
                break
        
        # Restore edge
        adj[u].append(v)
        adj[v].append(u)
        
        if not valid:
            continue  # edge is essential (removing breaks diameter-2 or disconnects)
        else:
            # Edge is NOT essential if diameter stays 2 after removal
            # Wait, we need: removing edge increases diameter above 2 OR disconnects
            # If after removal, diameter is still 2, edge is NOT essential
            return False, 0
    
    return True, len(edges)

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Start with a random graph and try to optimize
    # Strategy: start with a sparse diameter-2 graph, greedily add edges
    
    # First, create a base graph with diameter 2
    # Use a random approach: start with a cycle and add chords
    
    best_edges = 0
    best_adj = None
    
    # Try multiple random starts
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        # Create a random graph
        adj = [set() for _ in range(n)]
        
        # Start with a Hamiltonian cycle to ensure connectivity
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(n):
            u = perm[i]
            v = perm[(i+1) % n]
            adj[u].add(v)
            adj[v].add(u)
        
        # Add random edges to try to get diameter 2
        for i in range(n):
            for j in range(i+1, n):
                if i not in adj[j] and random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Convert to list for the check
        adj_list = [list(s) for s in adj]
        
        valid, num_edges = _diameter_and_essential(n, adj_list)
        
        if valid and num_edges > best_edges:
            best_edges = num_edges
            best_adj = [sorted(s) for s in adj]
    
    if best_adj is None:
        # Fallback: return a simple valid graph
        # A path doesn't work (diameter too large)
        # Let's return a known small example
        # Triangle + connections... 
        # Just return the seed
        return [[0, v] for v in range(1, n)]
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges
