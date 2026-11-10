# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
DEADLINE = 85  # seconds, safety margin from 90s limit

def _bfs_all_pairs(adj, n):
    """Return list of lists of distances from each vertex."""
    INF = float('inf')
    dists = []
    for s in range(n):
        d = [INF] * n
        d[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if d[v] == INF:
                    d[v] = d[u] + 1
                    queue.append(v)
        dists.append(d)
    return dists

def _graph_diameter(dists, n):
    INF = float('inf')
    mx = 0
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i][j]
            if d == INF:
                return -1  # disconnected
            if d > mx:
                mx = d
    return mx

def _is_valid(adj, n):
    """Check if graph has diameter exactly 2 and every edge is essential."""
    dists = _bfs_all_pairs(adj, n)
    diam = _graph_diameter(dists, n)
    if diam != 2:
        return False
    
    # Check each edge is essential
    for i in range(n):
        for j in range(i+1, n):
            if (j in adj[i]):
                # Remove edge (i,j)
                adj[i].remove(j)
                adj[j].remove(i)
                
                d2 = _bfs_all_pairs(adj, n)
                diam2 = _graph_diameter(d2, n)
                
                # Restore edge
                adj[i].append(j)
                adj[j].append(i)
                
                if diam2 == 2:
                    return False
    
    return True

def _add_random_edge(adj, n):
    """Add a random non-existent edge and return (u,v) or None."""
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                missing.append((i, j))
    if not missing:
        return None
    return random.choice(missing)

def construct():
    start_time = time.time()
    best_edges = None
    best_count = 0
    
    # Start with a star graph
    adj = [[] for _ in range(N)]
    for v in range(1, N):
        adj[0].append(v)
        adj[v].append(0)
    
    # Check if star is valid
    if _is_valid(adj, N):
        edge_set = set()
        for i in range(N):
            for j in adj[i]:
                if i < j:
                    edge_set.add((i, j))
        best_edges = list(edge_set)
        best_count = len(best_edges)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        if best_edges:
            return [[u, v] for u, v in best_edges]
        return [[0, v] for v in range(1, N)]
    
    # Try to improve by adding edges
    current_adj = [list(x) for x in adj]
    
    # Randomized search for better valid graphs
    attempt = 0
    while time.time() - start_time < DEADLINE - 2:
        attempt += 1
        
        # Create a fresh star-based graph
        adj2 = [[] for _ in range(N)]
        for v in range(1, N):
            adj2[0].append(v)
            adj2[v].append(0)
        
        # Try adding a few random edges
        max_additions = min(attempt % 5 + 1, 5)
        added = []
        for _ in range(max_additions):
            uv = _add_random_edge(adj2, N)
            if uv is None:
                break
            u, v = uv
            adj2[u].append(v)
            adj2[v].append(u)
            added.append((u, v))
        
        # Check validity
        if _is_valid(adj2, N):
            edge_set = set()
            for i in range(N):
                for j in adj2[i]:
                    if i < j:
                        edge_set.add((i, j))
            if len(edge_set) > best_count:
                best_edges = list(edge_set)
                best_count = len(edge_set)
        
        # Also try removing one edge from current best to see if we can add more
        if best_edges:
            adj3 = [[] for _ in range(N)]
            for u, v in best_edges:
                adj3[u].append(v)
                adj3[v].append(u)
            
            # Try removing each edge and adding a different one
            for u, v in list(best_edges):
                if time.time() - start_time > DEADLINE - 1:
                    break
                adj3[u].remove(v)
                adj3[v].remove(u)
                
                # Try adding a random missing edge
                uv2 = _add_random_edge(adj3, N)
                if uv2 is not None:
                    u2, v2 = uv2
                    adj3[u2].append(v2)
                    adj3[v2].append(u2)
                
                if _is_valid(adj3, N):
                    edge_set = set()
                    for i in range(N):
                        for j in adj3[i]:
                            if i < j:
                                edge_set.add((i, j))
                    if len(edge_set) > best_count:
                        best_edges = list(edge_set)
                        best_count = len(edge_set)
                
                # Restore
                if uv2 is not None:
                    u2, v2 = uv2
                    adj3[u2].remove(v2)
                    adj3[v2].remove(u2)
                adj3[u].append(v)
                adj3[v].append(u)
    
    if best_edges:
        return [[u, v] for u, v in best_edges]
    return [[0, v] for v in range(1, N)]

# EVOLVE-BLOCK-END
