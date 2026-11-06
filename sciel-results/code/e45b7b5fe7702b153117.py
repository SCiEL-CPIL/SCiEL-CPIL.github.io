"""K3,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 20
DEADLINE = time.time() + 145  # safety margin before 150s kill

def _count_common_neighbors(adj, a, b, c):
    """Count common neighbors of vertices a, b, c, excluding a, b, c themselves."""
    common = adj[a] & adj[b] & adj[c]
    common[a] = False
    common[b] = False
    common[c] = False
    return np.sum(common)

def _has_k33(adj):
    """Check if graph contains a K3,3 subgraph."""
    n = adj.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if _count_common_neighbors(adj, a, b, c) >= 3:
                    return True
    return False

def _count_k33_violations(adj):
    """Count number of triples with >= 3 common neighbors."""
    n = adj.shape[0]
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if _count_common_neighbors(adj, a, b, c) >= 3:
                    count += 1
    return count

def _build_edges_from_adj(adj):
    """Extract edge list from adjacency matrix."""
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v). Return True if it doesn't create K3,3."""
    if u == v or adj[u, v]:
        return False
    # Adding edge (u,v) could create K3,3 only if there exists a triple
    # that now has >= 3 common neighbors. The new edge can only affect
    # triples that include u or v, or triples where u or v becomes a common neighbor.
    # Simple approach: just check all triples after adding the edge.
    adj[u, v] = True
    adj[v, u] = True
    if _has_k33(adj):
        adj[u, v] = False
        adj[v, u] = False
        return False
    return True

def _greedy_build():
    """Build a K3,3-free graph by greedily adding edges in random order."""
    adj = np.zeros((N, N), dtype=bool)
    # Generate random edge order
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if _try_add_edge(adj, u, v):
            pass
        else:
            # Check deadline
            if time.time() > DEADLINE:
                break
    
    return adj

def _local_search(adj):
    """Try to improve the graph by removing edges that allow more edges to be added."""
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        edges = _build_edges_from_adj(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > DEADLINE:
                return adj
            # Try removing this edge
            adj[u, v] = False
            adj[v, u] = False
            edges_after_removal = len(_build_edges_from_adj(adj))
            
            # Now try to greedily add edges back
            all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(all_edges)
            adj2 = adj.copy()
            for a, b in all_edges:
                if time.time() > DEADLINE:
                    break
                if adj2[a, b]:
                    continue
                adj2[a, b] = True
                adj2[b, a] = True
                if _has_k33(adj2):
                    adj2[a, b] = False
                    adj2[b, a] = False
            
            edges_after = len(_build_edges_from_adj(adj2))
            if edges_after > edges_after_removal:
                adj = adj2
                improved = True
                break
    
    return adj

def _random_restart_greedy():
    """Do a random restart with greedy edge addition."""
    adj = _greedy_build()
    if time.time() < DEADLINE:
        adj = _local_search(adj)
    return adj

def construct():
    best_adj = None
    best_count = 0
    
    # First, try a deterministic dense construction
    # Start with a known K3,3-free structure and try to add edges
    
    # Strategy 1: Random restarts with greedy + local search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        adj = _random_restart_greedy()
        count = len(_build_edges_from_adj(adj))
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
    
    # Strategy 2: Try starting from specific structures
    # Complete bipartite K2,18 (the seed)
    adj = np.zeros((N, N), dtype=bool)
    for a in range(2):
        for b in range(2, N):
            adj[a, b] = True
            adj[b, a] = True
    count = len(_build_edges_from_adj(adj))
    if count > best_count:
        best_count = count
        best_adj = adj.copy()
    
    # Try to add more edges to K2,18
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() > DEADLINE - 5:
            break
        if adj[u, v]:
            continue
        if _try_add_edge(adj, u, v):
            pass
    count = len(_build_edges_from_adj(adj))
    if count > best_count:
        best_count = count
        best_adj = adj.copy()
    
    # More random restarts if time permits
    while time.time() < DEADLINE - 10:
        adj = _random_restart_greedy()
        count = len(_build_edges_from_adj(adj))
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
    
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, N)]
    
    return _build_edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
