"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_regular_induced(mask, adj_matrix, n):
    """Check if the induced subgraph on vertices in mask is regular."""
    verts = [i for i in range(n) if (mask >> i) & 1]
    k = len(verts)
    if k < 6:
        return False
    # Compute degrees within the induced subgraph
    for v in verts:
        deg = 0
        for u in verts:
            if u != v and adj_matrix[v][u]:
                deg += 1
        # Check all other vertices have same degree
        first_deg = deg
        for u in verts:
            if u == v:
                continue
            deg2 = 0
            for w in verts:
                if w != u and adj_matrix[u][w]:
                    deg2 += 1
            if deg2 != first_deg:
                return False
        break  # Only need to check consistency from first vertex's perspective
    # Actually need to verify ALL vertices have the same degree
    degs = []
    for v in verts:
        d = 0
        for u in verts:
            if u != v and adj_matrix[v][u]:
                d += 1
        degs.append(d)
    if len(set(degs)) == 1:
        return True
    return False

def _has_regular_subgraph(adj_matrix, n):
    """Check if any subset of >= 6 vertices induces a regular graph."""
    from itertools import combinations
    for k in range(6, n + 1):
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_induced(mask, adj_matrix, n):
                return True
    return False

def _check_valid(adj_matrix, n):
    """Full validation: no induced subgraph of size >= 6 is regular."""
    return not _has_regular_subgraph(adj_matrix, n)

def _random_graph(n, p=0.5, rng=None):
    """Generate a random graph on n vertices with edge probability p."""
    if rng is None:
        rng = random
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
    return adj, edges

def _local_search(adj, n, max_steps=500, rng=None, deadline=None):
    """Local search: flip edges to eliminate regular induced subgraphs."""
    if rng is None:
        rng = random
    adj = [row[:] for row in adj]
    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find a regular induced subgraph
        regular_found = False
        from itertools import combinations
        for k in range(6, n + 1):
            if deadline is not None and time.time() > deadline:
                break
            found = False
            for comb in combinations(range(n), k):
                if deadline is not None and time.time() > deadline:
                    break
                verts = list(comb)
                degs = []
                for v in verts:
                    d = 0
                    for u in verts:
                        if u != v and adj[v][u]:
                            d += 1
                    degs.append(d)
                if len(set(degs)) == 1:
                    # Found a regular induced subgraph; try to break it
                    regular_found = True
                    found = True
                    # Pick a random edge within this subgraph and flip it
                    inner_edges = []
                    for idx in range(len(verts)):
                        for jdx in range(idx + 1, len(verts)):
                            inner_edges.append((verts[idx], verts[jdx]))
                    if inner_edges:
                        u, v = rng.choice(inner_edges)
                        adj[u][v] = not adj[u][v]
                        adj[v][u] = adj[u][v]
                        # Update edge list
                        if adj[u][v]:
                            if [u, v] not in edges and [v, u] not in edges:
                                edges.append([min(u,v), max(u,v)])
                        else:
                            edges = [[a, b] for a, b in edges if not ((a == u and b == v) or (a == v and b == u))]
                    break
            if found:
                break
            if deadline is not None and time.time() > deadline:
                break
        
        if not regular_found:
            # Check if truly valid
            if _check_valid(adj, n):
                edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                return adj, edges, True
        
        step += 1
    
    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    return adj, edges, _check_valid(adj, n)

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_n = 5
    best_edges = []
    
    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() > deadline - 10:
            break
        
        found_valid = False
        max_restarts = 30 if n <= 15 else 15
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Start with random graph
            p = 0.5
            adj, edges = _random_graph(n, p, rng)
            
            # Try local search
            adj, edges, valid = _local_search(adj, n, max_steps=200, rng=rng, deadline=deadline)
            
            if valid:
                best_n = n
                best_edges = edges
                found_valid = True
                break
        
        if found_valid:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
