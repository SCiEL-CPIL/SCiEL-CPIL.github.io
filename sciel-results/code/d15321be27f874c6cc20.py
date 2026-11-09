import random
import time
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def _is_k4_free(adj):
    """Check if graph with adjacency matrix adj (18x18 bool) is K4-free."""
    n = 18
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            # For each edge (u,v), check if there are 3 common neighbors
            # that form a triangle among themselves
            common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
            for a, b, c in combinations(common, 3):
                if adj[a][b] and adj[a][c] and adj[b][c]:
                    return False
    return True

def _max_triangle_free_induced(adj):
    """Find the maximum number of vertices inducing a triangle-free subgraph.
    Uses branch and bound."""
    n = 18
    # We want to find the largest subset S such that G[S] has no triangle.
    # Branch and bound: build S incrementally, check if adding next vertex
    # would create a triangle with existing vertices in S.
    
    best = [0]
    
    def has_triangle_in_subset(subset):
        """Check if the induced subgraph on subset has a triangle."""
        for a, b, c in combinations(subset, 3):
            if adj[a][b] and adj[a][c] and adj[b][c]:
                return True
        return False
    
    def branch(vertices_to_consider, current_subset, remaining):
        # Upper bound: len(current_subset) + len(vertices_to_consider)
        if len(current_subset) + len(vertices_to_consider) <= best[0]:
            return
        if not vertices_to_consider:
            if len(current_subset) > best[0]:
                best[0] = len(current_subset)
            return
        
        # Try adding the first vertex in vertices_to_consider
        v = vertices_to_consider[0]
        rest = vertices_to_consider[1:]
        
        # Option 1: exclude v
        branch(rest, current_subset, remaining - 1)
        
        # Option 2: include v (if it doesn't create a triangle)
        # Check if v forms a triangle with any two vertices in current_subset
        can_add = True
        for i, a in enumerate(current_subset):
            if adj[v][a]:
                for b in current_subset[i+1:]:
                    if adj[v][b] and adj[a][b]:
                        can_add = False
                        break
            if not can_add:
                break
        
        if can_add:
            branch(rest, current_subset + [v], remaining - 1)
    
    branch(list(range(n)), [], n)
    return best[0]

def _turan_graph_18_3():
    """Construct K6,6,6 (Turan graph T(18,3))."""
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def _edges_to_adj(edges):
    adj = [[False] * 18 for _ in range(18)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _adj_to_edges(adj):
    edges = []
    for u in range(18):
        for v in range(u + 1, 18):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def _random_k4_free_graph(adj):
    """Generate a random K4-free graph by random edge additions/removals."""
    adj2 = [row[:] for row in adj]
    n = 18
    random.seed()
    for _ in range(200):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        # Try to flip edge (u,v)
        if adj2[u][v]:
            # Remove edge
            adj2[u][v] = False
            adj2[v][u] = False
        else:
            # Add edge - check K4-freeness
            adj2[u][v] = True
            adj2[v][u] = True
            if not _is_k4_free(adj2):
                adj2[u][v] = False
                adj2[v][u] = False
    return adj2

def _local_search(adj, deadline):
    """Perform local search to minimize max triangle-free induced subgraph size."""
    best_adj = [row[:] for row in adj]
    best_score = _max_triangle_free_induced(adj)
    
    while time.time() < deadline - 5:
        # Try random perturbation
        adj2 = _random_k4_free_graph(adj)
        score = _max_triangle_free_induced(adj2)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj2]
            adj = adj2
            if best_score <= 10:
                break
    
    return best_adj, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s budget to leave margin
    
    # Start with Turan graph T(18,3) which has score 12
    edges = _turan_graph_18_3()
    adj = _edges_to_adj(edges)
    best_adj = [row[:] for row in adj]
    best_score = _max_triangle_free_induced(adj)  # Should be 12
    
    # Try local search
    if time.time() < deadline - 10:
        best_adj, best_score = _local_search(adj, deadline)
    
    # If time allows, try more random restarts
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Random K4-free graph
            adj2 = [[False] * 18 for _ in range(18)]
            random.seed(random.randint(0, 10**6))
            for u in range(18):
                for v in range(u + 1, 18):
                    if random.random() < 0.5:
                        adj2[u][v] = True
                        adj2[v][u] = True
            if _is_k4_free(adj2):
                score = _max_triangle_free_induced(adj2)
                if score < best_score:
                    best_score = score
                    best_adj = [row[:] for row in adj2]
    
    return _adj_to_edges(best_adj)

# EVOLVE-BLOCK-END
