"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_degrees(adj):
    return [len(s) for s in adj]

def _find_all_triangles(adj, n):
    """Find all triangles in the graph."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w > v:
                    triangles.append((u, v, w))
    return triangles

def _max_triangle_degree_sum(adj, n):
    """Compute the maximum degree sum over all triangles."""
    deg = _compute_degrees(adj)
    triangles = _find_all_triangles(adj, n)
    if not triangles:
        return 0
    max_sum = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > max_sum:
            max_sum = s
    return max_sum

def _edge_set(edges):
    return set(frozenset((u, v)) for u, v in edges)

def _try_improve(adj, n, edge_set, max_iter=50000, deadline=None):
    """
    Iterative local search: try to swap edges to reduce max triangle degree sum.
    """
    deg = _compute_degrees(adj)
    triangles = _find_all_triangles(adj, n)
    if not triangles:
        return adj, edge_set
    
    current_max = max(deg[u] + deg[v] + deg[w] for u, v, w in triangles)
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edge_set)
    best_max = current_max
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if time.time() - start_time > 110:
            break
        
        # Find the worst triangle
        worst_tri = None
        worst_sum = -1
        for u, v, w in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > worst_sum:
                worst_sum = s
                worst_tri = (u, v, w)
        
        if worst_tri is None:
            break
        
        u, v, w = worst_tri
        
        # Try to remove one edge from the worst triangle and add a non-edge
        # that doesn't create new high-degree-sum triangles
        candidates = [(u, v), (u, w), (v, w)]
        random.shuffle(candidates)
        
        improved = False
        for rem_u, rem_v in candidates:
            # Try removing edge (rem_u, rem_v)
            adj[rem_u].discard(rem_v)
            adj[rem_v].discard(rem_u)
            edge_set.discard(frozenset((rem_u, rem_v)))
            deg[rem_u] -= 1
            deg[rem_v] -= 1
            
            # Find non-edges to add back (need to maintain exactly 65 edges)
            # We removed 1, need to add 1
            # Try adding edges that don't create triangles with high degree sums
            possible_adds = []
            for a in range(n):
                for b in range(a + 1, n):
                    if b not in adj[a]:
                        # Check if adding this edge creates triangles
                        new_tris = []
                        for c in adj[a] & adj[b]:
                            new_tris.append((a, b, c))
                        # Estimate the max degree sum if we add this edge
                        max_new = 0
                        for x, y, z in new_tris:
                            s = deg[x] + deg[y] + deg[z]
                            if s > max_new:
                                max_new = s
                        possible_adds.append((max_new, a, b))
            
            if not possible_adds:
                # Restore
                adj[rem_u].add(rem_v)
                adj[rem_v].add(rem_u)
                edge_set.add(frozenset((rem_u, rem_v)))
                deg[rem_u] += 1
                deg[rem_v] += 1
                continue
            
            # Sort by estimated max new triangle degree sum
            possible_adds.sort()
            
            # Try the best few candidates
            for est_max, add_u, add_v in possible_adds[:5]:
                # Add the edge
                adj[add_u].add(add_v)
                adj[add_v].add(add_u)
                edge_set.add(frozenset((add_u, add_v)))
                deg[add_u] += 1
                deg[add_v] += 1
                
                # Recompute triangles and max
                new_triangles = _find_all_triangles(adj, n)
                if not new_triangles:
                    new_max = 0
                else:
                    new_max = max(deg[x] + deg[y] + deg[z] for x, y, z in new_triangles)
                
                if new_max < best_max:
                    best_max = new_max
                    best_adj = [set(s) for s in adj]
                    best_edge_set = set(edge_set)
                    triangles = new_triangles
                    improved = True
                    break
                else:
                    # Undo
                    adj[add_u].discard(add_v)
                    adj[add_v].discard(add_u)
                    edge_set.discard(frozenset((add_u, add_v)))
                    deg[add_u] -= 1
                    deg[add_v] -= 1
            
            if improved:
                break
            
            # Restore removed edge
            adj[rem_u].add(rem_v)
            adj[rem_v].add(rem_u)
            edge_set.add(frozenset((rem_u, rem_v)))
            deg[rem_u] += 1
            deg[rem_v] += 1
        
        if not improved:
            # Try a random perturbation
            if random.random() < 0.3:
                # Randomly remove an edge from a triangle and add a random non-edge
                if triangles:
                    u, v, w = random.choice(triangles)
                    rem_u, rem_v = random.choice([(u, v), (u, w), (v, w)])
                    adj[rem_u].discard(rem_v)
                    adj[rem_v].discard(rem_u)
                    edge_set.discard(frozenset((rem_u, rem_v)))
                    deg[rem_u] -= 1
                    deg[rem_v] -= 1
                    
                    # Find all non-edges
                    non_edges = []
                    for a in range(n):
                        for b in range(a + 1, n):
                            if b not in adj[a]:
                                non_edges.append((a, b))
                    
                    if non_edges:
                        add_u, add_v = random.choice(non_edges)
                        adj[add_u].add(add_v)
                        adj[add_v].add(add_u)
                        edge_set.add(frozenset((add_u, add_v)))
                        deg[add_u] += 1
                        deg[add_v] += 1
                        
                        triangles = _find_all_triangles(adj, n)
                        if triangles:
                            new_max = max(deg[x] + deg[y] + deg[z] for x, y, z in triangles)
                            if new_max < best_max:
                                best_max = new_max
                                best_adj = [set(s) for s in adj]
                                best_edge_set = set(edge_set)
    
    return best_adj, best_edge_set

def _initial_graph(n=16, target_edges=65):
    """
    Start with K_{8,8} + 1 edge, then try to optimize.
    """
    # K_{8,8}: vertices 0-7 and 8-15, all cross edges
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    # Add one intra-part edge to get 65 edges
    edges.append([0, 1])
    
    adj = _build_adjacency(n, edges)
    edge_set = _edge_set(edges)
    
    return adj, edge_set

def construct():
    n = 16
    target_edges = 65
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Start with the base construction
    adj, edge_set = _initial_graph(n, target_edges)
    
    # Run local search
    adj, edge_set = _try_improve(adj, n, edge_set, max_iter=100000, deadline=deadline)
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Verify
    assert len(edges) == target_edges, f"Expected {target_edges} edges, got {len(edges)}"
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
