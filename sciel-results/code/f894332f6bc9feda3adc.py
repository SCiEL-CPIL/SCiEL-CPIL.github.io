"""Rewritten program for equal-lcm-triple avoidance using Minimum Vertex Cover B&B."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n + 1))
    
    m = len(bad_triples)
    
    if m == 0:
        return list(range(1, n + 1))
    
    # We want to find the minimum vertex cover in this 3-uniform hypergraph.
    # The size of the maximum independent set is n - min_vertex_cover_size.
    
    # Map vertices to indices 0..n-1 for bitmask operations if needed, 
    # but for B&B on edges, we can just work with the list of edges.
    
    # Precompute for each vertex, the list of edge indices it belongs to.
    # This helps in quickly removing edges when a vertex is selected.
    vert_to_edges = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        vert_to_edges[a].append(idx)
        vert_to_edges[b].append(idx)
        vert_to_edges[c].append(idx)
    
    # Active edges bitmask? m might be too large for int bitmask if m > 64? 
    # Python ints are arbitrary precision, so bitmask is fine.
    # However, iterating over bits is slow. 
    # Alternative: Use a list of active edges and a "covered" flag array?
    # Or use a set of active edge indices.
    
    # Given m is likely small (< 10000), a set of active edges is efficient.
    
    best_cover_size = n  # Upper bound
    best_cover_set = set(range(1, n + 1))
    
    # Heuristic for initial solution: Greedy vertex cover
    active_edges_greedy = set(range(m))
    cover_greedy = set()
    # Count degree of each vertex in active hypergraph
    deg = {i: len(vert_to_edges[i]) for i in range(1, n + 1)}
    
    while active_edges_greedy:
        if time.time() > deadline:
            break
        # Find vertex with max degree
        max_deg = -1
        best_v = None
        for v in range(1, n + 1):
            if deg[v] > max_deg:
                max_deg = deg[v]
                best_v = v
        
        if best_v is None or max_deg == 0:
            break
            
        cover_greedy.add(best_v)
        
        # Remove all edges containing best_v
        edges_to_remove = set(vert_to_edges[best_v]) & active_edges_greedy
        for e in edges_to_remove:
            active_edges_greedy.discard(e)
            a, b, c = bad_triples[e]
            for v in (a, b, c):
                deg[v] -= 1
                if deg[v] < 0: deg[v] = 0 # Safety
                
    if len(cover_greedy) < best_cover_size:
        best_cover_size = len(cover_greedy)
        best_cover_set = set(cover_greedy)
        
    # Branch and Bound
    # State: active_edges (set), current_cover (set)
    # To speed up, we can use a list of edges and a boolean array for active, 
    # but passing sets in recursion is okay for small m.
    
    # Optimization: Order vertices by degree to pick good candidates for cover?
    # In B&B, we pick an edge, then pick a vertex from that edge to add to cover.
    
    def search(active_edges, current_cover_size, current_cover_set):
        nonlocal best_cover_size, best_cover_set
        
        if time.time() > deadline:
            return
        
        if not active_edges:
            if current_cover_size < best_cover_size:
                best_cover_size = current_cover_size
                best_cover_set = set(current_cover_set)
            return
        
        # Pruning
        if current_cover_size >= best_cover_size:
            return
            
        # Lower bound: We need at least 1 more vertex.
        # Better bound: Packing of disjoint edges.
        # If we have k disjoint edges, we need at least k vertices.
        # Computing max matching in 3-uniform hypergraph is hard, 
        # but we can greedily find a set of disjoint edges for a lower bound.
        # Greedy packing:
        remaining = list(active_edges)
        packing = 0
        used_verts = set()
        # Sort edges by size? All size 3.
        # Random order might be better for packing?
        import random
        random.shuffle(remaining)
        for e in remaining:
            a, b, c = bad_triples[e]
            if a not in used_verts and b not in used_verts and c not in used_verts:
                packing += 1
                used_verts.add(a)
                used_verts.add(b)
                used_verts.add(c)
                
        if current_cover_size + packing >= best_cover_size:
            return

        # Pick an edge to branch on
        # Heuristic: Pick edge with vertices having highest degrees in the active subgraph?
        # Or just the first one.
        # Let's pick the edge with the "densest" neighborhood to prune faster.
        # Actually, picking an edge where vertices have high degree is good.
        
        # Calculate "weight" of each edge: sum of degrees of its vertices
        # Degrees need to be updated dynamically or approximated.
        # For simplicity and speed, let's just pick the first edge.
        # Or pick an edge arbitrarily.
        
        e_idx = next(iter(active_edges))
        a, b, c = bad_triples[e_idx]
        
        # Branch on adding a, b, or c to the cover
        
        for v in (a, b, c):
            # New active edges: remove all edges containing v
            new_active = active_edges - set(vert_to_edges[v])
            new_cover_size = current_cover_size + 1
            new_cover_set = current_cover_set | {v}
            
            search(new_active, new_cover_size, new_cover_set)
            
            if time.time() > deadline:
                return

    # Initial active edges
    initial_active = set(range(m))
    search(initial_active, 0, set())
    
    # Construct result: all vertices NOT in best_cover_set
    result = [i for i in range(1, n + 1) if i not in best_cover_set]
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
