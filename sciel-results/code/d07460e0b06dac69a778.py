"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    N = 16
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Bitmask utilities
    def adj_to_set(adj):
        return [set() for _ in range(N)]

    def make_adj(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_triangle_free(adj):
        for u in range(N):
            for v in adj[u]:
                if v > u:
                    # Check if u and v share a common neighbor
                    common = adj[u] & adj[v]
                    if common:
                        return False
        return True

    def is_3_colorable(adj):
        """Backtracking 3-coloring. Returns True if 3-colorable."""
        color = [-1] * N
        # Order vertices by degree descending for faster pruning
        order = sorted(range(N), key=lambda x: -len(adj[x]))
        pos_in_order = {v: i for i, v in enumerate(order)}
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Available colors
            used = set()
            for nb in adj[v]:
                if color[nb] != -1:
                    used.add(color[nb])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def is_4_chromatic(adj):
        """Check if graph is not 3-colorable (i.e., chromatic number >= 4)."""
        return not is_3_colorable(adj)

    # Start with a known 4-chromatic triangle-free graph: Grötzsch graph (11 vertices)
    # Extend to 16 vertices by adding isolated vertices, then try to add edges.
    
    # Grötzsch graph construction
    # Vertices 0-4: cycle C5
    # Vertex 5-9: copies of cycle vertices
    # Vertex 10: connected to 5-9
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Copies: 5+i connected to i and i+1 (mod 5)
    for i in range(5):
        edges.append([5 + i, i])
        edges.append([5 + i, (i + 1) % 5])
    # Vertex 10 connected to 5,6,7,8,9
    for i in range(5):
        edges.append([10, 5 + i])
    
    # Vertices 11-15 are isolated initially
    adj = make_adj(edges)
    
    # Verify initial graph is valid
    assert is_triangle_free(adj), "Initial graph has triangles"
    assert is_4_chromatic(adj), "Initial graph is 3-colorable"
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add edges incrementally
    # All possible edges not yet in graph
    all_pairs = set()
    for u in range(N):
        for v in range(u + 1, N):
            all_pairs.add((u, v))
    
    current_edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    def try_add_edges(adj, edge_set):
        """Try to greedily add edges that maintain triangle-free and 4-chromaticity."""
        adj = [set(s) for s in adj]
        edge_set = set(edge_set)
        
        changed = True
        while changed:
            changed = False
            # Try all possible edges in random order
            candidates = list(all_pairs - edge_set)
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                # Check triangle-free: u and v must not share a common neighbor
                if adj[u] & adj[v]:
                    continue
                # Try adding edge
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                
                # Check 4-chromaticity
                if is_4_chromatic(adj):
                    changed = True
                else:
                    # Revert
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard((u, v))
            
            if time.time() > deadline:
                break
        
        return adj, edge_set

    # Greedy phase
    adj, edge_set = try_add_edges(adj, current_edge_set)
    current_count = len(edge_set)
    if current_count > best_count:
        best_count = current_count
        best_edges = [[u, v] for u, v in edge_set]
    
    # Local search phase: try swapping edges
    # Remove one edge, try to add multiple others
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        adj_copy = [set(s) for s in adj]
        edge_set_copy = set(edge_set)
        
        # Try removing a random edge and re-optimizing
        if not edge_set_copy:
            break
        removed = random.choice(list(edge_set_copy))
        u, v = removed
        adj_copy[u].discard(v)
        adj_copy[v].discard(u)
        edge_set_copy.discard(removed)
        
        # Now try to add edges
        adj_new, edge_set_new = try_add_edges(adj_copy, edge_set_copy)
        new_count = len(edge_set_new)
        
        if new_count > current_count:
            current_count = new_count
            adj = adj_new
            edge_set = edge_set_new
            if current_count > best_count:
                best_count = current_count
                best_edges = [[u, v] for u, v in edge_set]
    
    # Final polish: one more greedy pass
    if time.time() < deadline - 5:
        adj, edge_set = try_add_edges(adj, edge_set)
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u, v in edge_set]
    
    # Verify final result
    final_adj = make_adj(best_edges)
    if not is_triangle_free(final_adj) or not is_4_chromatic(final_adj):
        # Fallback to initial graph
        return edges
    
    return best_edges
# EVOLVE-BLOCK-END
