"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(edges, n):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation."""
    # edges: list of tuples of vertices
    # Return True if 2-colorable, False otherwise
    
    # Precompute edge incidence
    edge_sets = [frozenset(e) for e in edges]
    
    # For each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: color[v] = -1 (unassigned), 0 (red), 1 (blue)
    color = [-1] * n
    
    def propagate():
        """Unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_sets):
                counts = [0, 0]
                unassigned = []
                for v in e:
                    if color[v] == -1:
                        unassigned.append(v)
                    else:
                        counts[color[v]] += 1
                # If 4 of one color, the 5th must be opposite
                if len(unassigned) == 1:
                    if counts[0] == 4:
                        if color[unassigned[0]] == 1:
                            return False
                        color[unassigned[0]] = 0
                        changed = True
                    elif counts[1] == 4:
                        if color[unassigned[0]] == 0:
                            return False
                        color[unassigned[0]] = 1
                        changed = True
        return True
    
    def backtrack():
        if not propagate():
            return False
        
        # Find unassigned vertex with highest degree (MRV heuristic)
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            # All assigned, check all edges
            for e in edge_sets:
                colors = [color[v] for v in e]
                if len(set(colors)) == 1:
                    return False
            return True
        
        # Pick vertex with most edges
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        # Try both colors
        for c in [0, 1]:
            saved = color[v]
            color[v] = c
            if backtrack():
                return True
            color[v] = saved
        
        return False
    
    return backtrack()


def construct():
    """Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Known construction: complete 5-uniform hypergraph on 9 vertices
    # C(9,5) = 126 edges, but we can do better
    # Let's try to find a smaller construction
    
    # Strategy: Start with a known valid hypergraph and try to remove edges
    # while maintaining non-2-colorability
    
    # Initial: complete 5-uniform on 9 vertices
    n = 9
    all_edges = list(combinations(range(n), 5))
    edges = list(all_edges)
    
    # Verify it's not 2-colorable
    if _is_2_colorable(edges, n):
        # Should never happen for complete hypergraph on 9 vertices
        # but just in case, return as-is
        return {'n': n, 'edges': [list(e) for e in edges]}
    
    best_edges = edges
    best_count = len(edges)
    
    # Try to remove edges one by one (greedy removal)
    random.seed(42)
    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    
    for idx in edge_indices:
        if time.time() > deadline:
            break
        
        if idx >= len(edges):
            continue
            
        # Try removing this edge
        test_edges = edges[:idx] + edges[idx+1:]
        
        if not _is_2_colorable(test_edges, n):
            # Still not 2-colorable, so we can remove this edge
            edges = test_edges
            best_edges = edges
            best_count = len(edges)
    
    # If we have time, try more aggressive optimization
    if time.time() < deadline - 5:
        # Try local search: swap edges
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(edges)):
                if time.time() > deadline:
                    break
                for j in range(len(all_edges)):
                    if time.time() > deadline:
                        break
                    if all_edges[j] in edges:
                        continue
                    
                    # Try replacing edges[i] with all_edges[j]
                    test_edges = edges[:i] + [all_edges[j]] + edges[i+1:]
                    if not _is_2_colorable(test_edges, n):
                        edges = test_edges
                        best_edges = edges
                        best_count = len(edges)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return {'n': n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
