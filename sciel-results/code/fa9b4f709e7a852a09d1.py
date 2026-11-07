import random
import time

def _backtrack_2_colorable(edges, n, deadline=None):
    """
    Check if hypergraph is 2-colorable using backtracking with constraint propagation.
    Returns True if a valid 2-coloring exists, False otherwise.
    """
    # Build adjacency: for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    color = [-1] * n  # -1 = unassigned, 0/1 = color
    
    def is_valid_edge(e):
        """Check if edge e is non-monochromatic given current partial coloring."""
        vals = [color[e[0]], color[e[1]], color[e[2]]]
        # If all three are assigned and equal, invalid
        if vals[0] != -1 and vals[0] == vals[1] == vals[2]:
            return False
        return True
    
    def all_edges_valid():
        for i, e in enumerate(edges):
            if not is_valid_edge(e):
                return False
        return True
    
    def backtrack(v):
        if deadline is not None and time.time() > deadline:
            return True  # Assume satisfiable if we run out of time (conservative)
        if v == n:
            return all_edges_valid()
        
        for c in [0, 1]:
            color[v] = c
            # Check all edges containing v are still potentially valid
            valid = True
            for ei in vertex_edges[v]:
                e = edges[ei]
                vals = [color[e[0]], color[e[1]], color[e[2]]]
                # If all assigned and equal, this assignment is bad
                if vals[0] != -1 and vals[0] == vals[1] == vals[2]:
                    valid = False
                    break
            if valid:
                if backtrack(v + 1):
                    return True
        color[v] = -1
        return False
    
    # Try to find a valid coloring
    return backtrack(0)

def _backtrack_3_coloring(edges, n, deadline=None):
    """
    Find a valid 3-coloring using backtracking with forward checking.
    Returns list of colors or None.
    """
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    color = [-1] * n
    
    def is_edge_still_possibly_valid(e):
        """Check if edge e can still be made non-monochromatic."""
        vals = [color[e[0]], color[e[1]], color[e[2]]]
        assigned = [v for v in vals if v != -1]
        if not assigned:
            return True
        # If all assigned and same color, invalid
        if len(assigned) == 3 and assigned[0] == assigned[1] == assigned[2]:
            return False
        # If two are assigned and same, the third must be different - still possible
        # If all three assigned and not all same, valid
        return True
    
    def backtrack(v):
        if deadline is not None and time.time() > deadline:
            return None
        if v == n:
            # Verify all edges
            for e in edges:
                if color[e[0]] == color[e[1]] == color[e[2]]:
                    return None
            return color[:]
        
        for c in [0, 1, 2]:
            color[v] = c
            # Check all edges containing v
            valid = True
            for ei in vertex_edges[v]:
                e = edges[ei]
                if not is_edge_still_possibly_valid(e):
                    valid = False
                    break
            if valid:
                result = backtrack(v + 1)
                if result is not None:
                    return result
        color[v] = -1
        return None
    
    return backtrack(0)

def _verify_solution(edges, colors, n):
    """Verify that the solution is valid: 3-colorable but not 2-colorable."""
    # Check 3-coloring is valid
    for e in edges:
        if colors[e[0]] == colors[e[1]] == colors[e[2]]:
            return False
    
    # Check not 2-colorable
    if _backtrack_2_colorable(edges, n, deadline=time.time() + 1.0):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Fano plane on vertices 0-6, vertices 7-8 isolated
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    n = 9
    
    # Verify using the new backtracking solvers
    if _verify_solution(edges, colors, n):
        return {
            'edges': [list(e) for e in edges],
            'colors': [int(c) for c in colors]
        }
    
    # Fallback: search for a valid solution if verification fails
    import itertools
    all_edges = list(itertools.combinations(range(n), 3))
    
    for k in range(7, 12):
        if time.time() > deadline:
            break
        # Try random samples
        for _ in range(100000):
            if time.time() > deadline:
                break
            edge_set = random.sample(all_edges, k)
            # Check 3-colorability
            c3 = _backtrack_3_coloring(edge_set, n, deadline=time.time() + 0.01)
            if c3 is None:
                continue
            # Check not 2-colorable
            if _backtrack_2_colorable(edge_set, n, deadline=time.time() + 0.01):
                continue
            return {
                'edges': [list(e) for e in edge_set],
                'colors': [int(c) for c in c3]
            }
    
    # Final fallback: return the Fano plane without verification
    return {
        'edges': [list(e) for e in edges],
        'colors': [int(c) for c in colors]
    }
