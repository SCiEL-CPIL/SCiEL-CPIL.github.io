import time
from itertools import combinations

def _check_no_2coloring(edges, n):
    """Check if the hypergraph has no proper 2-colouring using backtracking with unit propagation."""
    if not edges:
        return n < 5  # trivially no edges means 2-colourable unless no vertices, but n>=5
    
    # Build adjacency: for each vertex, list of edges containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # We need to check: does there exist a 2-coloring where every edge has both colors?
    # Equivalent: is the hypergraph NOT 2-colorable in the sense that there's NO coloring where all edges are bichromatic.
    # A hypergraph has Property B if there exists a 2-coloring with no monochromatic edge.
    # We want: NO Property B, i.e., for ALL 2-colorings, some edge is monochromatic.
    # This is equivalent to: the hypergraph is NOT 2-colorable (in the standard sense where 2-colorable means there exists a coloring with no mono edge).
    
    # Backtracking: assign colors 0/1 to vertices, with constraint that no edge is monochromatic.
    # If we find ANY valid coloring, Property B holds (return False for "no 2coloring").
    # If no valid coloring exists, Property B does NOT hold (return True for "no 2coloring").
    
    color = [-1] * n  # -1 unassigned, 0, 1 assigned
    
    def propagate():
        """Unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite."""
        changed = True
        while changed:
            changed = False
            for e in edges:
                # Count colors in this edge
                c0 = c1 = c2 = 0
                unassigned_in_edge = []
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        unassigned_in_edge.append(v)
                if c0 == 4 and len(unassigned_in_edge) == 1:
                    # The unassigned vertex must be 1
                    v = unassigned_in_edge[0]
                    if color[v] == -1:
                        color[v] = 1
                        changed = True
                    elif color[v] == 0:
                        return False  # contradiction
                elif c1 == 4 and len(unassigned_in_edge) == 1:
                    v = unassigned_in_edge[0]
                    if color[v] == -1:
                        color[v] = 0
                        changed = True
                    elif color[v] == 1:
                        return False
        return True
    
    def backtrack(idx):
        """Try to find a valid 2-coloring. Returns True if found (Property B holds)."""
        if idx == n:
            # Check all edges are bichromatic
            for e in edges:
                c0 = sum(1 for v in e if color[v] == 0)
                c1 = sum(1 for v in e if color[v] == 1)
                if c0 == 0 or c1 == 0:
                    return False
            return True
        
        # Try color 0
        color[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = -1
        
        # Try color 1
        color[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = -1
        
        return False
    
    return not backtrack(0)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    base_edges = list(combinations(range(9), 5))
    best_edges = list(base_edges)
    best_count = len(best_edges)
    
    import random
    
    # Greedy deletion with multiple random restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_edges = list(best_edges)
        random.shuffle(current_edges)
        
        # Greedy deletion: try to remove each edge in random order
        kept = []
        for e in current_edges:
            # Try removing e
            candidate = [edge for edge in kept if edge != e] + [e]
            # Actually, we should test: if we remove e from the set of all edges we've seen so far...
            # Simpler approach: maintain the set, try removing each edge one by one
            pass
        
        # Better greedy: maintain current set, try removing each edge
        current_set = set(current_edges)
        for e in current_edges:
            if time.time() > deadline:
                break
            if e not in current_set:
                continue
            candidate_set = current_set - {e}
            if len(candidate_set) < 5:
                continue
            if _check_no_2coloring(list(candidate_set), 9):
                current_set = candidate_set
        
        if len(current_set) < best_count:
            best_count = len(current_set)
            best_edges = list(current_set)
    
    return {'n': 9, 'edges': [list(e) for e in best_edges]}
