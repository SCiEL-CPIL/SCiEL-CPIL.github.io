import time
import random
from itertools import combinations

def _check_2colorable(n, edges):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation."""
    if not edges:
        return True
    
    # Convert edges to frozensets for fast lookup
    edge_sets = [frozenset(e) for e in edges]
    
    # State: color assignment, -1 = unassigned
    colors = [-1] * n
    
    def propagate():
        """Unit propagation: if an edge has all but one vertex the same color, force the last."""
        changed = True
        while changed:
            changed = False
            for es in edge_sets:
                verts = list(es)
                assigned = [v for v in verts if colors[v] != -1]
                unassigned = [v for v in verts if colors[v] == -1]
                
                if len(unassigned) == 1 and len(assigned) == 4:
                    # Check if all assigned have same color
                    colors_assigned = [colors[v] for v in assigned]
                    if all(c == colors_assigned[0] for c in colors_assigned):
                        # Must be opposite color
                        colors[unassigned[0]] = 1 - colors_assigned[0]
                        changed = True
                elif len(unassigned) == 0:
                    # All assigned - check if monochromatic
                    colors_assigned = [colors[v] for v in assigned]
                    if all(c == colors_assigned[0] for c in colors_assigned):
                        return False
        return True
    
    def backtrack():
        # Find unassigned vertex with most constraints (MRV heuristic)
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return propagate()
        
        # Pick vertex in most edges
        def count_edges(v):
            count = 0
            for es in edge_sets:
                if v in es:
                    count += 1
            return count
        
        v = max(unassigned, key=count_edges)
        
        for color in [0, 1]:
            colors[v] = color
            if propagate():
                if backtrack():
                    return True
            colors[v] = -1
        
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Known construction: try to find a small non-2-colorable 5-uniform hypergraph
    # Start with a candidate and verify
    
    # Try n=13 with a specific edge set
    # This is based on known minimal non-2-colorable hypergraphs
    
    # Let's try a systematic search for small n
    best_edges = None
    best_n = None
    
    for n in range(9, 15):
        if time.time() > deadline - 5:
            break
        
        # Try random constructions and local search
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random edges
            num_edges = random.randint(15, 40)
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            edges = [list(e) for e in all_possible[:num_edges]]
            
            # Check if non-2-colorable
            if not _check_2colorable(n, edges):
                if best_edges is None or len(edges) < len(best_edges):
                    best_edges = edges
                    best_n = n
                    # Try to reduce edges
                    reduced = True
                    while reduced:
                        reduced = False
                        for i in range(len(best_edges)):
                            if time.time() > deadline - 10:
                                break
                            candidate = best_edges[:i] + best_edges[i+1:]
                            if candidate and not _check_2colorable(best_n, candidate):
                                best_edges = candidate
                                reduced = True
                        if time.time() > deadline - 10:
                            break
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: use the complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
