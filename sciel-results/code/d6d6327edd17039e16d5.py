"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges):
    """Check if hypergraph is non-2-colorable using backtracking with unit propagation."""
    edge_set = [tuple(sorted(e)) for e in edges]
    edge_set = list(set(edge_set))
    
    # Map vertices to list of edges containing them
    vert_to_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
    
    def backtrack(assignment, edge_status):
        """
        assignment: dict vertex -> 0 or 1, or None
        edge_status: dict edge_idx -> 'ok', 'mono0', 'mono1', 'dead'
        """
        # Check for dead edges
        for i in range(len(edge_set)):
            if edge_status.get(i) == 'dead':
                return False
        
        # Unit propagation: if an edge has 4 vertices of same color, 5th must be opposite
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                if edge_status.get(i) == 'dead':
                    continue
                colors = [assignment.get(v) for v in e]
                unknown = [v for v, c in zip(e, colors) if c is None]
                if len(unknown) == 1:
                    # Count colors of known vertices
                    known_colors = [c for c in colors if c is not None]
                    if len(known_colors) == 4:
                        if known_colors.count(0) == 4:
                            # All 4 are 0, 5th must be 1
                            if assignment.get(unknown[0]) == 0:
                                edge_status[i] = 'dead'
                                return False
                            elif assignment.get(unknown[0]) is None:
                                assignment[unknown[0]] = 1
                                changed = True
                        elif known_colors.count(1) == 4:
                            if assignment.get(unknown[0]) == 1:
                                edge_status[i] = 'dead'
                                return False
                            elif assignment.get(unknown[0]) is None:
                                assignment[unknown[0]] = 0
                                changed = True
                # Check if edge is already monochromatic
                if all(c is not None for c in colors):
                    if all(c == 0 for c in colors):
                        edge_status[i] = 'mono0'
                    elif all(c == 1 for c in colors):
                        edge_status[i] = 'mono1'
        
        # Find unassigned vertex
        unassigned = [v for v in range(n) if assignment.get(v) is None]
        if not unassigned:
            # All assigned, check no monochromatic edges
            for i, e in enumerate(edge_set):
                colors = [assignment[v] for v in e]
                if len(set(colors)) == 1:
                    return False
            return True
        
        # Pick vertex with most constraints (MRV heuristic)
        v = min(unassigned, key=lambda x: len(vert_to_edges[x]))
        
        # Try both colors
        for color in [0, 1]:
            new_assignment = dict(assignment)
            new_assignment[v] = color
            new_edge_status = dict(edge_status)
            
            # Check if this assignment makes any edge dead
            valid = True
            for i, e in enumerate(edge_set):
                if v in e:
                    colors = [new_assignment.get(u) for u in e]
                    known = [c for c in colors if c is not None]
                    if len(known) == 5:
                        if len(set(known)) == 1:
                            valid = False
                            break
                    elif len(known) == 4:
                        if len(set(known)) == 1:
                            # 4 same, 1 unknown - not dead yet
                            pass
                        # Check if we have 4 of one color and the 5th is assigned to same
                        # Already handled above
                    # More careful: count
                    if len(known) >= 4:
                        if known.count(0) == len(known) and len(known) == 4:
                            # 4 zeros, one unknown - ok
                            pass
                        elif known.count(1) == len(known) and len(known) == 4:
                            pass
                        elif len(known) == 5:
                            if known.count(0) == 5 or known.count(1) == 5:
                                valid = False
                                break
            
            if valid:
                if backtrack(new_assignment, new_edge_status):
                    return True
        
        return False
    
    return not backtrack({}, {})

def construct():
    """Return a non-2-colorable 5-uniform hypergraph with minimal edges."""
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Known good construction: try to find one with fewer edges
    # Start with a candidate and verify
    # Use a known small non-2-colorable 5-uniform hypergraph
    
    # Let's try n=15 with a structured set of edges
    # One approach: take all 5-subsets of a 9-vertex set that contain at least 
    # 3 vertices from a specific 5-vertex subset... this gets complicated.
    
    # Simpler: use randomized search to find small non-2-colorable hypergraphs
    best_edges = None
    best_n = 9
    
    # First, try some structured constructions
    # Complete hypergraph on 9 vertices has 126 edges - too many
    
    # Try: for n vertices, add edges randomly and check
    for n in range(10, 21):
        if time.time() > deadline - 30:
            break
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            # Generate random edges
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            edges = []
            for e in all_edges:
                edges.append(list(e))
                if len(edges) >= 30:
                    break
            if _is_non_2colorable(n, edges):
                if best_edges is None or len(edges) < len(best_edges):
                    best_edges = edges
                    best_n = n
                    print(f"Found non-2-colorable with n={n}, edges={len(edges)}")
                break  # Found one for this n, try smaller edge count
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: complete hypergraph on 9 vertices
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
