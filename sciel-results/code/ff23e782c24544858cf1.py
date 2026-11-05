"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring (no 2-coloring
    leaves every edge with both colours). Minimize number of edges.
    
    Strategy:
    1. Start from a known small non-2-colorable 5-uniform hypergraph.
    2. Try to find smaller ones via randomized search + verification.
    3. Use efficient backtracking with unit propagation for verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit
    
    # First, return a valid solution immediately (complete hypergraph on 9 vertices)
    # This ensures we always have a valid answer.
    from itertools import combinations as _comb
    initial_n = 9
    initial_edges = [list(e) for e in _comb(range(initial_n), 5)]
    best = {'n': initial_n, 'edges': initial_edges}
    best_count = len(initial_edges)  # 126
    
    # Try to find smaller non-2-colorable hypergraphs
    # Known: best known construction has 51 edges.
    # We'll try random search on n=10..14 vertices with 30..60 edges.
    
    # For verification, we use backtracking with unit propagation
    # A hypergraph is NOT 2-colorable iff there's no assignment of colors {0,1}
    # to vertices such that every edge has both colors.
    # Equivalently, every edge must NOT be monochromatic.
    
    def is_2_colorable(n, edges):
        """Check if hypergraph has a proper 2-coloring (every edge has both colors).
        Returns True if 2-colorable, False if not."""
        # For each edge, constraint: not all same color.
        # This is equivalent to: for each edge, at least one vertex is 0 AND at least one is 1.
        
        # Use backtracking with unit propagation
        # Variables: vertex colors (0 or 1)
        # Constraints: for each edge, not all 0 and not all 1
        
        # Build incidence: for each vertex, which edges it's in
        vertex_edges = [[] for _ in range(n)]
        for ei, edge in enumerate(edges):
            for v in edge:
                vertex_edges[v].append(ei)
        
        # Unit propagation: if an edge has all but one vertex assigned to color c,
        # the last must be 1-c (to avoid monochromatic).
        
        assignment = [-1] * n  # -1 = unassigned
        edge_colors = [set() for _ in range(len(edges))]  # colors seen in each edge
        edge_unassigned = [set(edge) for edge in edges]  # unassigned vertices in each edge
        
        def propagate(queue):
            """Process queue of (vertex, color) assignments via unit propagation.
            Returns False if contradiction, True otherwise."""
            while queue:
                v, c = queue.pop()
                if assignment[v] != -1:
                    if assignment[v] != c:
                        return False
                    continue
                assignment[v] = c
                for ei in vertex_edges[v]:
                    if c in edge_colors[ei]:
                        # Edge already has both colors - satisfied
                        pass
                    else:
                        edge_colors[ei].add(c)
                        edge_unassigned[ei].discard(v)
                        # Check if edge is now forced
                        if len(edge_unassigned[ei]) == 1:
                            # One vertex left unassigned in this edge
                            remaining_v = next(iter(edge_unassigned[ei]))
                            if assignment[remaining_v] == -1:
                                # Must be opposite color to avoid mono
                                forced_c = 1 - c
                                queue.append((remaining_v, forced_c))
                        elif len(edge_unassigned[ei]) == 0:
                            # All assigned - check if mono
                            if len(edge_colors[ei]) == 1:
                                return False
            return True
        
        def backtrack(unassigned_vertices):
            """Try to find a valid 2-coloring."""
            if not unassigned_vertices:
                # Check all edges are satisfied (have both colors)
                for ei in range(len(edges)):
                    if len(edge_colors[ei]) < 2:
                        return False
                return True
            
            # Choose vertex with fewest options (most constrained)
            # Use heuristic: pick vertex in most edges
            v = max(unassigned_vertices, key=lambda x: len(vertex_edges[x]))
            
            for c in [0, 1]:
                # Try assigning v = c
                # Save state
                old_assignment = assignment[v]
                old_edge_colors = [set(ec) for ec in edge_colors]
                old_edge_unassigned = [set(eu) for eu in edge_unassigned]
                
                queue = [(v, c)]
                if not propagate(queue):
                    assignment[v] = old_assignment
                    edge_colors[:] = old_edge_colors
                    edge_unassigned[:] = old_edge_unassigned
                    continue
                
                new_unassigned = unassigned_vertices - {v}
                # Remove any that got assigned during propagation
                for u in list(new_unassigned):
                    if assignment[u] != -1:
                        new_unassigned.discard(u)
                
                if backtrack(new_unassigned):
                    return True
                
                # Restore state
                for u in range(n):
                    if assignment[u] != -1 and old_assignment == -1:
                        assignment[u] = -1
                    elif u == v:
                        assignment[u] = old_assignment
                # Simpler: just reset everything for this branch
                assignment[v] = old_assignment
                edge_colors[:] = old_edge_colors
                edge_unassigned[:] = old_edge_unassigned
            
            return False
        
        return backtrack(set(range(n)))
    
    # Random search for smaller non-2-colorable hypergraphs
    rng = random.Random(42)
    
    # Try various n and edge counts
    for n in range(10, 15):
        if time.time() > deadline:
            break
        # Try edge counts from low to high
        for m in range(35, min(65, best_count)):
            if time.time() > deadline:
                break
            # Try a few random instances
            num_tries = 20
            for trial in range(num_tries):
                if time.time() > deadline:
                    break
                
                # Generate random 5-uniform hypergraph
                all_possible = list(_comb(range(n), 5))
                rng.shuffle(all_possible)
                edges = [list(e) for e in all_possible[:m]]
                
                # Check if non-2-colorable
                if not is_2_colorable(n, edges):
                    if m < best_count:
                        best = {'n': n, 'edges': edges}
                        best_count = m
                    break  # Found one for this m, move to next m
    
    # If we found something better, return it; otherwise return initial
    return best

# EVOLVE-BLOCK-END
