"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Strategy:
    1. Start with a known small non-2-colorable 5-uniform hypergraph (e.g., complete on 9 vertices = 126 edges).
    2. Use local search / edge removal to minimize edges while preserving non-2-colorability.
    3. Verify non-2-colorability exactly via backtracking with unit propagation.
    
    We use a deadline guard to ensure we finish within 150s wall-clock.
    """
    deadline = time.time() + 140  # 10s safety margin
    
    def is_2_colorable(n, edges):
        """
        Check if hypergraph is 2-colorable using backtracking with unit propagation.
        Returns True if 2-colorable, False otherwise.
        """
        if not edges:
            return True
        
        # Preprocess: remove duplicate edges
        edges = list(set(tuple(sorted(e)) for e in edges))
        if not edges:
            return True
        
        n_edges = len(edges)
        
        # Unit propagation + backtracking
        # color[v] = -1 unassigned, 0 or 1 assigned
        # For each edge, we need at least one vertex of each color.
        # Equivalent: no edge is monochromatic.
        
        sys.setrecursionlimit(100000)
        
        def backtrack(color, edge_idx):
            # Try to assign colors to remaining edges
            if edge_idx == n_edges:
                return True
            
            edge = edges[edge_idx]
            # Check if this edge is already satisfied or violated
            # An edge is violated if all its vertices have the same color
            # An edge is satisfied if it has both colors
            
            # Check current state of this edge
            colors_in_edge = [color[v] for v in edge]
            
            # If all assigned and monochromatic -> fail
            if all(c != -1 for c in colors_in_edge):
                if all(c == colors_in_edge[0] for c in colors_in_edge):
                    return False
                # Otherwise, edge is satisfied, move on
                return backtrack(color, edge_idx + 1)
            
            # If some assigned, check if edge can still be satisfied
            # If all unassigned vertices can't fix a monochromatic prefix...
            # Actually, let's use a simpler approach: assign vertices one by one
            
            # Find the first unassigned vertex in the edge
            # Or better: use the standard approach of assigning vertices globally
            
            return True  # placeholder, we'll use vertex-based backtracking
        
        # Better approach: vertex-based backtracking with unit propagation
        # We'll assign colors to vertices 0, 1, ..., n-1 in order
        # and use unit propagation to detect contradictions early
        
        # Precompute for each edge which vertices it contains
        # Use a more efficient backtracking
        
        def solve():
            color = [-1] * n
            # Order vertices by degree (number of edges they're in)
            deg = [0] * n
            for e in edges:
                for v in e:
                    deg[v] += 1
            
            # Vertex ordering: highest degree first
            vertex_order = sorted(range(n), key=lambda v: -deg[v])
            
            # For each edge, track how many vertices are colored 0 and 1
            edge_state = []
            for e in edges:
                edge_state.append([0, 0, len(e)])  # [count0, count1, uncolored]
            
            def propagate(color, edge_state, vertex_order, pos):
                """
                Unit propagation: if an edge has all but one vertex colored the same,
                the last must be the other color.
                If an edge is monochromatic, fail.
                """
                changed = True
                while changed:
                    changed = False
                    for i, e in enumerate(edges):
                        c0, c1, unc = edge_state[i]
                        if unc == 0:
                            # All colored
                            if c0 > 0 and c1 > 0:
                                continue  # satisfied
                            else:
                                return False  # monochromatic
                        elif unc == 1:
                            # One uncolored
                            if c0 > 0 and c1 == 0:
                                # Must color the last vertex 1
                                for v in e:
                                    if color[v] == -1:
                                        if color[v] == 1:
                                            return False
                                        color[v] = 1
                                        edge_state[i] = [c0, 1, 0]
                                        changed = True
                                        break
                            elif c1 > 0 and c0 == 0:
                                # Must color the last vertex 0
                                for v in e:
                                    if color[v] == -1:
                                        if color[v] == 0:
                                            return False
                                        color[v] = 0
                                        edge_state[i] = [1, c1, 0]
                                        changed = True
                                        break
                            else:
                                # Both c0 and c1 are 0, but unc==1 means one vertex uncolored
                                # This edge can still be satisfied
                                pass
                        else:
                            # More than one uncolored
                            if c0 > 0 and c1 > 0:
                                # Already satisfied
                                pass
                            # else: can still be satisfied
                    # Check for contradictions: if a vertex is forced to two colors
                    # This is handled by the assignment checks above
                return True
            
            def backtrack(pos):
                if pos == len(vertex_order):
                    # All vertices colored, check all edges
                    for i, e in enumerate(edges):
                        c0, c1, unc = edge_state[i]
                        if unc == 0 and not (c0 > 0 and c1 > 0):
                            return False
                    return True
                
                v = vertex_order[pos]
                if color[v] != -1:
                    return backtrack(pos + 1)
                
                for c in [0, 1]:
                    # Try color c
                    color[v] = c
                    # Update edge states
                    saved_states = []
                    ok = True
                    for i, e in enumerate(edges):
                        if v in e:
                            c0, c1, unc = edge_state[i]
                            new_c0 = c0 + (1 if c == 0 else 0)
                            new_c1 = c1 + (1 if c == 1 else 0)
                            new_unc = unc - 1
                            saved_states.append((i, (c0, c1, unc)))
                            edge_state[i] = [new_c0, new_c1, new_unc]
                    
                    if not propagate(color, edge_state, vertex_order, pos):
                        # Undo
                        color[v] = -1
                        for i, state in saved_states:
                            edge_state[i] = list(state)
                        continue
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    color[v] = -1
                    for i, state in saved_states:
                        edge_state[i] = list(state)
                
                return False
            
            return backtrack(0)
        
        return solve()
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = list(all_edges)
    
    # Verify initial hypergraph is not 2-colorable
    if is_2_colorable(n, current_edges):
        # Should not happen for complete hypergraph on 9 vertices
        return {'n': n, 'edges': [list(e) for e in current_edges]}
    
    best_edges = current_edges[:]
    
    # Try to remove edges while preserving non-2-colorability
    # Greedy edge removal with randomization
    rng = random.Random(42)
    
    # Create edge list with indices
    edge_set = set(tuple(sorted(e)) for e in current_edges)
    
    # Try to remove edges one by one
    # Use a heuristic: remove edges that are "redundant"
    
    # We'll do multiple passes of random edge removal attempts
    max_attempts = 5000
    attempts = 0
    
    while time.time() < deadline and attempts < max_attempts:
        if not current_edges:
            break
        
        # Pick a random edge to try removing
        idx = rng.randrange(len(current_edges))
        edge_to_remove = current_edges[idx]
        
        # Try removing it
        trial_edges = current_edges[:idx] + current_edges[idx+1:]
        
        # Check if still not 2-colorable
        if not is_2_colorable(n, trial_edges):
            current_edges = trial_edges
            if len(current_edges) < len(best_edges):
                best_edges = current_edges[:]
        
        attempts += 1
    
    # Also try a more systematic approach: sort edges and try removing in order
    # But first, let's try local search with edge swaps
    
    # Try to find a smaller construction from scratch using known results
    # The best known is 51 edges. Let's see if we can get close.
    
    # For now, return the best we found
    return {'n': n, 'edges': [list(e) for e in best_edges]}

# EVOLVE-BLOCK-END
