import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def is_2_colorable(n, edges):
        """Check if hypergraph has a proper 2-colouring using backtracking with unit propagation.
        Returns True if 2-colourable (SAT), False if not (UNSAT)."""
        m = len(edges)
        if m == 0:
            return True
        
        # Build adjacency: for each vertex, list of edges containing it
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(i)
        
        # assignment: -1 = unassigned, 0 or 1 = colour
        assignment = [-1] * n
        
        # For unit propagation, track for each edge the count of 0s and 1s
        edge_count_0 = [0] * m
        edge_count_1 = [0] * m
        edge_unassigned = [5] * m  # 5 vertices per edge initially unassigned
        
        # Initialize counts
        for i, e in enumerate(edges):
            for v in e:
                if assignment[v] == 0:
                    edge_count_0[i] += 1
                elif assignment[v] == 1:
                    edge_count_1[i] += 1
                else:
                    edge_unassigned[i] -= 1
        
        # Check for initial contradictions
        for i in range(m):
            if edge_count_0[i] == 5 or edge_count_1[i] == 5:
                return False
        
        # Stack for backtracking
        # Each entry: (vertex, color, snapshot of assignment, edge counts)
        # We'll use a simpler approach: recursive with undo log
        
        def propagate(assignment, edge_count_0, edge_count_1, edge_unassigned):
            """Apply unit propagation. Returns False if contradiction, True otherwise."""
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    # If 4 of 5 are same color, 5th must be opposite
                    if edge_count_0[i] == 4 and edge_unassigned[i] == 1:
                        # Find the unassigned vertex in this edge
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 1
                                edge_count_1[i] += 1
                                edge_unassigned[i] -= 1
                                # Update other edges containing v
                                for j in vert_edges[v]:
                                    if j != i:
                                        edge_count_1[j] += 1
                                        edge_unassigned[j] -= 1
                                changed = True
                                break
                    elif edge_count_1[i] == 4 and edge_unassigned[i] == 1:
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 0
                                edge_count_0[i] += 1
                                edge_unassigned[i] -= 1
                                for j in vert_edges[v]:
                                    if j != i:
                                        edge_count_0[j] += 1
                                        edge_unassigned[j] -= 1
                                changed = True
                                break
                    # Check for contradiction
                    if edge_count_0[i] == 5 or edge_count_1[i] == 5:
                        return False
            return True
        
        def backtrack(assignment, edge_count_0, edge_count_1, edge_unassigned):
            """Backtracking search. Returns True if a valid coloring exists."""
            # Check time
            if time.time() > deadline - 1:
                return True  # Assume colorable if we run out of time (safe for our purpose)
            
            # Find unassigned vertex with fewest unassigned edges (MRV heuristic)
            min_unassigned = float('inf')
            min_vertex = -1
            for v in range(n):
                if assignment[v] == -1:
                    # Count how many edges are still "active" (have unassigned vertices)
                    active = 0
                    for e_idx in vert_edges[v]:
                        if edge_unassigned[e_idx] > 0:
                            active += 1
                    if active < min_unassigned:
                        min_unassigned = active
                        min_vertex = v
                    if active == 0:
                        break
            
            if min_vertex == -1:
                return True  # All assigned, no contradiction
            
            v = min_vertex
            # Try both colors
            for color in [0, 1]:
                # Save state
                saved_assign = assignment[v]
                saved_e0 = [edge_count_0[i] for i in vert_edges[v]]
                saved_e1 = [edge_count_1[i] for i in vert_edges[v]]
                saved_eu = [edge_unassigned[i] for i in vert_edges[v]]
                
                # Assign
                assignment[v] = color
                for e_idx in vert_edges[v]:
                    if color == 0:
                        edge_count_0[e_idx] += 1
                    else:
                        edge_count_1[e_idx] += 1
                    edge_unassigned[e_idx] -= 1
                
                # Propagate
                if propagate(assignment, edge_count_0, edge_count_1, edge_unassigned):
                    if backtrack(assignment, edge_count_0, edge_count_1, edge_unassigned):
                        return True
                
                # Undo
                assignment[v] = saved_assign
                for k, e_idx in enumerate(vert_edges[v]):
                    edge_count_0[e_idx] = saved_e0[k]
                    edge_count_1[e_idx] = saved_e1[k]
                    edge_unassigned[e_idx] = saved_eu[k]
            
            return False
        
        # Start backtracking
        if not propagate(assignment, edge_count_0, edge_count_1, edge_unassigned):
            return False
        return backtrack(assignment, edge_count_0, edge_count_1, edge_unassigned)

    def greedy_construct(n, target_edges):
        """Greedily add edges to make hypergraph non-2-colorable."""
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        selected = []
        
        # Use a smarter approach: start with random edges, check if non-colorable
        # If not, add more edges that "break" colorings
        
        # First, try a random set of target_edges edges
        if len(all_edges) >= target_edges:
            candidate = random.sample(all_edges, target_edges)
            if not is_2_colorable(n, candidate):
                return candidate
        
        # Greedy: start empty, add edges that reduce the number of valid colorings
        # But checking all colorings is expensive. Instead, use a different strategy.
        
        # Strategy: Start with a small random set, then add edges one by one
        # that are most "constraining"
        
        current_edges = []
        # Start with a few random edges
        initial_size = min(10, len(all_edges))
        if initial_size > 0:
            current_edges = list(random.sample(all_edges, initial_size))
        
        # Check if already non-colorable
        if not is_2_colorable(n, current_edges):
            return current_edges
        
        # Add edges one by one
        remaining = [e for e in all_edges if e not in current_edges]
        random.shuffle(remaining)
        
        for edge in remaining:
            if len(current_edges) >= target_edges:
                break
            if time.time() > deadline - 10:
                break
            
            new_edges = current_edges + [edge]
            if not is_2_colorable(n, new_edges):
                current_edges = new_edges
                if len(current_edges) <= target_edges:
                    return current_edges
        
        return current_edges if not is_2_colorable(n, current_edges) else None

    best_result = None
    best_m = 200

    # Try different n values and target edge counts
    # Known best is 51 edges. Try to find <= 51.
    
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline - 20:
            break
        
        # Try target of 51 edges first
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            result = greedy_construct(n, 51)
            if result is not None:
                m = len(result)
                if m < best_m:
                    best_m = m
                    best_result = (n, result)
                if m <= 51:
                    return {'n': n, 'edges': [list(e) for e in result]}
        
        # If we haven't found 51, try smaller targets to see if we can get lucky
        for target in [30, 40, 45, 50]:
            if time.time() > deadline - 5:
                break
            for trial in range(10):
                if time.time() > deadline - 3:
                    break
                result = greedy_construct(n, target)
                if result is not None:
                    m = len(result)
                    if m < best_m:
                        best_m = m
                        best_result = (n, result)
                    if m <= 51:
                        return {'n': n, 'edges': [list(e) for e in result]}

    # Fallback: return the best we found, or a known valid hypergraph
    if best_result is not None:
        n, edges = best_result
        return {'n': n, 'edges': [list(e) for e in edges]}
    
    # Absolute fallback: complete hypergraph on 9 vertices (126 edges)
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
