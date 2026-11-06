import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    # Precompute edge indices
    edge_indices = [tuple(e) for e in edges]
    
    # Color assignments: -1 unassigned, 0/1 assigned
    colors = [-1] * n
    
    # Adjacency: for each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_indices):
        for v in e:
            vertex_edges[v].append(i)
    
    # Counters for colors in each edge
    edge_color_counts = [[0, 0] for _ in range(len(edge_indices))]
    
    def propagate(v, c):
        """
        Assign color c to vertex v and propagate implications.
        Returns False if contradiction is found.
        """
        if colors[v] != -1:
            return colors[v] == c
        colors[v] = c
        edge_color_counts[v][c] += 1 # Wait, this logic is wrong. 
        # Better to just check edges containing v.
        
        # Re-do propagation logic properly
        # We need to check all edges containing v.
        # If an edge has 4 vertices of color 1-c, the last one must be c.
        # If an edge has 5 vertices of color c, contradiction.
        
        # Actually, let's implement a simpler check inside the backtrack.
        # The "unit propagation" can be done by checking all edges affected by v.
        
        return True

    # Let's use a standard backtracking with a check function.
    # To be efficient, we can use the fact that if an edge has 4 vertices of one color, the 5th is forced.
    
    def check_and_propagate():
        """
        Check for contradictions and apply unit propagation.
        Returns (success, changed) where changed is a list of (v, c) to assign.
        """
        changed = []
        # We need to iterate until no more changes.
        # For simplicity in this small n, we can just check all edges.
        for i, e in enumerate(edge_indices):
            # Count colors in this edge
            c0 = 0
            c1 = 0
            unassigned = []
            for v in e:
                if colors[v] == 0:
                    c0 += 1
                elif colors[v] == 1:
                    c1 += 1
                else:
                    unassigned.append(v)
            
            if c0 == 5 or c1 == 5:
                return False, [] # Contradiction
            
            if c0 == 4 and len(unassigned) == 1:
                # The unassigned vertex must be 1
                if colors[unassigned[0]] != 1:
                    if colors[unassigned[0]] == -1:
                        changed.append((unassigned[0], 1))
                    else:
                        return False, []
            elif c1 == 4 and len(unassigned) == 1:
                # The unassigned vertex must be 0
                if colors[unassigned[0]] != 0:
                    if colors[unassigned[0]] == -1:
                        changed.append((unassigned[0], 0))
                    else:
                        return False, []
        
        return True, changed

    def backtrack(idx):
        if idx == n:
            # Check if all edges are valid (non-monochromatic)
            # Actually, if we propagated correctly, they should be.
            # But we must ensure no edge is monochromatic.
            for e in edge_indices:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try color 0
        old_color = colors[idx]
        colors[idx] = 0
        
        # Propagate
        success, changes = check_and_propagate()
        if success:
            # Apply changes
            for v, c in changes:
                if colors[v] != -1:
                    # Should have been handled
                    pass
                colors[v] = c
            # Note: Applying changes might cause new contradictions, 
            # so we should loop propagate or just let backtrack handle it.
            # For small n, this is fine.
            
            if backtrack(idx + 1):
                return True
        
        # Undo changes? 
        # This is tricky. It's better to use a recursive function that takes a state.
        
        return False

    # The above implementation is flawed due to state management.
    # Let's use a cleaner backtracking with explicit state copying or careful undo.
    # Given n <= 26, copying state is cheap.
    
    def solve(colors_state):
        # Check for propagation
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_indices):
                c0 = sum(1 for v in e if colors_state[v] == 0)
                c1 = sum(1 for v in e if colors_state[v] == 1)
                unassigned = [v for v in e if colors_state[v] == -1]
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(unassigned) == 1:
                    if colors_state[unassigned[0]] != 1:
                        if colors_state[unassigned[0]] == -1:
                            colors_state[unassigned[0]] = 1
                            changed = True
                        else:
                            return False
                elif c1 == 4 and len(unassigned) == 1:
                    if colors_state[unassigned[0]] != 0:
                        if colors_state[unassigned[0]] == -1:
                            colors_state[unassigned[0]] = 0
                            changed = True
                        else:
                            return False
        
        # Find first unassigned vertex
        idx = -1
        for i in range(n):
            if colors_state[i] == -1:
                idx = i
                break
        
        if idx == -1:
            # All assigned, check final validity
            for e in edge_indices:
                c0 = sum(1 for v in e if colors_state[v] == 0)
                if c0 == 5 or c0 == 0:
                    return False
            return True
        
        # Try 0
        new_state = colors_state.copy()
        new_state[idx] = 0
        if solve(new_state):
            return True
        
        # Try 1
        new_state = colors_state.copy()
        new_state[idx] = 1
        if solve(new_state):
            return True
        
        return False

    return solve([-1] * n)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Try to find a non-2-colorable hypergraph
    # Start with n=10, m=50
    # If not found, try n=11, m=50, etc.
    
    target_n = 10
    target_m = 50
    
    while target_n <= 26 and time.time() < deadline:
        while target_m <= 200 and time.time() < deadline:
            # Try random instances
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Generate random edges
                all_edges = list(combinations(range(target_n), 5))
                random.shuffle(all_edges)
                edges = all_edges[:target_m]
                
                # Check if 2-colorable
                if not is_2_colorable(target_n, edges):
                    return {'n': target_n, 'edges': [list(e) for e in edges]}
            
            # If not found with this m, increase m
            target_m += 5
        
        # If not found with this n, increase n
        target_n += 1
        target_m = 50
        
    # Fallback: return the complete graph on 9 vertices (126 edges)
    # This is guaranteed to be non-2-colorable.
    edges = [list(e) for e in combinations(range(9), 5)]
    return {'n': 9, 'edges': edges}
