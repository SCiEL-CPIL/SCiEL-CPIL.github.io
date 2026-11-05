import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    # Check if hypergraph is 2-colorable
    # Returns True if 2-colorable, False otherwise
    # Uses DPLL with unit propagation
    
    # Pre-process: if any edge is size < 2, it's trivially 2-colorable? No, edges are size 5.
    # If n < 5, no edges, so 2-colorable.
    if not edges:
        return True
        
    # Map vertices to indices for fast access
    # We represent the coloring as a list of -1, 0, 1
    # We need to check if there is an assignment such that no edge is monochromatic.
    
    # Convert edges to sets of indices for faster intersection?
    # Or just lists.
    
    def propagate(coloring):
        # Unit propagation
        # If an edge has 4 vertices of color 0, the 5th must be 1.
        # If an edge has 4 vertices of color 1, the 5th must be 0.
        # If an edge has all 5 vertices of same color, contradiction (return None).
        changed = True
        while changed:
            changed = False
            for e in edges:
                colors = [coloring[v] for v in e]
                # Check for contradiction
                if all(c == 0 for c in colors if c != -1) and -1 not in colors:
                    return None
                if all(c == 1 for c in colors if c != -1) and -1 not in colors:
                    return None
                
                # Check for unit propagation
                # Count 0s and 1s
                count_0 = sum(1 for c in colors if c == 0)
                count_1 = sum(1 for c in colors if c == 1)
                count_unknown = sum(1 for c in colors if c == -1)
                
                if count_unknown == 1:
                    if count_0 == 4:
                        # The unknown must be 1
                        v = e[colors.index(-1)]
                        if coloring[v] == -1:
                            coloring[v] = 1
                            changed = True
                        elif coloring[v] == 0:
                            return None
                    elif count_1 == 4:
                        # The unknown must be 0
                        v = e[colors.index(-1)]
                        if coloring[v] == -1:
                            coloring[v] = 0
                            changed = True
                        elif coloring[v] == 1:
                            return None
        return coloring

    def backtrack(coloring):
        # Find an unassigned vertex
        unassigned = [i for i in range(n) if coloring[i] == -1]
        if not unassigned:
            # All assigned, check if valid
            # Actually, propagation should have caught contradictions.
            # But we need to ensure no edge is monochromatic.
            # If we reached here, it means no propagation triggered, 
            # but we still need to verify the final state.
            for e in edges:
                colors = [coloring[v] for v in e]
                if all(c == 0 for c in colors):
                    return False
                if all(c == 1 for c in colors):
                    return False
            return True
            
        # Choose a variable
        # Heuristic: pick the one in the most constrained edge?
        # For simplicity, pick the first one.
        v = unassigned[0]
        
        # Try 0
        coloring_copy = coloring[:]
        coloring_copy[v] = 0
        propagated = propagate(coloring_copy)
        if propagated is not None:
            if backtrack(propagated):
                return True
                
        # Try 1
        coloring_copy = coloring[:]
        coloring_copy[v] = 1
        propagated = propagate(coloring_copy)
        if propagated is not None:
            if backtrack(propagated):
                return True
                
        return False

    coloring = [-1] * n
    propagated = propagate(coloring)
    if propagated is None:
        return False
    return backtrack(propagated)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Start with K_9^(5)
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Greedy removal
    # Sort edges randomly to break ties
    random.seed(42)
    
    improved = False
    while time.time() < deadline:
        improved = False
        # Shuffle edges to try different removal orders
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            
            # Try removing current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if still non-2-colorable
            if not is_2_colorable(n, test_edges):
                current_edges = test_edges
                improved = True
                break # Restart the loop with the new list
        
        if not improved:
            # Try a different random start or restart?
            # If stuck, try starting from a random subset?
            # Or just break.
            # Let's try to restart the greedy process with a different shuffle.
            # Actually, if no edge can be removed, we are at a local minimum.
            # We can try to swap edges? No, we want to minimize count.
            # Let's try a different random seed for the shuffle.
            random.seed()
            
    return {'n': n, 'edges': current_edges}
