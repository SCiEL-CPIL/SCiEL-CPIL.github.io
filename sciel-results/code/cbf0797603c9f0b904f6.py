import itertools
import random

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph has a proper 2-coloring.
    Returns False if it does not (i.e., it is non-2-colorable).
    """
    # Convert edges to frozensets for faster lookup
    edge_set = [frozenset(e) for e in edges]
    
    # colors: -1 unassigned, 0 or 1 assigned
    colors = [-1] * n
    
    # Pre-compute for each vertex, the list of edges it belongs to
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vertex_to_edges[v].append(i)
            
    def propagate():
        """
        Unit propagation.
        Returns False if a contradiction is found.
        Modifies 'colors' and 'queue'.
        """
        queue = []
        # Initialize queue with any forced assignments? 
        # Actually, we just need to check if current state is consistent and propagate.
        # We'll use a simple loop.
        
        # We need to handle the propagation dynamically.
        # Let's use a stack for variables to assign.
        # This function is called when a variable is assigned.
        # For simplicity in this recursive backtracking, we can just check consistency
        # and find forced moves.
        
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                # Check if this edge forces an assignment
                zeros = 0
                ones = 0
                unassigned = -1
                conflict = False
                
                for v in e:
                    if colors[v] == 0:
                        zeros += 1
                    elif colors[v] == 1:
                        ones += 1
                    else:
                        if unassigned != -1:
                            # More than 1 unassigned, cannot force yet
                            # But we need to check for conflicts:
                            # If zeros == 5, conflict (all 0)
                            # If ones == 5, conflict (all 1)
                            # Actually, if we have 5 assigned, we check.
                            # If we have 4 assigned to 0, the 5th must be 1.
                            # If we have 4 assigned to 1, the 5th must be 0.
                            pass
                        unassigned = v
                            
                # Check for immediate conflicts
                if zeros == 5:
                    return False
                if ones == 5:
                    return False
                    
                # Check for forced assignments
                if zeros == 4 and unassigned != -1 and colors[unassigned] == -1:
                    colors[unassigned] = 1
                    changed = True
                elif ones == 4 and unassigned != -1 and colors[unassigned] == -1:
                    colors[unassigned] = 0
                    changed = True
                    
        return True

    def backtrack(idx):
        if idx == n:
            # All vertices assigned. Check if all edges are valid.
            # Actually, propagation should have caught conflicts.
            # But we must ensure no edge is monochromatic.
            for e in edge_set:
                vals = set(colors[v] for v in e)
                if len(vals) == 1:
                    return False
            return True
            
        # Try 0
        if colors[idx] == -1:
            colors[idx] = 0
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
        # Try 1
        if colors[idx] == -1:
            colors[idx] = 1
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
        return False

    # Optimization: Order vertices by degree
    degrees = [len(vertex_to_edges[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Remap indices? No, just use the order for backtracking.
    # But the `propagate` function relies on the global `colors` array.
    # The `backtrack` function should iterate through `order`.
    
    def backtrack_order(pos):
        if pos == n:
            for e in edge_set:
                vals = set(colors[v] for v in e)
                if len(vals) == 1:
                    return False
            return True
            
        v = order[pos]
        
        # Try 0
        if colors[v] == -1:
            colors[v] = 0
            if propagate():
                if backtrack_order(pos + 1):
                    return True
            colors[v] = -1
            
        # Try 1
        if colors[v] == -1:
            colors[v] = 1
            if propagate():
                if backtrack_order(pos + 1):
                    return True
            colors[v] = -1
            
        return False

    return backtrack_order(0)

def construct():
    random.seed(42) # For reproducibility, though not strictly required
    
    # Search for a non-2-colorable hypergraph
    # Start with small n and small E
    for n in range(10, 16):
        total_edges = len(list(itertools.combinations(range(n), 5)))
        if total_edges < 29:
            continue
            
        for E in range(29, min(total_edges, 100) + 1):
            # Try a few random instances
            for attempt in range(20):
                all_possible = list(itertools.combinations(range(n), 5))
                edges = random.sample(all_possible, E)
                # Ensure distinct (sample does this)
                
                if not is_2_colorable(n, edges):
                    return {'n': n, 'edges': [list(e) for e in edges]}
                    
    # Fallback: Return the complete hypergraph on 9 vertices (126 edges)
    # This is guaranteed to be non-2-colorable.
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
