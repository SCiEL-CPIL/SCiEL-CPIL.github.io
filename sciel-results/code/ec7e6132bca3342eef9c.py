import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150 - 10  # 10s safety margin
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    # Convert to frozensets for faster lookup in solver? Or just lists?
    # For the solver, sets are better.
    
    def is_not_2_colorable(edges):
        # edges is a list of lists of 5 ints
        # Returns True if NOT 2-colorable
        # Backtracking with unit propagation
        
        # Optimization: Use bitmasks?
        # 9 vertices fit in an int.
        # Color 0: 0, Color 1: 1, Unassigned: -1
        # We can represent the state as a tuple of 9 ints.
        
        # Let's use a simple recursive backtracking.
        # To speed up, we can order vertices by degree or just use 0..8.
        
        # Pre-process edges for the solver
        # For each edge, store the 5 vertices.
        
        def check(edges, colors):
            # colors: list of 9, -1 is unassigned, 0/1 is assigned
            # Check for conflicts
            for e in edges:
                c0 = 0
                c1 = 0
                un = 0
                for v in e:
                    c = colors[v]
                    if c == 0: c0 += 1
                    elif c == 1: c1 += 1
                    else: un += 1
                
                if c0 == 5 or c1 == 5:
                    return False, None # Conflict
            
            # Unit Propagation
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    un_idx = []
                    for v in e:
                        c = colors[v]
                        if c == 0: c0 += 1
                        elif c == 1: c1 += 1
                        else: un_idx.append(v)
                    
                    if c0 == 4 and len(un_idx) == 1:
                        v = un_idx[0]
                        if colors[v] == -1:
                            colors[v] = 1
                            changed = True
                            # Check if this creates a conflict immediately?
                            # The loop will handle it.
                    elif c1 == 4 and len(un_idx) == 1:
                        v = un_idx[0]
                        if colors[v] == -1:
                            colors[v] = 0
                            changed = True
            
            # Check for conflicts again after propagation
            for e in edges:
                c0 = 0
                c1 = 0
                for v in e:
                    c = colors[v]
                    if c == 0: c0 += 1
                    elif c == 1: c1 += 1
                    if c0 == 5 or c1 == 5:
                        return False, None
            
            # Find next unassigned vertex
            for i in range(n):
                if colors[i] == -1:
                    # Try 0
                    colors[i] = 0
                    res, _ = check(edges, colors)
                    if res:
                        return True, colors
                    # Try 1
                    colors[i] = 1
                    res, _ = check(edges, colors)
                    if res:
                        return True, colors
                    # Backtrack
                    colors[i] = -1
                    return False, None
            
            # All assigned, no conflicts found
            return True, colors

        initial_colors = [-1] * n
        res, _ = check(edges, initial_colors)
        return not res # Return True if NOT 2-colorable (i.e. check failed to find coloring)
        # Wait, check returns True if a coloring IS found.
        # So if check returns False, it is NOT 2-colorable.
        # So is_not_2_colorable should return (not res).
        
    # Start with all edges
    current_edges = all_edges[:]
    
    # Randomized Greedy Minimization
    best_edges = current_edges[:]
    
    # We can try multiple restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        # Try to remove each edge
        # We iterate through the shuffled list
        # If removing an edge keeps it not-2-colorable, we remove it permanently.
        
        # To make it efficient, we can just loop through indices
        # But if we remove one, the list changes.
        # Better to iterate through a copy of indices or just check all pairs?
        # Greedy: 
        # For i in range(len(current_edges)):
        #   if is_not_2_colorable(current_edges without i):
        #       remove i
        
        # This is O(M * CostCheck). M=126. CostCheck is small.
        
        i = 0
        while i < len(current_edges):
            if time.time() > deadline:
                break
            
            # Try removing current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_not_2_colorable(test_edges):
                # It is still not 2-colorable, so we can remove this edge
                current_edges = test_edges
                # Do not increment i, because the next element shifted into i
            else:
                i += 1
        
        if len(current_edges) < len(best_edges):
            best_edges = current_edges[:]
            
    return {'n': n, 'edges': best_edges}
