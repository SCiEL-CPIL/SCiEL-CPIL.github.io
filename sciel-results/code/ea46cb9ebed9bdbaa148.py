import time
import itertools
import random

DEADLINE = 140.0

def _unit_propagate(edges, colors):
    """
    In-place unit propagation. Returns True if consistent, False if conflict.
    """
    changed = True
    while changed:
        changed = False
        for edge in edges:
            c0 = 0
            c1 = 0
            uncolored = []
            for v in edge:
                if colors[v] == 0:
                    c0 += 1
                elif colors[v] == 1:
                    c1 += 1
                else:
                    uncolored.append(v)
            
            if c0 == 5 or c1 == 5:
                return False
            
            if len(uncolored) == 1:
                if c0 == 4:
                    colors[uncolored[0]] = 1
                    changed = True
                elif c1 == 4:
                    colors[uncolored[0]] = 0
                    changed = True
    return True

def _is_not_2_colorable(edges, n, deadline=None):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation and degree-based variable ordering.
    """
    if deadline is None:
        deadline = 10.0
    
    start_time = time.time()
    
    # Calculate degrees
    degrees = [0] * n
    for edge in edges:
        for v in edge:
            degrees[v] += 1
    
    nodes_visited = 0
    max_nodes = 100000
    
    def bt(colors):
        nonlocal nodes_visited
        nodes_visited += 1
        
        if time.time() - start_time > deadline or nodes_visited > max_nodes:
            return None  # Timeout
        
        # Find uncolored vertex with max degree
        best_v = -1
        best_deg = -1
        for i in range(n):
            if colors[i] == -1:
                if degrees[i] > best_deg:
                    best_deg = degrees[i]
                    best_v = i
        
        if best_v == -1:
            # All colored, no conflicts found during propagation
            return True  # Found valid coloring -> IS 2-colorable
        
        for c in [0, 1]:
            new_colors = colors[:]
            new_colors[best_v] = c
            if _unit_propagate(edges, new_colors):
                result = bt(new_colors)
                if result is None:
                    return None
                if result:
                    return True
        return False
        
    initial_colors = [-1] * n
    result = bt(initial_colors)
    if result is None:
        # Timeout - assume it might be colorable (conservative)
        return False
    # If bt returns True, a coloring exists -> IS 2-colorable -> return False
    # If bt returns False, no coloring exists -> NOT 2-colorable -> return True
    return not result

def construct():
    start_time = time.time()
    
    def _time_left():
        return DEADLINE - 5 - (time.time() - start_time)
    
    best_result = None
    best_e = float('inf')
    
    # Strategy: Start with complete hypergraph on 9 vertices (126 edges),
    # then greedily remove edges while maintaining non-2-colorability.
    
    n = 9
    all_edges = list(itertools.combinations(range(n), 5))
    edges_list = [list(e) for e in all_edges]
    
    # Verify it's non-2-colorable
    if not _is_not_2_colorable(edges_list, n, deadline=5.0):
        # Shouldn't happen, but return as-is
        return {'n': n, 'edges': edges_list}
    
    best_e = len(edges_list)
    best_result = {'n': n, 'edges': edges_list}
    
    # Greedy edge removal
    rng = random.Random(42)
    remove_order = list(range(len(edges_list)))
    rng.shuffle(remove_order)
    
    max_iterations = 200
    iteration = 0
    
    while iteration < max_iterations and _time_left() > 10:
        iteration += 1
        improved = False
        for idx in remove_order:
            if _time_left() < 5:
                break
            if idx >= len(edges_list):
                continue
            
            removed_edge = edges_list.pop(idx)
            
            if _is_not_2_colorable(edges_list, n, deadline=3.0):
                # Still non-2-colorable, keep it removed
                improved = True
                if len(edges_list) < best_e:
                    best_e = len(edges_list)
                    best_result = {'n': n, 'edges': list(edges_list)}
                # Re-shuffle for next pass
                remove_order = list(range(len(edges_list)))
                rng.shuffle(remove_order)
                break  # Restart the pass with new order
            else:
                # Restore the edge
                edges_list.insert(idx, removed_edge)
        
        if not improved:
            # Try a different random order
            remove_order = list(range(len(edges_list)))
            rng.shuffle(remove_order)
            if _time_left() < 10:
                break
    
    if best_result is not None:
        return best_result
    
    # Absolute fallback
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
