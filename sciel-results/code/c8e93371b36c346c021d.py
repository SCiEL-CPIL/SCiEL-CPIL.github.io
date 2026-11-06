import time
import random
from itertools import combinations

def is_2_colorable_fast(edges, n):
    """
    Fast 2-colorability check using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Build vertex-to-edge index
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For each edge, store as a tuple for fast access
    edge_tuples = [tuple(e) for e in edges]
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    def propagate(v, c):
        """Assign color c to vertex v with unit propagation. Returns False on contradiction."""
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        # Process all edges containing v
        for e_idx in vertex_edges[v]:
            e = edge_tuples[e_idx]
            count_c = 0
            count_1c = 0
            unassigned = -1
            
            for u in e:
                cu = colors[u]
                if cu == c:
                    count_c += 1
                elif cu == 1 - c:
                    count_1c += 1
                else:
                    unassigned = u
            
            # Contradiction check
            if count_c == 5 or count_1c == 5:
                return False
            
            # Forced assignment
            if count_c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            elif count_1c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, c):
                        return False
        
        return True
    
    def backtrack():
        # Find unassigned vertex with highest degree
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True  # All assigned, no contradictions
        
        # Try color 0
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Use n=9, K_9^5 has 126 edges.
    # Greedy deletion is very effective here.
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify initial state is non-2-colorable
    if is_2_colorable_fast(current_edges, n):
        # This should never happen for K_9^5, but just in case
        # Fallback: Return a known small non-2-colorable hypergraph if we have one hardcoded?
        # Or just return the initial one if it was colorable? No, it must be non-colorable.
        # If K_9^5 is colorable (it's not), we'd need a different start.
        # But K_9^5 is famously non-2-colorable.
        pass

    # Greedy Deletion
    # We try to remove edges one by one.
    # We repeat passes until no edge can be removed.
    # To avoid getting stuck in a local minimum of "cannot remove any single edge",
    # we can shuffle the order each pass.
    
    while time.time() < deadline - 5.0:
        # Shuffle current edges to randomize removal order
        random.shuffle(current_edges)
        
        removed_any = False
        # We iterate through a copy of the indices or the list itself?
        # If we modify the list while iterating, indices shift.
        # Better: iterate through indices of the current list.
        
        # Since we shuffle, we can just iterate from 0 to len-1 and check.
        # If we remove an edge, the list shrinks.
        
        i = 0
        while i < len(current_edges):
            if time.time() > deadline - 5.0:
                break
            
            # Try to remove current_edges[i]
            # Create a test list without this edge
            # Optimization: If len is small, this is fast.
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if not is_2_colorable_fast(test_edges, n):
                # Removal successful
                current_edges = test_edges
                removed_any = True
                # Do not increment i, because the next element shifted into position i
                # is the one we just skipped, and we should check it now.
                # Actually, since we shuffled, the "next" element is arbitrary.
                # We just continue with the new list at index i.
            else:
                i += 1
        
        # If we didn't remove any edges in this pass, we might be at a local minimum
        # where no single edge removal works.
        # However, for K_9^5, greedy deletion usually works well.
        # If we want to be more aggressive, we could try adding random edges and removing others,
        # but that's complex. Let's see if single-pass greedy converges.
        # If removed_any is False, we break the outer loop.
        if not removed_any:
            break
    
    # Ensure we return a valid non-empty list
    if not current_edges:
        # Should not happen, but fallback to initial
        current_edges = all_edges[:]
        
    return {'n': n, 'edges': current_edges}
