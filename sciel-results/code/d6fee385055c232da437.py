import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Returns False if no 2-coloring exists (non-2-colorable).
    """
    if not edges:
        return True
        
    m = len(edges)
    # Map vertex to list of edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # State
    colors = [-1] * n
    # edge_counts[e] = [count_0, count_1, count_unknown]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        """
        Process queue of (vertex, color) assignments.
        Returns False if contradiction, True otherwise.
        Modifies colors and edge_counts.
        """
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            # Assign v to c
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                # Update counts
                edge_counts[e_idx][2] -= 1 # unknown
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                # Check contradiction: monochromatic
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation
                if cU == 1:
                    # One unknown left
                    # Find it
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        # Must be 1
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        # Must be 0
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Heuristic: Pick unassigned vertex with highest "pressure"
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                if cU == 1:
                    # This vertex is the last unknown in this edge.
                    if c0 == 4 or c1 == 4:
                        score += 10 # Strong constraint
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True # All assigned
            
        v = best_v
        
        # Try 0 then 1
        for c in [0, 1]:
            # Save state
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            # Tentative assignment
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            # Restore state
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_result = None
    best_edge_count = 200  # Maximum allowed
    
    # Try different n values
    for n in [11, 12, 13, 14]:
        if time.time() > deadline - 5.0:
            break
        
        all_edges = list(combinations(range(n), 5))
        num_all_edges = len(all_edges)
        
        # Strategy: Start with a random dense set, ensure non-2-colorable, then minimize
        # We know m(5) <= 51. Let's aim for that range.
        # If we start too sparse, we might be 2-colorable.
        # If we start too dense, we have more to remove.
        
        # Try a few different starting densities
        for start_m in [40, 50, 60, 80]:
            if time.time() > deadline - 5.0:
                break
            
            # Multiple trials
            for trial in range(10):
                if time.time() > deadline - 5.0:
                    break
                
                random.shuffle(all_edges)
                current_edges = all_edges[:start_m]
                
                # Ensure it's non-2-colorable
                # If it is 2-colorable, add more edges until it isn't
                if is_2_colorable(n, [list(e) for e in current_edges]):
                    # Add edges one by one
                    for e in all_edges[start_m:]:
                        if time.time() > deadline - 5.0:
                            break
                        current_edges.append(e)
                        if not is_2_colorable(n, [list(e2) for e2 in current_edges]):
                            break
                    if time.time() > deadline - 5.0:
                        break
                    if is_2_colorable(n, [list(e2) for e2 in current_edges]):
                        continue # Still 2-colorable, skip this trial
                
                # Now current_edges is non-2-colorable.
                # Try to minimize by removing edges
                improved = True
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    # Shuffle to avoid order bias
                    indices = list(range(len(current_edges)))
                    random.shuffle(indices)
                    
                    for i in indices:
                        if time.time() > deadline - 5.0:
                            break
                        
                        # Try removing edge i
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        
                        # Check if still non-2-colorable
                        if not is_2_colorable(n, [list(e) for e in test_edges]):
                            # Accept the removal
                            current_edges = test_edges
                            improved = True
                            break  # Restart from beginning after removal
                
                # Update best
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                    
                if best_edge_count <= 30: # Very good, stop early
                    break
                    
        if best_edge_count <= 30:
            break

    # Fallback: n=9, all edges (126)
    if best_result is None:
        n_fb = 9
        all_edges_fb = list(combinations(range(n_fb), 5))
        return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
        
    return best_result
