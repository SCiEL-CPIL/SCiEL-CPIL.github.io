import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Pre-compute edge indices for faster access
    # edges is list of lists of vertex indices
    
    # State: color of each vertex: -1 (uncolored), 0, 1
    colors = [-1] * n
    
    # For unit propagation, we need to know which edges are affected by a vertex
    # vertex_to_edges: dict mapping vertex -> list of edge indices
    vertex_to_edges = {v: [] for v in range(n)}
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_to_edges[v].append(i)
            
    # Edge state: count of 0s, count of 1s, list of uncolored vertices
    # But it's easier to just check the edge when needed.
    
    # To speed up, we can maintain for each edge:
    # count_0, count_1, uncolored_vertices (list)
    
    edge_state = []
    for edge in edges:
        edge_state.append({
            'count_0': 0,
            'count_1': 0,
            'uncolored': set(edge)
        })
        
    def propagate(changed_vertex=None, new_color=None):
        """
        Propagate constraints. Returns False if contradiction found.
        """
        # If a specific vertex was colored, start from its edges
        # Otherwise, we might need to check all? 
        # In backtracking, we only change one vertex at a time.
        
        if changed_vertex is None:
            # Initial check or full propagation? 
            # We call this after assigning a color.
            return True
            
        queue = list(vertex_to_edges[changed_vertex])
        visited = set(queue)
        
        while queue:
            edge_idx = queue.pop(0)
            if edge_idx in visited:
                continue
            visited.add(edge_idx)
            
            es = edge_state[edge_idx]
            edge = edges[edge_idx]
            
            # Update state based on current colors? 
            # It's better to maintain state incrementally.
            # But for simplicity and correctness in a small solver, 
            # let's recalculate the state of the edge from scratch.
            # n is small, edges are small.
            
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
                    
            # Update edge_state
            es['count_0'] = c0
            es['count_1'] = c1
            es['uncolored'] = uncolored # list for iteration
            
            # Check for contradiction
            if c0 == 5:
                return False
            if c1 == 5:
                return False
                
            # Unit propagation
            if c0 == 4:
                # Must be 1
                for v in uncolored:
                    if colors[v] == -1:
                        colors[v] = 1
                        # Recurse or add to queue?
                        # Since we are in a loop, we can just mark it and let it propagate?
                        # It's safer to add the vertex to a "changed" list and process.
                        # But standard unit propagation works by queueing edges.
                        # Here, coloring v affects all edges containing v.
                        for e_idx in vertex_to_edges[v]:
                            if e_idx not in visited:
                                queue.append(e_idx)
                                visited.add(e_idx)
                    elif colors[v] == 0:
                        return False # Conflict: v must be 1 but is 0
            elif c1 == 4:
                # Must be 0
                for v in uncolored:
                    if colors[v] == -1:
                        colors[v] = 0
                        for e_idx in vertex_to_edges[v]:
                            if e_idx not in visited:
                                queue.append(e_idx)
                                visited.add(e_idx)
                    elif colors[v] == 1:
                        return False # Conflict: v must be 0 but is 1
                        
        return True

    def backtrack():
        # Find uncolored vertex with fewest uncolored neighbors? 
        # Or just first uncolored.
        # Heuristic: Most constrained vertex.
        # Let's pick the uncolored vertex that appears in the most "tight" edges?
        # Simple: first uncolored vertex.
        
        # Optimization: If all colored, check if valid (should be if no contradiction)
        # Actually, if we reach here without contradiction, is it a solution?
        # Yes, if all vertices are colored.
        
        uncolored_vertices = [v for v in range(n) if colors[v] == -1]
        if not uncolored_vertices:
            return True
            
        # Pick a vertex
        # Heuristic: Choose vertex that maximizes propagation?
        # Let's just pick the first one for now, or random.
        # Random might be better for exploration? No, we want to find *a* coloring.
        # If it exists, we want to find it fast.
        # If it doesn't exist, we want to prove it fast.
        # For "is it colorable", if we find one, we stop.
        # If we fail, we backtrack.
        
        v = uncolored_vertices[0]
        
        # Try color 0
        colors[v] = 0
        if propagate(v, 0):
            if backtrack():
                return True
        colors[v] = -1
        
        # Try color 1
        colors[v] = 1
        if propagate(v, 1):
            if backtrack():
                return True
        colors[v] = -1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_edges_count = float('inf')
    best_result = None
    
    # Try different n
    for n in range(10, 15):
        if time.time() > deadline:
            break
            
        # Number of edges to try
        # We want to find a non-2-colorable one.
        # Random hypergraphs become non-2-colorable when edge density is high enough.
        # For 5-uniform, threshold is around p ~ 0.5?
        # Let's start with a moderately dense graph and peel.
        
        for trial in range(10): # 10 restarts per n
            if time.time() > deadline:
                break
                
            # Generate random hypergraph
            # Start with all edges? No, too many.
            # Start with a random subset of size M.
            # If M is too small, it's likely colorable.
            # If M is too large, it's non-colorable but we have to peel many.
            # Let's try M = 50, 60, 70.
            
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Try different sizes
            for m in [40, 50, 60, 80]:
                if time.time() > deadline:
                    break
                    
                edges = all_edges[:m]
                
                # Check if non-2-colorable
                # We want NOT colorable.
                # _is_2_colorable returns True if colorable.
                # So we want _is_2_colorable to return False.
                
                if _is_2_colorable(n, edges):
                    continue # It is colorable, skip
                
                # It is non-2-colorable. Now minimize.
                # Peel edges.
                current_edges = edges[:]
                current_count = len(current_edges)
                
                # Reorder edges to try removing them?
                # Random order is fine.
                random.shuffle(current_edges)
                
                # To make peeling efficient, we should try to remove edges that are "redundant".
                # An edge is redundant if removing it makes the graph colorable? No, 
                # if removing it keeps it non-colorable, it's redundant for the property.
                
                # We iterate through edges and try to remove each one.
                # If removing it makes the graph colorable, we put it back.
                # If it remains non-colorable, we keep it removed.
                
                i = 0
                while i < len(current_edges):
                    if time.time() > deadline:
                        break
                        
                    edge_to_try = current_edges[i]
                    # Temporarily remove
                    temp_edges = current_edges[:i] + current_edges[i+1:]
                    
                    # Check if temp_edges is non-2-colorable
                    # i.e., _is_2_colorable(n, temp_edges) is False
                    if not _is_2_colorable(n, temp_edges):
                        # It is still non-2-colorable. Keep it removed.
                        current_edges = temp_edges
                        # Do not increment i, because the next edge shifted into position i
                    else:
                        # It became colorable. We must keep the edge.
                        i += 1
                        
                # Update best
                if len(current_edges) < best_edges_count:
                    best_edges_count = len(current_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                    
    if best_result is None:
        # Fallback: Complete graph on 9 vertices
        edges = [list(e) for e in combinations(range(9), 5)]
        return {'n': 9, 'edges': edges}
        
    return best_result
