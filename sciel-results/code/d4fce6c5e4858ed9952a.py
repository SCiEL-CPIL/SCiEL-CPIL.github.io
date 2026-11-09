import random
import time
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Pre-process edges for faster access
    # Map vertex to list of edges containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_to_edges[v].append(i)
            
    # State:
    # color: -1 unassigned, 0 or 1
    color = [-1] * n
    # edge_status: 
    # 0: unprocessed
    # 1: has at least one 0 and at least one 1 (satisfied)
    # 2: all assigned 0 (conflict if size > 1, but here we just track counts)
    # Actually, for 2-coloring, an edge is violated if all vertices have the same color.
    # We need to ensure NO edge is monochromatic.
    # So if an edge has 4 vertices of color 0, the 5th MUST be 1.
    # If an edge has 5 vertices of color 0, it's a conflict.
    
    # counts[edge_idx][color]
    counts = [[0, 0] for _ in range(len(edges))]
    # assigned_count[edge_idx]
    assigned = [0] * len(edges)
    
    # Queue for propagation
    queue = []
    
    def propagate():
        while queue:
            v = queue.pop()
            c = color[v]
            # Process edges containing v
            for e_idx in vertex_to_edges[v]:
                # If this edge already has both colors, it's satisfied and no longer needs propagation
                if counts[e_idx][0] > 0 and counts[e_idx][1] > 0:
                    continue
                # If we already processed this vertex for this edge? 
                # We need to track if we already updated counts for this v in this e_idx.
                # Simple approach: just update counts. But we might double count if v is processed multiple times?
                # No, v is assigned once. But we might push v multiple times?
                # We should only push v when it changes from -1 to 0/1.
                
                # Check if we already counted v for this edge
                # This simple implementation doesn't track "counted" status per vertex-edge pair.
                # This is a bug risk. 
                # Better: Only push vertices when they are assigned. 
                # And in propagate, we assume the vertex is newly assigned.
                # But what if an edge had 3 unassigned, and we assign one, then another?
                # The counts are cumulative.
                # The issue is: when v is assigned, we update counts for all its edges.
                # This happens exactly once per vertex assignment.
                
                counts[e_idx][c] += 1
                assigned[e_idx] += 1
                
                # Check for conflict: all 5 vertices same color
                if assigned[e_idx] == 5 and counts[e_idx][1-c] == 0:
                    return False # Conflict
                
                # Check for unit propagation: 4 vertices same color
                if counts[e_idx][c] == 4 and counts[e_idx][1-c] == 0:
                    # The remaining unassigned vertex must be 1-c
                    # Find the unassigned vertex in this edge
                    for u in edges[e_idx]:
                        if color[u] == -1:
                            color[u] = 1 - c
                            queue.append(u)
                            break
                            
        return True

    # Initial assignment: try to assign vertex 0 to 0
    # To be symmetric, we can just pick an unassigned vertex.
    # Heuristic: pick vertex with highest degree or most constrained edges.
    
    def select_variable():
        # Pick unassigned vertex with most assigned neighbors in edges
        best_v = -1
        best_score = -1
        for v in range(n):
            if color[v] == -1:
                # Score: number of assigned vertices in its edges
                score = 0
                for e_idx in vertex_to_edges[v]:
                    score += assigned[e_idx]
                if score > best_score:
                    best_score = score
                    best_v = v
        return best_v

    def backtrack():
        v = select_variable()
        if v == -1:
            return True # All assigned, no conflict found
        
        for c in [0, 1]:
            # Save state
            saved_color = color[v]
            saved_counts = [list(c_) for c_ in counts]
            saved_assigned = assigned[:]
            
            color[v] = c
            queue.append(v)
            
            if propagate():
                if backtrack():
                    return True
            
            # Restore state
            color[v] = saved_color
            counts = [list(c_) for c_ in saved_counts]
            assigned = saved_assigned[:]
            # Note: queue is local to propagate, so no need to restore it explicitly if we recreate it or handle carefully.
            # In this structure, queue is modified in place. We should probably pass it or reset.
            # For simplicity and correctness in a tight loop, let's restructure propagate to be cleaner.
            # Actually, the current restore logic is flawed because `queue` is a list in the outer scope and gets popped.
            # We need to restore the queue or use a different approach.
            
            # Let's rewrite the solver to be more robust.
            return False

    # The above backtracking is tricky to restore correctly with the queue.
    # Let's use a simpler, slightly slower but correct backtracking for the checker.
    
    def _check(color_state):
        # Check for conflicts
        for e_idx in range(len(edges)):
            if assigned[e_idx] == 5:
                if counts[e_idx][0] == 5 or counts[e_idx][1] == 5:
                    return False
        # Find unassigned vertex
        unassigned = [i for i in range(n) if color_state[i] == -1]
        if not unassigned:
            return True
        
        # Pick a vertex
        # Heuristic: vertex in an edge with 4 assigned of same color
        v = -1
        for e_idx in range(len(edges)):
            if assigned[e_idx] == 4:
                if counts[e_idx][0] == 4:
                    for u in edges[e_idx]:
                        if color_state[u] == -1:
                            v = u
                            break
                elif counts[e_idx][1] == 4:
                    for u in edges[e_idx]:
                        if color_state[u] == -1:
                            v = u
                            break
                if v != -1: break
        
        if v == -1:
             # Pick any unassigned
             v = unassigned[0]
             
        # Try 0
        color_state[v] = 0
        # Update counts for edges containing v
        # We need to track which edges we updated to undo them
        updated_edges = []
        for e_idx in vertex_to_edges[v]:
            counts[e_idx][0] += 1
            assigned[e_idx] += 1
            updated_edges.append(e_idx)
            
        if _check(color_state):
            return True
            
        # Undo
        for e_idx in updated_edges:
            counts[e_idx][0] -= 1
            assigned[e_idx] -= 1
        color_state[v] = -1
        
        # Try 1
        color_state[v] = 1
        updated_edges = []
        for e_idx in vertex_to_edges[v]:
            counts[e_idx][1] += 1
            assigned[e_idx] += 1
            updated_edges.append(e_idx)
            
        if _check(color_state):
            return True
            
        # Undo
        for e_idx in updated_edges:
            counts[e_idx][1] -= 1
            assigned[e_idx] -= 1
        color_state[v] = -1
        
        return False

    # Initialize
    initial_color = [-1] * n
    return _check(initial_color)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find a non-2-colorable hypergraph with 51 edges
    # Use greedy removal from a denser graph
    
    n_vertices = 20
    target_edges = 51
    
    best_edges = None
    best_count = float('inf')
    
    # Generate a starting hypergraph
    # We need a non-2-colorable one. 
    # Random sparse graphs are likely 2-colorable. 
    # We start with a dense one and remove edges.
    
    # Try a few random starting points
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Start with ~80 edges
        num_edges = 80
        possible_edges = list(combinations(range(n_vertices), 5))
        random.shuffle(possible_edges)
        current_edges = [list(e) for e in possible_edges[:num_edges]]
        
        if _is_2_colorable(n_vertices, current_edges):
            continue
            
        # We have a non-2-colorable hypergraph. Now minimize edges.
        # Greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle edges to randomize removal order
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if i >= len(current_edges):
                    continue
                    
                # Try removing edge i
                edge_to_remove = current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                if not _is_2_colorable(n_vertices, new_edges):
                    current_edges = new_edges
                    changed = True
                    if len(current_edges) < best_count:
                        best_count = len(current_edges)
                        best_edges = current_edges[:]
                        
                    # If we reached target, we can stop early if we are confident, 
                    # but we should keep trying to remove more if possible? 
                    # No, 51 is the minimum. If we hit 51, we can stop.
                    if best_count <= target_edges:
                        break
                        
            if best_count <= target_edges:
                break
                
    if best_edges:
        return {'n': n_vertices, 'edges': best_edges}
    else:
        # Fallback: Return the seed program's result if we failed to find a better one
        # This ensures we always return a valid result
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
