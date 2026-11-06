"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: kill long-running work well before the evaluator limit.
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin under 120s

def construct():
    """
    Return a 5-uniform hypergraph on n vertices (5 <= n <= 26, <=200 edges)
    that has no proper 2-colouring, minimizing the number of edges.
    
    Strategy:
    1. Start with the complete 5-graph on 9 vertices (126 edges).
    2. Perform multiple randomized greedy deletion passes.
    3. In each pass, iterate through a shuffled list of edges. 
       Try to remove each edge if the remaining hypergraph is still non-2-colorable.
    4. Keep the result with the minimum number of edges.
    
    The non-2-colorability check uses backtracking with unit propagation.
    """
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Helper to check if a hypergraph is 2-colorable.
    # Returns True if a valid 2-coloring exists (i.e., it IS 2-colorable).
    # Returns False if no valid 2-coloring exists (i.e., it is NOT 2-colorable).
    def is_2_colorable(edges):
        if not edges:
            return True
        
        # Pre-compute vertex indices for quick access
        # adj[v] is a list of edge indices containing vertex v
        adj = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                adj[v].append(i)
        
        # State:
        # assign[v]: -1 (unassigned), 0, or 1
        assign = [-1] * n
        # edge_state[i]: [count_0, count_1]
        # If count_0 == 5 -> conflict (all 0)
        # If count_1 == 5 -> conflict (all 1)
        # Unit prop: if count_0 == 4, the unassigned vertex must be 1.
        #            if count_1 == 4, the unassigned vertex must be 0.
        edge_state = [[0, 0] for _ in range(len(edges))]
        
        def propagate():
            # Queue of vertices to process for propagation
            # We use a simple loop that continues until no changes occur
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edges):
                    c0, c1 = edge_state[i]
                    
                    # Check for immediate contradiction
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    # Unit propagation
                    if c0 == 4:
                        # All 5 must not be 0, so the unassigned one must be 1
                        # Find the unassigned vertex in this edge
                        for v in e:
                            if assign[v] == -1:
                                assign[v] = 1
                                # Update edges containing v
                                for j in adj[v]:
                                    edge_state[j][1] += 1
                                    if edge_state[j][1] == 5:
                                        # This edge now has all 1s -> conflict? 
                                        # Wait, if an edge has all 1s, it's a valid monochromatic edge?
                                        # No, the constraint is "not all 0 AND not all 1".
                                        # So c1 == 5 is a CONFLICT.
                                        return False
                                changed = True
                                break
                    elif c1 == 4:
                        # All 5 must not be 1, so the unassigned one must be 0
                        for v in e:
                            if assign[v] == -1:
                                assign[v] = 0
                                for j in adj[v]:
                                    edge_state[j][0] += 1
                                    if edge_state[j][0] == 5:
                                        return False
                                changed = True
                                break
            return True

        import sys
        sys.setrecursionlimit(5000)

        def backtrack():
            if time.time() - _START_TIME > _TIME_LIMIT - 2:
                # Timeout safety: assume colorable to avoid infinite loops in evaluation
                # Though this might lead to a false "keep" in greedy, it's better than TLE.
                return True
                
            if not propagate():
                return False
            
            # Find an unassigned vertex
            # Heuristic: pick the one with highest degree in the remaining unassigned subgraph?
            # Or just the first one? Let's use a simple heuristic: pick vertex with most unassigned neighbors in edges
            unassigned = [v for v in range(n) if assign[v] == -1]
            if not unassigned:
                # All assigned, check if any edge is monochromatic (should have been caught by propagation)
                # But re-verify just in case
                for i, e in enumerate(edges):
                    if edge_state[i][0] == 5 or edge_state[i][1] == 5:
                        return False
                return True
            
            # Select variable with highest degree (most edges)
            # Simple heuristic: just pick the first unassigned? 
            # Let's try picking the one with highest impact.
            # For speed, let's just pick the first one for now, or random if we want variety in checks?
            # Deterministic is better for reproducibility of the check itself.
            v = unassigned[0]
            
            # Try 0
            assign[v] = 0
            for j in adj[v]:
                edge_state[j][0] += 1
                if edge_state[j][0] == 5:
                    # Conflict immediately, undo and try 1
                    # Actually, just return False from this branch
                    edge_state[j][0] -= 1
                    assign[v] = -1
                    break
            else:
                # No immediate conflict in assignment
                if backtrack():
                    return True
            
            # Undo 0
            assign[v] = -1
            for j in adj[v]:
                edge_state[j][0] -= 1
            
            # Try 1
            assign[v] = 1
            for j in adj[v]:
                edge_state[j][1] += 1
                if edge_state[j][1] == 5:
                    edge_state[j][1] -= 1
                    assign[v] = -1
                    break
            else:
                if backtrack():
                    return True
            
            # Undo 1
            assign[v] = -1
            for j in adj[v]:
                edge_state[j][1] -= 1
            
            return False
        
        return backtrack()

    best_edges = all_edges
    best_count = len(all_edges)
    
    # Multi-start greedy deletion
    # We run several iterations with different random seeds for shuffling
    for _ in range(100):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        current_edges = list(all_edges)
        
        # Shuffle edges to break symmetry and escape local minima of a fixed order
        random.shuffle(current_edges)
        
        # Greedy deletion
        i = 0
        while i < len(current_edges):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
                
            # Try to remove current_edges[i]
            candidate = current_edges[:i] + current_edges[i+1:]
            
            # Check if candidate is still non-2-colorable
            # is_2_colorable returns True if it CAN be colored (i.e., has Property B)
            # We want to keep it if it is NOT 2-colorable (is_2_colorable returns False)
            if not is_2_colorable(candidate):
                # We can remove this edge
                # Remove it from the list
                # current_edges.pop(i) would shift indices, so we just don't increment i
                current_edges.pop(i)
                # Do not increment i, check the next edge at the same index (which is now the old i+1)
            else:
                # Cannot remove this edge
                i += 1
        
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
        # If we get very close to the theoretical lower bound (29), we might stop early?
        # But 51 is the known best, so let's keep searching until time runs out.
        
    # Final verification
    assert 5 <= n <= 26
    assert len(best_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in best_edges)) == len(best_edges)
    for e in best_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)
    
    # Verify final result is NOT 2-colorable
    # Note: is_2_colorable might be slow if the graph is complex, but we've been using it
    # in the loop, so it should be manageable.
    if is_2_colorable(best_edges):
        # Fallback to the full complete graph if something went wrong
        return {'n': n, 'edges': all_edges}
        
    return {'n': n, 'edges': best_edges}

# EVOLVE-BLOCK-END
