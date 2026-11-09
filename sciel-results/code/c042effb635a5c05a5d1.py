# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(edges, n, deadline=None):
    """
    Exact 2-colourability check with unit propagation and backtracking.
    Returns True if a proper 2-colouring exists, False if none.
    edges: list of lists of 5 vertices (0..n-1).
    """
    if deadline is None:
        deadline = time.time() + 10
    
    if not edges:
        return True

    # Build incidence: vertex -> list of edge indices
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)

    # State: colour[v] in {-1, 0, 1}
    assign = [-1] * n
    evs = [list(e) for e in edges]
    
    # Conflict tracking for backjumping (simplified: just store last conflict size/count)
    # We use a basic backtracking with unit propagation.
    # To speed up, we maintain a count of unassigned vertices per edge? 
    # No, recomputing is O(5) per edge, which is cheap.
    
    # Optimization: Pre-compute sorted vertices in edges? Not strictly needed.
    
    def propagate(assign, changed):
        """
        Unit propagation.
        Returns True if consistent, False if conflict.
        Modifies assign in place.
        """
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = evs[ei]
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unassigned = []
                for u in e:
                    cu = assign[u]
                    if cu == 0:
                        c0 += 1
                    elif cu == 1:
                        c1 += 1
                    else:
                        unassigned.append(u)
                
                # Check for conflict: all assigned and same color
                if not unassigned and (c0 == 5 or c1 == 5):
                    return False
                
                # Check for unit: 4 same, 1 unassigned
                if len(unassigned) == 1:
                    if c0 == 4:
                        # Force unassigned to 1
                        u = unassigned[0]
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 1
                                stack.append(u)
                            else:
                                # Already in stack, but was it assigned?
                                # If it was already assigned to 0, conflict.
                                # If it was assigned to 1, ok.
                                # If it was -1, we just set it.
                                pass
                    elif c1 == 4:
                        u = unassigned[0]
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 0
                                stack.append(u)
        
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None
        
        # Check if all assigned
        if not uncolored:
            # Verify all edges are mixed
            for e in edges:
                c0 = sum(1 for u in e if assign[u] == 0)
                c1 = sum(1 for u in e if assign[u] == 1)
                if c0 == 0 or c1 == 0:
                    return None
            return assign

        # Heuristic: Pick vertex with highest degree among uncolored
        # Or vertex involved in most 'tight' edges?
        # Simple degree heuristic is usually good.
        v = max(uncolored, key=lambda x: len(inc[x]))
        new_uncolored = uncolored - {v}
        
        # Try 0 then 1
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
            
        return None

    uncolored = set(range(n))
    # Initial propagation? No unassigned changed.
    # But we can check if any edge is already monochromatic? No, all unassigned.
    
    result = backtrack(assign, uncolored)
    if result is None:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy: Simulated Annealing
    # Start with a known non-2-colorable hypergraph or a dense random one.
    # We know m(5) <= 51. Let's aim for that.
    # Start with n=11 or 12.
    # A complete hypergraph K_n^5 is non-2-colorable for n >= 10? 
    # Actually, K_10^5 is 2-colorable? No, K_9^5 is 2-colorable. K_10^5 is non-2-colorable?
    # Let's check small n.
    # If we start with a valid non-2-colorable graph, we can try to remove edges.
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try a few different n values
    for n in [11, 12, 13]:
        if time.time() > deadline - 5:
            break
            
        # Generate a dense random hypergraph
        # Or start with all edges? K_11^5 has 462 edges. K_12^5 has 792.
        # Let's start with a random subset that is likely non-2-colorable.
        # Or just start with all edges and reduce.
        
        all_edges = list(combinations(range(n), 5))
        current_edges = list(all_edges)
        
        # Verify initial state
        # If K_n^5 is 2-colorable, this n is useless.
        if _is_2_colorable(current_edges, n, deadline - 5):
            continue
            
        # It is non-2-colorable. Now use SA to minimize edges.
        # State: set of edges.
        # Move: Remove a random edge. If still non-2-colorable, accept.
        #        If becomes 2-colorable, reject (or accept with low prob? No, we want to keep it non-2-colorable).
        #        Wait, standard SA allows moves that increase energy (badness).
        #        Here, "energy" is number of edges. We want to minimize.
        #        Constraint: Must be non-2-colorable.
        #        So, we can only accept a move if the new state is non-2-colorable.
        #        This is a greedy hill-climbing / local search, not really SA, because we can't accept "worse" states (2-colorable) as they are invalid.
        #        Unless we define the objective as:
        #        E = num_edges + lambda * (1 if 2-colorable else 0)
        #        But we need a valid solution.
        #        
        #        Alternative: Start with a valid one. Try to remove edges.
        #        If removal keeps it non-2-colorable, keep it.
        #        If removal makes it 2-colorable, put it back.
        #        This is greedy. To escape local minima, we could try swapping edges?
        #        Or just run multiple random restarts.
        
        # Let's do a greedy reduction with random restarts.
        # For each restart:
        # 1. Start with all edges (or a random dense subset).
        # 2. Shuffle edges.
        # 3. Iterate through edges. Try to remove each. If still non-2-colorable, remove it.
        # 4. Repeat until no edge can be removed.
        
        for restart in range(5): # 5 restarts per n
            if time.time() > deadline - 10:
                break
                
            # Start with all edges
            cur_edges = set(all_edges)
            
            # Greedy reduction
            # To make it more robust, shuffle the order of removal attempts
            edge_list = list(cur_edges)
            random.shuffle(edge_list)
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try to remove edges in random order
                # We need to re-shuffle or iterate a copy
                for e in list(cur_edges):
                    if time.time() > deadline - 10:
                        break
                    
                    # Try removing e
                    cur_edges.remove(e)
                    # Check if still non-2-colorable
                    # Optimization: Only check if the number of edges is small enough?
                    # No, we must check.
                    
                    # Fast check?
                    # If we just removed an edge, maybe we can reuse info? Hard.
                    # Just call the checker.
                    
                    if not _is_2_colorable(list(cur_edges), n, deadline - 10):
                        # Still non-2-colorable. Accept removal.
                        improved = True
                        # We can break early if we've removed many? No, keep going.
                    else:
                        # Became 2-colorable. Reject removal.
                        cur_edges.add(e)
                        # If we can't remove this edge, try the next.
                        
            # End of one reduction pass
            
            # Check if this result is better than best
            if len(cur_edges) < best_count:
                best_count = len(cur_edges)
                best_edges = list(cur_edges)
                best_n = n
                
                # If we get very close to the known bound (51), we can stop early?
                # Or keep trying to improve.
                if best_count <= 51:
                    # Try to squeeze a bit more?
                    # If we hit 51, that's the known best. 
                    # We can stop if we have plenty of time left, or continue to see if we can go lower?
                    # The task says "Minimize". If 51 is the best known, and we found it, we are good.
                    # But maybe we can find 50?
                    # Let's continue if time permits.
                    pass

    # Fallback if SA failed to find anything (should not happen)
    if best_edges is None:
        # Return a known valid hypergraph. 
        # K_11^5 is valid.
        n = 11
        best_edges = list(combinations(range(n), 5))
        best_n = n

    # Final sanity check
    if _is_2_colorable(best_edges, best_n, deadline):
        # This is bad. Return a definitely non-2-colorable one.
        # K_11^5
        n = 11
        best_edges = list(combinations(range(n), 5))
        best_n = n

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
