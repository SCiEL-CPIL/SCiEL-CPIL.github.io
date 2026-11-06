import time
import random
from itertools import combinations

def _is_not_2_colorable_fast(n, edges):
    """
    Fast check using bitmasks and backtracking.
    edges: list of frozensets or tuples of vertices.
    """
    if not edges:
        return False
        
    m = len(edges)
    # Map edges to bitmasks of vertices? No, map vertices to bitmasks of edges?
    # Let's use standard arrays for speed in Python, but with careful optimization.
    
    # Precompute incidence
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # State
    # assignment[v] = -1 (unassigned), 0, 1
    assignment = [-1] * n
    # edge_state[i] = (count0, count1, unassigned)
    # We can pack this into integers for speed?
    # count0, count1, unassigned are small (0-5).
    # Let's use lists of ints.
    c0 = [0] * m
    c1 = [0] * m
    un = [5] * m
    
    # Symmetry breaking: fix vertex 0 to 0
    # If the graph is non-2-colorable, this branch will fail.
    # If the graph IS 2-colorable, this branch might succeed or fail.
    # If it fails, we must check vertex 0 = 1?
    # Actually, if we fix v0=0 and find no solution, it doesn't mean no solution exists (v0=1 might work).
    # So we must check both if the first one fails?
    # Or, we can just run the search without fixing v0, but that doubles the time.
    # Better: Fix v0=0. If we find a solution, return False (is colorable).
    # If we exhaust the search tree with v0=0 and find no solution, we MUST check v0=1.
    # However, for non-2-colorable graphs, both fail.
    # For colorable graphs, one succeeds.
    # So:
    # 1. Try v0=0. If solution found -> Return False (Colorable).
    # 2. If no solution with v0=0, try v0=1. If solution found -> Return False.
    # 3. If no solution with v0=1 -> Return True (Not Colorable).
    
    def propagate():
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if un[i] == 0:
                    if c0[i] == 5 or c1[i] == 5:
                        return False
                elif un[i] == 1:
                    # Find the unassigned vertex in this edge
                    for v in edges[i]:
                        if assignment[v] == -1:
                            # Must be opposite to the 4 others
                            if c0[i] == 4:
                                val = 1
                            elif c1[i] == 4:
                                val = 0
                            else:
                                # un=1 but c0!=4 and c1!=4? 
                                # c0+c1+un = 5. If un=1, c0+c1=4.
                                # So one of c0,c1 is 4.
                                return False # Should not happen if consistent
                            
                            # Assign v to val
                            assignment[v] = val
                            if val == 0:
                                c0[i] += 1
                            else:
                                c1[i] += 1
                            un[i] -= 1
                            
                            # Update other edges
                            for ei in vertex_edges[v]:
                                if ei == i: continue
                                if val == 0:
                                    c0[ei] += 1
                                else:
                                    c1[ei] += 1
                                un[ei] -= 1
                                if c0[ei] == 5 or c1[ei] == 5:
                                    return False
                            changed = True
                            break # Restart loop as state changed
                    # If no unassigned vertex found in edge i, but un[i]>0?
                    # This implies data inconsistency, should not happen.
        return True

    def backtrack():
        # Check for contradictions
        # (Handled in propagate and assignment steps)
        
        # Find unassigned vertex
        # Heuristic: Vertex in the most constrained edge (lowest unassigned count > 1)
        # Or simply first unassigned.
        # Let's try first unassigned for simplicity, but pick the one with highest degree?
        # MCV: Minimize unassigned count in incident edges?
        # Let's just iterate and pick the one with min min(un[e] for e in vertex_edges[v])
        
        best_v = -1
        best_score = 100
        for v in range(n):
            if assignment[v] == -1:
                # Score: min unassigned in incident edges
                min_un = 5
                for e in vertex_edges[v]:
                    if un[e] < min_un:
                        min_un = un[e]
                if min_un < best_score:
                    best_score = min_un
                    best_v = v
                    if best_score == 2: # Can't get better than 2 (1 is handled by prop)
                        break
        
        if best_v == -1:
            # All assigned
            # Check if any edge is monochromatic
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            return True # Valid coloring found
        
        # Branch on best_v
        # Try 0
        backup_c0 = c0[:]
        backup_c1 = c1[:]
        backup_un = un[:]
        
        assignment[best_v] = 0
        valid = True
        for e in vertex_edges[best_v]:
            c0[e] += 1
            un[e] -= 1
            if c0[e] == 5:
                valid = False
                break
        
        if valid and propagate() and backtrack():
            return True # Found coloring
        
        # Restore
        c0[:] = backup_c0
        c1[:] = backup_c1
        un[:] = backup_un
        assignment[best_v] = -1
        
        # Try 1
        backup_c0 = c0[:]
        backup_c1 = c1[:]
        backup_un = un[:]
        
        assignment[best_v] = 1
        valid = True
        for e in vertex_edges[best_v]:
            c1[e] += 1
            un[e] -= 1
            if c1[e] == 5:
                valid = False
                break
        
        if valid and propagate() and backtrack():
            return True # Found coloring
        
        # Restore
        c0[:] = backup_c0
        c1[:] = backup_c1
        un[:] = backup_un
        assignment[best_v] = -1
        
        return False # No coloring in this branch

    # Try v0 = 0
    assignment[0] = 0
    for e in vertex_edges[0]:
        c0[e] += 1
        un[e] -= 1
        if c0[e] == 5:
            # Contradiction immediately?
            # This specific branch fails.
            pass
    
    # We need to restore state for the second branch if the first fails.
    # It's easier to reset the state arrays for each top-level branch.
    
    def check_branch(val):
        # Reset state
        for i in range(n): assignment[i] = -1
        for i in range(m):
            c0[i] = 0; c1[i] = 0; un[i] = 5
            
        assignment[0] = val
        for e in vertex_edges[0]:
            if val == 0: c0[e] += 1
            else: c1[e] += 1
            un[e] -= 1
            if c0[e] == 5 or c1[e] == 5:
                return False
        
        if not propagate():
            return False
            
        return backtrack()

    if check_branch(0):
        return False
    if check_branch(1):
        return False
        
    return True

def _greedy_reduce_fast(n, edges, max_time=10.0):
    edges = list(edges)
    start_time = time.time()
    changed = True
    while changed and time.time() - start_time < max_time:
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        for i in range(len(edges)):
            if time.time() - start_time > max_time:
                break
            candidate = edges[:i] + edges[i+1:]
            if _is_not_2_colorable_fast(n, candidate):
                edges = candidate
                changed = True
                break
    return edges

def construct():
    start_time = time.time()
    deadline = 140 # Safety margin
    
    best_edges = 1000
    best_result = None
    
    # Strategy 1: Reduce K9(5)
    n = 9
    all_edges = list(combinations(range(n), 5))
    if _is_not_2_colorable_fast(n, all_edges):
        reduced = _greedy_reduce_fast(n, all_edges, max_time=30.0)
        if len(reduced) < best_edges:
            best_edges = len(reduced)
            best_result = {'n': n, 'edges': [list(e) for e in reduced]}
            
    # Strategy 2: Random dense graphs on n=10, 11, 12, 13
    # Threshold for non-2-colorable is roughly 0.2-0.3 * C(n,5)
    for n in [10, 11, 12, 13]:
        if time.time() > deadline - 20:
            break
        all_edges_n = list(combinations(range(n), 5))
        # Try a few random subsets
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            # Estimate threshold
            target_density = 0.3
            num_edges = int(len(all_edges_n) * target_density)
            # Ensure enough edges
            num_edges = max(num_edges, 50)
            
            random.seed(42 + n + _) # Deterministic for reproducibility? 
            # Actually, random shuffle
            random.shuffle(all_edges_n)
            edges_subset = all_edges_n[:num_edges]
            
            if _is_not_2_colorable_fast(n, edges_subset):
                reduced = _greedy_reduce_fast(n, edges_subset, max_time=10.0)
                if len(reduced) < best_edges:
                    best_edges = len(reduced)
                    best_result = {'n': n, 'edges': [list(e) for e in reduced]}
                
    # Fallback
    if best_result is None:
        return {'n': 9, 'edges': [list(e) for e in all_edges]}
        
    return best_result
