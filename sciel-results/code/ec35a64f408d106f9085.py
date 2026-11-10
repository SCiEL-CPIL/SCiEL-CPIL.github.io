import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Checks if the 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses brute force for n <= 16, backtracking otherwise.
    """
    if not edges:
        return True
    
    if n <= 16:
        # Brute force all 2^n colorings
        # Represent coloring by a bitmask
        for mask in range(1 << n):
            is_valid = True
            for e in edges:
                c0 = 0
                c1 = 0
                for v in e:
                    if (mask >> v) & 1:
                        c1 += 1
                    else:
                        c0 += 1
                if c0 == 5 or c1 == 5:
                    is_valid = False
                    break
            if is_valid:
                return True
        return False
    else:
        # Fallback to backtracking for larger n
        # (This part is kept from original but less likely to be used if we focus on small n)
        colors = [-1] * n
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for edge in edges:
                    c0 = 0
                    c1 = 0
                    unassigned = []
                    for v in edge:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unassigned.append(v)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    if c0 == 4 and len(unassigned) == 1:
                        if colors[unassigned[0]] == -1:
                            colors[unassigned[0]] = 1
                            changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        if colors[unassigned[0]] == -1:
                            colors[unassigned[0]] = 0
                            changed = True
            return True

        def backtrack(idx):
            if not propagate():
                return False
            
            next_v = -1
            for v in range(n):
                if colors[v] == -1:
                    next_v = v
                    break
            
            if next_v == -1:
                return True
            
            colors[next_v] = 0
            if backtrack(idx + 1):
                return True
            
            colors[next_v] = 1
            if backtrack(idx + 1):
                return True
            
            colors[next_v] = -1
            return False

        return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 0
    best_edges = []
    best_len = float('inf')
    
    def minimize(n, edges, time_limit):
        current_edges = list(edges)
        random.shuffle(current_edges)
        local_start = time.time()
        
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            e = current_edges[i]
            test_edges = [x for x in current_edges if x != e]
            
            if not _is_2_colorable(n, test_edges):
                current_edges = test_edges
                # Do not increment i
            else:
                i += 1
                
        return current_edges

    # Try minimizing K_n^5 for small n
    # K_9^5 is 126 edges. K_10^5 is 252 edges.
    # We only care about n where 2-colorability is hard, i.e., non-2-colorable.
    # K_9^5 is non-2-colorable.
    
    for n in range(9, 13): # 9, 10, 11, 12
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        
        # Check if K_n^5 is non-2-colorable
        if not _is_2_colorable(n, all_edges):
            # It is non-2-colorable. Minimize it.
            # Give it a generous time budget, but ensure we finish before deadline
            t_limit = min(time.time() + 20.0, deadline - 10.0)
            minimized = minimize(n, all_edges, t_limit)
            
            if len(minimized) < best_len:
                best_edges = minimized
                best_n = n
                best_len = len(minimized)
        
        # Optional: Try random subsets if time permits
        # But since brute force is fast for n<=12, we can do a few random restarts
        if n <= 12:
             for _ in range(3):
                 if time.time() > deadline:
                     break
                 random.shuffle(all_edges)
                 # Pick a subset of size ~ best_len + 10
                 k = min(len(all_edges), best_len + 20 if best_len < float('inf') else 80)
                 candidate_edges = all_edges[:k]
                 if not _is_2_colorable(n, candidate_edges):
                     t_limit = min(time.time() + 5.0, deadline - 10.0)
                     minimized = minimize(n, candidate_edges, t_limit)
                     if len(minimized) < best_len:
                         best_edges = minimized
                         best_n = n
                         best_len = len(minimized)

    if best_len == float('inf'):
        # Fallback: Return K_9^5 (known non-2-colorable)
        best_n = 9
        best_edges = list(combinations(range(9), 5))

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
