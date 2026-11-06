import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable using bitmask backtracking.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    num_edges = len(edge_masks)
    
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State: for each edge, count of 0s and 1s assigned
    # Use arrays for speed
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    def check_conflict():
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return True
        return False
    
    def propagate():
        """Unit propagation: if an edge has 4 of one color, force the 5th."""
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    # Need one more vertex to be color 1
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                elif c1[i] == 4 and c0[i] == 0:
                    # Need one more vertex to be color 0
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
        return True
    
    def backtrack(start_v):
        # Find next unassigned vertex
        v = -1
        for i in range(start_v, n):
            if color[i] == -1:
                v = i
                break
        if v == -1:
            # All assigned, check no conflicts
            return not check_conflict()
        
        # Try color 0
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 0
        ok = True
        for e_idx in vertex_edges[v]:
            c0[e_idx] += 1
            if c0[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack(v + 1):
                return True
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        # Try color 1
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 1
        ok = True
        for e_idx in vertex_edges[v]:
            c1[e_idx] += 1
            if c1[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack(v + 1):
                return True
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        return False
    
    return backtrack(0)


def is_2_colorable_fast(n, edges, deadline=None):
    """
    Fast checker. For small n, use brute force. For larger, use backtracking.
    """
    if not edges:
        return True
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    if n <= 12:
        # Brute force: try all 2^n colorings
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        return is_2_colorable(n, edges)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Strategy: For each n, start with complete 5-uniform hypergraph,
    # then greedily remove edges while maintaining non-2-colorability.
    
    # First, check n=9 complete (126 edges) as baseline
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    if not is_2_colorable_fast(n, all_edges):
        best_edges = all_edges[:]
        best_m = len(all_edges)
    
    # Try different n values with greedy edge removal
    for n in range(9, 17):
        if time.time() > deadline - 10:
            break
        
        all_possible = list(combinations(range(n), 5))
        num_total = len(all_possible)
        
        # Only try if complete graph is non-2-colorable (it should be for n >= 9)
        complete_edges = [list(e) for e in all_possible]
        if is_2_colorable_fast(n, complete_edges):
            continue  # Complete is 2-colorable, skip this n
        
        # Greedy removal: try to remove edges one by one
        current_edges = complete_edges[:]
        indices = list(range(len(current_edges)))
        
        # Multiple passes with different orderings
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            random.shuffle(indices)
            removed_any = True
            while removed_any and time.time() < deadline - 15:
                removed_any = False
                for i in indices:
                    if time.time() > deadline - 15:
                        break
                    if i >= len(current_edges):
                        continue
                    # Try removing edge i
                    new_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        indices = list(range(len(current_edges)))
                        random.shuffle(indices)
                        removed_any = True
                        break
            
            if len(current_edges) < best_m:
                best_m = len(current_edges)
                best_edges = current_edges[:]
                best_n = n
                if best_m <= 30:
                    break
        
        if best_m <= 30:
            break
    
    # If we found something good, also try to improve with more aggressive search
    if best_edges is not None and time.time() < deadline - 20:
        # Try to further reduce by trying different n values with the best m as target
        for n_try in range(9, best_n + 2):
            if time.time() > deadline - 20:
                break
            all_possible = list(combinations(range(n_try), 5))
            # Try random subsets of size best_m - 1, best_m - 2, etc.
            for m_target in range(best_m - 1, max(10, best_m - 20), -1):
                if time.time() > deadline - 20:
                    break
                if m_target >= len(all_possible):
                    continue
                for attempt in range(20):
                    if time.time() > deadline - 20:
                        break
                    random.shuffle(all_possible)
                    candidate = [list(e) for e in all_possible[:m_target]]
                    if not is_2_colorable_fast(n_try, candidate):
                        best_m = m_target
                        best_edges = candidate[:]
                        best_n = n_try
                        break
    
    if best_edges is None:
        # Fallback: complete 5-uniform on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}


if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
