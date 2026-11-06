import time
import random
from itertools import combinations

def check_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses optimized backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Build incidence structure
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    edge_c0 = [0] * len(edges)
    edge_c1 = [0] * len(edges)
    
    def backtrack():
        # 1. Unit Propagation
        forced = True
        while forced:
            forced = False
            for i in range(len(edges)):
                c0 = edge_c0[i]
                c1 = edge_c1[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            for j in vertex_to_edges[v]:
                                edge_c1[j] += 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            for j in vertex_to_edges[v]:
                                edge_c0[j] += 1
                            forced = True
                            break
                            
        # Check for contradictions
        for i in range(len(edges)):
            if edge_c0[i] == 5 or edge_c1[i] == 5:
                return False
        
        # Find unassigned vertex
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
                
        if v == -1:
            return True
        
        # Save state
        colors_copy = colors[:]
        c0_copy = edge_c0[:]
        c1_copy = edge_c1[:]
        
        # Try 0
        colors[v] = 0
        for j in vertex_to_edges[v]:
            edge_c0[j] += 1
            
        if backtrack():
            return True
            
        # Restore
        colors[:] = colors_copy
        edge_c0[:] = c0_copy
        edge_c1[:] = c1_copy
        
        # Try 1
        colors[v] = 1
        for j in vertex_to_edges[v]:
            edge_c1[j] += 1
            
        if backtrack():
            return True
            
        # Restore
        colors[:] = colors_copy
        edge_c0[:] = c0_copy
        edge_c1[:] = c1_copy
        
        return False

    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Known small non-2-colorable 5-uniform hypergraphs
    # We will try to find better ones by local search
    
    # Try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
            
        # Start with complete hypergraph
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Check if complete is non-2-colorable
        if not check_non_2_colorable(n, all_edges):
            continue
            
        # Local search: start from complete, try to remove edges
        # Use simulated annealing or greedy removal with restarts
        
        current_edges = all_edges[:]
        current_m = len(current_edges)
        
        # Greedy removal with random restarts
        for restart in range(10):  # Multiple restarts
            if time.time() > deadline - 5:
                break
            
            # Start from complete or current best
            if restart == 0:
                active = set(range(len(all_edges)))
            else:
                # Start from a random subset of size ~ best_m + 10
                target_size = max(10, best_m - 5) if best_m < float('inf') else len(all_edges) // 2
                active = set(random.sample(range(len(all_edges)), min(target_size, len(all_edges))))
            
            # Try to remove edges
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                if len(active) == 0:
                    break
                
                # Pick a random edge to try removing
                edge_idx = random.choice(list(active))
                new_active = active - {edge_idx}
                new_edges = [all_edges[i] for i in new_active]
                
                if check_non_2_colorable(n, new_edges):
                    active = new_active
                    if len(new_edges) < best_m:
                        best_m = len(new_edges)
                        best_edges = new_edges[:]
                        best_n = n
                else:
                    # Try adding a random edge back if we got stuck
                    if len(active) < len(all_edges):
                        inactive = set(range(len(all_edges))) - active
                        if inactive:
                            add_idx = random.choice(list(inactive))
                            active.add(add_idx)
                    else:
                        break
        
        # Final check for this n
        if best_edges is not None and best_n == n:
            # Verify
            if check_non_2_colorable(best_n, best_edges):
                pass
    
    # Final verification
    if best_edges is not None:
        if check_non_2_colorable(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a known small one if found, else empty
    # If no valid one found (shouldn't happen for n>=9), return empty
    return {'n': best_n if best_edges else 0, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    if res['edges']:
        assert not check_non_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
