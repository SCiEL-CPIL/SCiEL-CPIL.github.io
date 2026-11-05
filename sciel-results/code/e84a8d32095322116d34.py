import random
import time
import math

N = 40
TARGET_DIFFS = set(range(1, 39))

def construct():
    start_time = time.time()
    timeout = 85.0 # Safety margin
    
    def get_missing(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in TARGET_DIFFS:
                    diffs.add(d)
        return TARGET_DIFFS - diffs
    
    # Try to find S of size 10
    best_S = None
    
    # Simulated Annealing for size 10
    # We can run multiple restarts or one long anneal
    # Given the tight constraint, multiple restarts with strong local search might be better
    
    def solve_size(k):
        # Return S if found, else None
        # Use SA
        current_S = random.sample(range(1, 40), k)
        current_missing = get_missing(current_S)
        current_cost = len(current_missing)
        
        best_S_local = current_S.copy()
        best_cost_local = current_cost
        
        T = 1.0
        cooling = 0.99
        min_T = 1e-5
        
        while T > min_T:
            # Limit iterations per temperature to respect time
            # Or just run until time limit
            for _ in range(1000): # Adjust based on time
                if time.time() - start_time > timeout:
                    return best_S_local if best_cost_local == 0 else None
                
                # Perturb
                new_S = current_S.copy()
                idx = random.randint(0, k-1)
                old_val = new_S[idx]
                new_val = random.randint(1, 39)
                if new_val in new_S and new_val != old_val:
                    continue
                new_S[idx] = new_val
                
                new_missing = get_missing(new_S)
                new_cost = len(new_missing)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                    current_S = new_S
                    current_missing = new_missing
                    current_cost = new_cost
                    
                    if current_cost < best_cost_local:
                        best_cost_local = current_cost
                        best_S_local = current_S.copy()
                        
                        if best_cost_local == 0:
                            return best_S_local
                            
                T *= cooling
        return best_S_local if best_cost_local == 0 else None

    # Attempt 1: Size 10
    # Use a loop of restarts to cover the time budget
    restarts = 0
    while time.time() - start_time < timeout - 5.0: # Leave 5s for fallback
        restarts += 1
        S_found = solve_size(10)
        if S_found:
            best_S = S_found
            break
            
    # If size 10 not found, try size 11 (fallback)
    if best_S is None:
        # Size 11 is much easier
        # Just run a few restarts
        for _ in range(5):
            if time.time() - start_time > timeout:
                break
            S_found = solve_size(11)
            if S_found:
                best_S = S_found
                break
                
    if best_S is None:
        # Hard fallback: known valid size 11 or 12
        # The original code had a fallback.
        # We can use a simple greedy or the original fallback
        S = list(range(1, 40)) # Size 39, definitely works
        best_S = S

    # Construct graph
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found_cycle = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                # Ensure u < v for the range
                if u < v:
                     cycle_verts = [0] + list(range(u, v+1))
                else:
                     # This case shouldn't happen if we iterate u in S and v = u+diff
                     # If u > v, we swap? No, path is directed 1->2->...
                     # The cycle 0-u-...-v-0 requires path u->v.
                     # So we need u < v.
                     cycle_verts = [0] + list(range(v, u+1))
                     # Wait, if u > v, the path is v->...->u.
                     # The cycle is 0-u-...-v-0? No, 0-u is an edge.
                     # Path from u to v?
                     # The edges are undirected.
                     # The cycle is 0 -> u -> ... -> v -> 0.
                     # This requires path edges between u and v.
                     # The path is linear. So we traverse from min(u,v) to max(u,v).
                     # The cycle vertices list order doesn't matter for validity,
                     # but it must be a valid cycle.
                     # [0, u, u+1, ..., v, 0] works if u < v.
                     # If u > v, [0, u, u-1, ..., v, 0] works.
                     pass
                
                # Simplified:
                # The cycle is 0, then the path segment between u and v, then back to 0.
                # Vertices: 0, min(u,v), min(u,v)+1, ..., max(u,v), 0?
                # No, 0 is connected to u and v.
                # So the cycle is 0 - u - (path) - v - 0.
                # If u < v: 0, u, u+1, ..., v, 0.
                # If u > v: 0, u, u-1, ..., v, 0.
                
                # Let's just build it correctly.
                if u < v:
                    cycle_verts = [0] + list(range(u, v+1))
                else:
                    cycle_verts = [0] + list(range(u, v-1, -1))
                
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            # This should not happen if S is valid
            # But if it does, we have a bug.
            # For safety, if we failed to find a cycle, we return the edges and hope for the best?
            # No, we must raise or return invalid.
            # But we verified S covers all diffs.
            # So this should not happen.
            pass

    return {'edges': edges, 'cycles': cycles}
