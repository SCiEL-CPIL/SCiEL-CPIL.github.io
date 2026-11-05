import time
import random

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 5.0  # 5 seconds for optimization, plenty of margin
    
    # Target differences to cover: 1 to 38
    # Corresponds to cycle lengths 3 to 40
    # L = v - u + 2 => v - u = L - 2
    # If L=3, diff=1. If L=40, diff=38.
    target_diffs = set(range(1, 39))
    
    best_S = None
    best_coverage = -1
    
    # We know size 10 is the minimum for Path+Spoke topology.
    # We will try to find a set of size 10 that covers all diffs.
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs

    def is_valid(S):
        diffs = get_coverage(S)
        return diffs == target_diffs

    # Deterministic Greedy Search with Backtracking
    # This is more robust than pure random for small N
    
    def solve(size):
        # Initialize S with spread out elements to maximize initial coverage
        # Heuristic: Pick 1 and 39, then fill in
        S = [1, 39]
        candidates = list(range(2, 39))
        random.shuffle(candidates)
        S.extend(candidates[:size-2])
        
        # Local Search / Simulated Annealing style
        current_diffs = get_coverage(S)
        current_score = len(current_diffs)
        
        for _ in range(5000):
            if time.time() - start_time > timeout:
                break
                
            if current_score == len(target_diffs):
                return S
            
            # Pick a random index to change
            idx = random.randint(0, len(S)-1)
            old_val = S[idx]
            
            # Pick a new value
            new_val = random.randint(1, 39)
            
            # Ensure distinctness in S
            if new_val in S and new_val != old_val:
                continue
                
            S[idx] = new_val
            
            new_diffs = get_coverage(S)
            new_score = len(new_diffs)
            
            if new_score >= current_score:
                current_diffs = new_diffs
                current_score = new_score
            else:
                # Revert
                S[idx] = old_val
                
        return None

    # Try size 10 first (theoretical min)
    for _ in range(10): # 10 restarts
        if time.time() - start_time > timeout:
            break
        S = solve(10)
        if S and is_valid(S):
            best_S = S
            break
            
    if best_S is None:
        # Fallback to size 11 (should not happen, but safe)
        for _ in range(10):
            if time.time() - start_time > timeout:
                break
            S = solve(11)
            if S and is_valid(S):
                best_S = S
                break

    if best_S is None:
        # Fallback to seed construction (Fan) if search fails completely
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}

    # Construct graph with best_S
    # Path edges: 1-2, ..., 38-39
    path_edges = [(i, i+1) for i in range(1, N-1)]
    # Spoke edges: 0-s for s in S
    spoke_edges = [(0, s) for s in best_S]
    
    edges = path_edges + spoke_edges
    
    # Normalize edges
    normalized_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        normalized_edges.add((u, v))
    final_edges = list(normalized_edges)
    
    # Build mapping from diff to pair (u, v)
    # We need for each L in 3..40, a pair u, v in S such that v-u = L-2
    S_set = set(best_S)
    cycles = []
    
    # Pre-compute pairs for each diff to speed up cycle construction
    diff_to_pair = {}
    S_sorted = sorted(best_S)
    for i in range(len(S_sorted)):
        for j in range(i+1, len(S_sorted)):
            d = S_sorted[j] - S_sorted[i]
            if d not in diff_to_pair:
                diff_to_pair[d] = (S_sorted[i], S_sorted[j])
                
    for L in range(3, N+1):
        diff_needed = L - 2
        if diff_needed in diff_to_pair:
            u, v = diff_to_pair[diff_needed]
            # Cycle: 0 -> u -> ... -> v -> 0
            # Vertices: 0, u, u+1, ..., v
            cycle_verts = [0] + list(range(u, v+1))
            cycles.append(cycle_verts)
        else:
            # This should not happen if is_valid passed
            # Fallback: try to find any pair
            found = False
            for u in best_S:
                v = u + diff_needed
                if v in S_set:
                    cycle_verts = [0] + list(range(u, v+1))
                    cycles.append(cycle_verts)
                    found = True
                    break
            if not found:
                # Critical failure, return invalid to signal error
                raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': final_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
