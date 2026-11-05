import random
import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 85 seconds, safety margin before 90s kill
    
    # Target differences to cover: 1 to 38
    target_diffs = set(range(1, 39))
    num_targets = len(target_diffs)
    
    best_S = None
    best_size = 99
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return len(diffs), diffs
    
    def is_valid(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs == target_diffs
    
    # Try sizes from smallest to largest
    for size in [9, 10, 11, 12]:
        if time.time() - start_time > timeout - 5:
            break
        
        # Multiple restarts
        num_restarts = 50 if size == 9 else 20 if size == 10 else 10
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout - 2:
                break
            
            # Initialize S: spread out with some randomness
            S = []
            # Pick endpoints
            S.append(1)
            S.append(39)
            # Fill rest
            candidates = list(range(2, 39))
            random.shuffle(candidates)
            S.extend(candidates[:size-2])
            random.shuffle(S)
            
            current_score, _ = get_coverage(S)
            
            # Simulated annealing
            T = 10.0
            cooling = 0.999
            for iter in range(20000):
                if time.time() - start_time > timeout - 2:
                    break
                
                if current_score == num_targets:
                    best_S = S
                    break
                
                # Pick a random index to change
                idx = random.randint(0, len(S)-1)
                old_val = S[idx]
                
                # Pick a new value
                new_val = random.randint(1, 39)
                
                # Ensure distinctness in S
                if new_val in S and new_val != old_val:
                    continue
                    
                S[idx] = new_val
                
                new_score, _ = get_coverage(S)
                
                if new_score >= current_score:
                    current_score = new_score
                else:
                    # Accept with probability exp(-delta/T)
                    delta = current_score - new_score
                    if random.random() < (2.718281828 ** (-delta / T)):
                        current_score = new_score
                    else:
                        # Revert
                        S[idx] = old_val
                
                T *= cooling
            
            if current_score == num_targets:
                best_S = S
                break
        
        if best_S is not None:
            break
    
    if best_S is None:
        # Fallback to seed construction
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Construct graph with best_S
    # Path edges: 1-2, ..., 38-39
    path_edges = [(i, i+1) for i in range(1, N-1)]
    # Spoke edges: 0-s for s in best_S
    spoke_edges = [(0, s) for s in best_S]
    
    edges = path_edges + spoke_edges
    
    # Build mapping from diff to pair (u, v)
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        # Find u, v in best_S such that v - u == diff_needed
        found = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                # Cycle: 0 -> u -> ... -> v -> 0
                # Vertices: 0, u, u+1, ..., v
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found = True
                break
        if not found:
            raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': edges, 'cycles': cycles}
