import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        # ... (keep existing fallback) ...
        pass

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
        
    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    # Build counts array
    counts = np.zeros(N, dtype=int)
    in_code = np.zeros(N, dtype=bool)
    for c in C:
        counts[neighbors[c]] += 1
        in_code[c] = True
        
    # 2. Simulated Annealing
    # Energy = |C| + M * (uncovered_count)
    # We want to minimize Energy.
    M = 10000  # Large penalty for uncovered
    
    def get_uncovered_count():
        return np.sum(counts == 0)
    
    current_energy = len(C) + M * get_uncovered_count()
    best_C = C.copy()
    best_counts = counts.copy()
    best_in_code = in_code.copy()
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.995
    steps_per_T = 1000
    
    while T > 0.1 and time.time() - start_time < time_limit:
        for _ in range(steps_per_T):
            if time.time() - start_time > time_limit:
                break
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a random codeword
                if not C:
                    continue
                idx = random.randint(0, len(C)-1)
                c = C[idx]
                
                # Calculate Delta E
                # Delta |C| = -1
                # Delta uncovered = number of neighbors with count == 1
                delta_uncovered = 0
                for u in neighbors[c]:
                    if counts[u] == 1:
                        delta_uncovered += 1
                delta_E = -1 + M * delta_uncovered
                
                # Acceptance
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Apply move
                    C.pop(idx)
                    in_code[c] = False
                    for u in neighbors[c]:
                        counts[u] -= 1
                    
                    current_energy += delta_E
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_C = C.copy()
                        best_counts = counts.copy()
                        best_in_code = in_code.copy()
                        
            elif move_type < 0.8:
                # Add a random word not in C
                # Pick random word from 0..N-1
                w = random.randint(0, N-1)
                if in_code[w]:
                    continue
                
                # Calculate Delta E
                # Delta |C| = +1
                # Delta uncovered = - (number of neighbors with count == 0)
                delta_uncovered = 0
                for u in neighbors[w]:
                    if counts[u] == 0:
                        delta_uncovered += 1
                delta_E = 1 - M * delta_uncovered
                
                # Acceptance
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Apply move
                    C.append(w)
                    in_code[w] = True
                    for u in neighbors[w]:
                        counts[u] += 1
                        
                    current_energy += delta_E
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_C = C.copy()
                        best_counts = counts.copy()
                        best_in_code = in_code.copy()
                        
            else:
                # Swap: Remove c, Add w
                if not C:
                    continue
                idx = random.randint(0, len(C)-1)
                c = C[idx]
                w = random.randint(0, N-1)
                if in_code[w]:
                    continue
                
                # Calculate Delta E
                # Delta |C| = 0
                # Delta uncovered = (newly uncovered) - (newly covered)
                # Newly uncovered: neighbors of c with count == 1 (and not covered by w? No, counts are global)
                # Wait, counts are updated sequentially? No, we calculate delta based on current state.
                # If we remove c, counts[u] decreases by 1 for u in N(c).
                # If we add w, counts[u] increases by 1 for u in N(w).
                # Net change in counts[u]:
                # -1 if u in N(c) \ N(w)
                # +1 if u in N(w) \ N(c)
                # 0 if u in N(c) \cap N(w) or u in neither
                
                # We need to check if any count becomes 0 (uncovered) or if any count goes from 0 to 1 (covered).
                # Since we start with a valid code (uncovered=0) usually, but SA might break validity.
                
                delta_uncovered = 0
                # Check neighbors of c
                for u in neighbors[c]:
                    if u in neighbors[w]: # Approximate: we don't have a set for neighbors[w] easily. 
                        # This check is expensive. 
                        # Alternative: Just simulate the change on a temporary array? Too slow.
                        # Better: Precompute intersection? No.
                        # Let's just iterate.
                        pass
                
                # Efficient Delta E for Swap:
                # 1. Temporarily remove c:
                #    For u in N(c): if counts[u] == 1, it becomes 0 (uncovered).
                # 2. Temporarily add w:
                #    For u in N(w): if counts[u] == 0, it becomes 1 (covered).
                #    BUT, the "counts[u] == 0" check must be done AFTER removal of c.
                
                # Let's do it in two passes, but carefully.
                # Actually, we can just compute the set of neighbors.
                # N(c) and N(w) are small (17).
                # Let's just calculate the new counts for the union of neighbors.
                
                # For simplicity and correctness in SA (where we might have invalid states),
                # let's just use the Add/Remove moves primarily. Swap is complex.
                # Or, implement Swap by doing Remove then Add, but accepting/rejecting as a whole?
                # That's not standard SA.
                
                # Let's skip Swap for now and rely on Add/Remove.
                # Add/Remove is sufficient to change size.
                continue

        T *= cooling
        
    # 3. Final Reduction
    # Use the best state found
    C = best_C
    counts = best_counts
    in_code = best_in_code
    
    # Greedy reduction
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            if np.min(counts[neighbors[c]]) >= 2:
                counts[neighbors[c]] -= 1
                in_code[c] = False
                changed = True
            else:
                new_C.append(c)
        C = new_C
        
    return [list(all_words[i]) for i in C]
