import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    random.seed(42)
    
    # Precompute comparability using NumPy for speed
    # comp[i][j] = 1 if i subset j or j subset i
    masks = np.arange(total_masks, dtype=np.int32)
    # For each pair, check if (i & ~j) == 0 or (j & ~i) == 0
    # Vectorized: for each i, compute which j are supersets or subsets
    comp = np.zeros((total_masks, total_masks), dtype=np.int8)
    
    # Build comp efficiently: for each mask i, find all supersets and subsets
    # Supersets of i: j where (i & ~j) == 0, i.e., j | i == j
    # Subsets of i: j where (j & ~i) == 0, i.e., j | i == i
    
    # For each i, supersets are j where j has all bits of i
    # This is equivalent to: j & i == i
    # Subsets: j & i == j
    
    # We can precompute using bit manipulation
    # For speed, use the fact that for mask i with k bits set,
    # there are 2^(n-k) supersets and 2^k subsets (including itself)
    
    for i in range(total_masks):
        # Find all supersets of i
        # j is superset of i iff (j & i) == i
        # Enumerate: start with i, add any subset of the complement
        complement = total_masks - 1 - i
        sub = complement
        while True:
            j = i | sub
            comp[i][j] = 1
            comp[j][i] = 1
            if sub == 0:
                break
            sub = (sub - 1) & complement
    
    # Now comp is symmetric and includes self-comparison (comp[i][i]=1)
    # But we only care about distinct pairs, so self doesn't matter for scoring
    # Actually for the contribution calculation, comp[i][i] should be 0 since i != j
    # Let's zero out the diagonal
    np.fill_diagonal(comp, 0)
    
    # For local search, we need fast access. Use numpy arrays.
    # comp is (total_masks, total_masks) int8 array
    
    def greedy_init(seed_offset=0):
        """Generate a diverse greedy initialization."""
        rng = random.Random(42 + seed_offset)
        family = []
        used = set()
        
        # Start with a random chain-like structure
        # Pick a random permutation of elements to build a chain
        perm = list(range(n))
        rng.shuffle(perm)
        
        cur = 0
        family.append(cur)
        used.add(cur)
        for i in perm:
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        # Greedily add remaining sets
        for _ in range(m - len(family)):
            best_gain = -1
            best_mask = None
            
            # Use numpy for fast gain computation
            # gain[s] = sum of comp[s][f] for f in family, excluding if s in used
            fam_arr = np.array(family, dtype=np.int32)
            gains = comp[:, fam_arr].sum(axis=1)
            
            # Mask out used sets
            for u in used:
                gains[u] = -1
            
            best_mask = int(np.argmax(gains))
            best_gain = int(gains[best_mask])
            
            if best_gain < 0:
                break
            
            family.append(best_mask)
            used.add(best_mask)
        
        # Fill remaining randomly
        while len(family) < m:
            s = rng.randint(0, total_masks - 1)
            if s not in used:
                family.append(s)
                used.add(s)
        
        return family
    
    def local_search(family, deadline, rng):
        """Simulated annealing local search with single-swap moves."""
        fam = list(family)
        in_fam = [False] * total_masks
        for mask in fam:
            in_fam[mask] = True
        
        # sum_comp[s] = number of family members comparable to s
        sum_comp = np.zeros(total_masks, dtype=np.int32)
        fam_arr = np.array(fam, dtype=np.int32)
        # sum_comp = row sums of comp[fam_arr, :]
        sum_comp = comp[fam_arr, :].sum(axis=0)
        
        contribs = sum_comp[fam_arr].copy()
        current_score = int(contribs.sum() // 2)
        best_score = current_score
        best_fam = fam[:]
        
        T = 5.0
        cooling = 0.9995
        T_min = 0.05
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline:
            # Find best single-swap move
            # For each position i, best replacement s maximizes:
            #   sum_comp[s] - comp[s][fam[i]] - contribs[i]
            # = (sum_comp[s] - comp[s][fam[i]]) - contribs[i]
            
            best_delta = -1000
            best_i = -1
            best_s = -1
            
            # Vectorized: for each i, compute val[s] = sum_comp[s] - comp[s][fam[i]]
            # Then find argmax over s not in fam
            # This is m * total_masks operations, do it in chunks or accept the cost
            
            # Alternative: precompute for all s the "potential" and adjust per i
            # potential[s] = sum_comp[s]  (same for all i)
            # val_i[s] = potential[s] - comp[s][fam[i]]
            # We want max_s val_i[s] for s not in fam, then delta = max_val - contribs[i]
            
            # Since m=40 and total_masks=4096, 40*4096 = 163840 ops per iteration
            # That's fine in Python if we use numpy
            
            for i in range(m):
                old_val = fam[i]
                # val[s] = sum_comp[s] - comp[s][old_val]
                # comp[s][old_val] is column old_val of comp
                val = sum_comp - comp[:, old_val]
                
                # Mask out in-fam
                for f in fam:
                    val[f] = -1000
                
                max_val = int(np.max(val))
                best_s_for_i = int(np.argmax(val))
                
                if max_val > -999:
                    delta = max_val - int(contribs[i])
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_s = best_s_for_i
            
            accept = False
            move_i = -1
            move_s = -1
            move_delta = 0
            
            if best_delta > 0:
                accept = True
                move_i = best_i
                move_s = best_s
                move_delta = best_delta
            else:
                # Occasionally accept worse moves (annealing)
                if rng.random() < 0.15:
                    move_i = rng.randint(0, m - 1)
                    while True:
                        s = rng.randint(0, total_masks - 1)
                        if not in_fam[s]:
                            break
                    move_s = s
                    
                    old_val = fam[move_i]
                    new_contrib = int(sum_comp[move_s] - comp[move_s][old_val])
                    move_delta = new_contrib - int(contribs[move_i])
                    
                    if move_delta >= 0 or rng.random() < np.exp(move_delta / T):
                        accept = True
                    else:
                        accept = False
                else:
                    accept = False
            
            if accept:
                old_val = fam[move_i]
                new_val = move_s
                
                fam[move_i] = new_val
                in_fam[old_val] = False
                in_fam[new_val] = True
                
                # Update sum_comp: for each s, sum_comp[s] changes by
                # comp[s][new_val] - comp[s][old_val]
                sum_comp += comp[:, new_val] - comp[:, old_val]
                
                # Update contribs
                contribs[move_i] = int(sum_comp[new_val])
                for k in range(m):
                    if k != move_i:
                        contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                
                current_score = int(contribs.sum() // 2)
                
                if current_score > best_score:
                    best_score = current_score
                    best_fam = fam[:]
                    no_improve = 0
                else:
                    no_improve += 1
                
                T *= cooling
                if T < T_min:
                    T = T_min
                
                if no_improve > max_no_improve:
                    # Reset to best
                    fam = best_fam[:]
                    in_fam = [False] * total_masks
                    for mask in fam:
                        in_fam[mask] = True
                    fam_arr = np.array(fam, dtype=np.int32)
                    sum_comp = comp[fam_arr, :].sum(axis=0)
                    contribs = sum_comp[fam_arr].copy()
                    current_score = best_score
                    T = T_min * 3
                    no_improve = 0
            else:
                no_improve += 1
                T *= cooling
                if T < T_min:
                    T = T_min
        
        return best_fam, best_score
    
    # Multi-start: try several diverse initializations
    start_time = time.time()
    total_time_budget = 80  # Safety margin before 90s kill
    
    # Allocate time: 3 starts, first gets 40%, next two get 30% each
    num_starts = 3
    time_splits = [0.4, 0.3, 0.3]
    
    best_overall_family = None
    best_overall_score = 0
    
    for start_idx in range(num_starts):
        start_deadline = start_time + total_time_budget * sum(time_splits[:start_idx+1])
        # Leave a small buffer
        actual_deadline = min(start_deadline, start_time + total_time_budget) - 0.5
        
        if time.time() >= actual_deadline:
            break
        
        family = greedy_init(seed_offset=start_idx * 100)
        fam, score = local_search(family, actual_deadline, random.Random(42 + start_idx * 7))
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam
    
    # Final polish: one more local search pass on the best solution with tighter parameters
    if time.time() < start_time + total_time_budget - 1:
        polish_deadline = start_time + total_time_budget - 0.5
        # Use a fresh RNG for polish
        best_overall_family, best_overall_score = local_search(
            best_overall_family, polish_deadline, random.Random(999)
        )
    
    # Convert to list of lists
    result = []
    for mask in best_overall_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
