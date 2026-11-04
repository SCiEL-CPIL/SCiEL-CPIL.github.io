import itertools
import random
import sys
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115
    
    # 1. Precomputation
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubsets)
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    ksub_to_idx = {ksub: i for i, ksub in enumerate(all_ksubsets)}
    
    # Incidence data
    ksub_tsubs = []
    t_to_ksubs = [[] for _ in range(n_tsubs)]
    
    for i, ksub in enumerate(all_ksubsets):
        ts_idxs = []
        for tsub in itertools.combinations(ksub, t):
            t_idx = tsub_to_idx[tsub]
            ts_idxs.append(t_idx)
            t_to_ksubs[t_idx].append(i)
        ksub_tsubs.append(ts_idxs)
        
    # 2. Initial Solution via Fast Greedy (Matrix based)
    # Build incidence matrix for fast greedy
    incidence = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ts_list in enumerate(ksub_tsubs):
        for t_idx in ts_list:
            incidence[i, t_idx] = 1
            
    uncovered_mask = np.ones(n_tsubs, dtype=np.uint8)
    init_blocks = []
    
    while uncovered_mask.any():
        covered_counts = incidence @ uncovered_mask
        best_idx = int(np.argmax(covered_counts))
        if covered_counts[best_idx] == 0:
            break
        init_blocks.append(list(all_ksubsets[best_idx]))
        uncovered_mask[incidence[best_idx] == 1] = 0
        
    # Convert to indices
    current_indices = [ksub_to_idx[tuple(b)] for b in init_blocks]
    
    # Compute counts
    current_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_indices:
        for t_idx in ksub_tsubs[idx]:
            current_counts[t_idx] += 1
            
    best_indices = current_indices.copy()
    best_size = len(current_indices)
    
    rng = random.Random(42)
    
    # 3. ILS Loop
    while time.time() - start_time < timeout:
        if not current_indices:
            break
            
        # Perturb: Remove a random block
        pos = rng.randint(0, len(current_indices) - 1)
        removed_idx = current_indices[pos]
        
        new_indices = current_indices.copy()
        new_indices.pop(pos)
        
        new_counts = current_counts.copy()
        for t_idx in ksub_tsubs[removed_idx]:
            new_counts[t_idx] -= 1
            
        uncovered = set(np.where(new_counts == 0)[0])
        
        # Repair
        while uncovered:
            # Pick a random uncovered t-subset to focus on
            t_target = next(iter(uncovered))
            candidates = t_to_ksubs[t_target]
            
            best_candidate = -1
            best_gain = 0
            
            for k_idx in candidates:
                gain = 0
                for t_idx in ksub_tsubs[k_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = k_idx
                    
            if best_candidate == -1:
                break
                
            new_indices.append(best_candidate)
            for t_idx in ksub_tsubs[best_candidate]:
                new_counts[t_idx] += 1
                if new_counts[t_idx] == 1:
                    uncovered.discard(t_idx)
                    
        # Accept/Reject
        if len(new_indices) < best_size:
            best_indices = new_indices
            best_size = len(new_indices)
            current_indices = new_indices
            current_counts = new_counts
        elif len(new_indices) == best_size:
            # Accept with some probability to explore
            if rng.random() < 0.3:
                current_indices = new_indices
                current_counts = new_counts
        # Else reject, keep current
        
    # Return best solution
    return [list(all_ksubsets[i]) for i in best_indices]
