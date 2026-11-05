import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Leave 5s margin
    
    n = 256
    
    # 1. Build AP Structure
    # APs are (start, step) where start < n, step < n, and length >= 2
    aps = []
    ap_indices = [] # List of lists of indices
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for r in range(d):
            if r >= n: continue
            # Indices: r, r+d, r+2d...
            idxs = list(range(r, n, d))
            if len(idxs) >= 2:
                ap_id = len(aps)
                aps.append((r, d))
                ap_indices.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
                    
    num_aps = len(aps)
    
    # 2. Initialize Signs
    # Thue-Morse
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    # 3. Initialize Sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for i, idxs in enumerate(ap_indices):
        ap_sums[i] = np.sum(signs[idxs])
        
    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    # 4. Local Search
    # We will alternate between Greedy and SA
    
    # Greedy Phase
    no_improve_count = 0
    max_no_improve_greedy = 50
    
    while time.time() < deadline - 10:
        if no_improve_count > max_no_improve_greedy:
            break
            
        # Find candidates
        # Get indices of top 20 worst APs
        # abs_sums = np.abs(ap_sums)
        # worst_aps = np.argpartition(abs_sums, -20)[-20:]
        
        # Instead of top 20, let's take all APs with abs(sum) > current_max - 2
        # This ensures we are looking at the "active" constraints.
        active_aps = np.where(np.abs(ap_sums) >= current_max - 1)[0]
        
        candidates = set()
        for ap_id in active_aps:
            for idx in ap_indices[ap_id]:
                candidates.add(idx)
                
        if not candidates:
            candidates = set(range(n))
            
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        improved = False
        for idx in cand_list:
            if time.time() > deadline - 10: break
            
            old_val = signs[idx]
            new_val = -old_val
            
            # Update affected APs
            affected_aps = index_to_aps[idx]
            old_sums = np.zeros(len(affected_aps), dtype=np.int32)
            for k, ap_id in enumerate(affected_aps):
                old_sums[k] = ap_sums[ap_id]
                ap_sums[ap_id] -= 2 * old_val # s_i becomes -s_i, sum changes by -2*s_i
                
            new_max = int(np.max(np.abs(ap_sums)))
            
            if new_max < current_max:
                # Accept
                signs[idx] = new_val
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True
                # Restart candidate search? Or continue?
                # Restarting is safer to find the new active set.
                break 
            else:
                # Revert
                for k, ap_id in enumerate(affected_aps):
                    ap_sums[ap_id] = old_sums[k]
                    
        if not improved:
            no_improve_count += 1
            # If stuck, maybe switch to SA?
            # For now, just break greedy phase
            
    # SA Phase
    if time.time() < deadline - 5:
        signs = best_signs.copy()
        # Recompute sums for best_signs
        for i, idxs in enumerate(ap_indices):
            ap_sums[i] = np.sum(signs[idxs])
        current_max = int(np.max(np.abs(ap_sums)))
        
        temp = 5.0
        cool_rate = 0.99
        no_improve_sa = 0
        max_no_improve_sa = 1000
        
        while time.time() < deadline - 5 and no_improve_sa < max_no_improve_sa:
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            new_val = -old_val
            
            affected_aps = index_to_aps[idx]
            old_sums = np.zeros(len(affected_aps), dtype=np.int32)
            for k, ap_id in enumerate(affected_aps):
                old_sums[k] = ap_sums[ap_id]
                ap_sums[ap_id] -= 2 * old_val
                
            new_max = int(np.max(np.abs(ap_sums)))
            delta = new_max - current_max
            
            if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / temp)):
                signs[idx] = new_val
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    no_improve_sa = 0
                else:
                    no_improve_sa += 1
            else:
                for k, ap_id in enumerate(affected_aps):
                    ap_sums[ap_id] = old_sums[k]
                no_improve_sa += 1
                
            temp *= cool_rate
            
            if best_max == 1:
                return best_signs.tolist()
                
    return best_signs.tolist()
