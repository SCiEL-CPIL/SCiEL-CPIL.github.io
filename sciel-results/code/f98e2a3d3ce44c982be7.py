import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256
    
    # Precompute AP structure
    ap_to_idx = {}
    idx_to_ap = {}
    ap_indices = []
    num_aps = 0
    
    for d in range(1, n):
        for a in range(d):
            length = (n - 1 - a) // d + 1
            if length < 2:
                continue
            ap_idx = num_aps
            num_aps += 1
            ap_to_idx[(a, d)] = ap_idx
            idx_to_ap[ap_idx] = (a, d)
            ap_indices.append(list(range(a, n, d)))
            
    index_to_aps = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        a, d = idx_to_ap[ap_idx]
        for idx in ap_indices[ap_idx]:
            index_to_aps[idx].append(ap_idx)
            
    # Initialize signs
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    ap_max_abs = np.zeros(num_aps, dtype=np.int32)
    ap_max_val = np.zeros(num_aps, dtype=np.int32)
    
    def compute_ap_stats(ap_idx):
        a, d = idx_to_ap[ap_idx]
        vals = signs[a::d]
        cum = np.cumsum(vals, dtype=np.int32)
        abs_cum = np.abs(cum)
        max_abs = np.max(abs_cum)
        max_idx = np.argmax(abs_cum)
        max_val = cum[max_idx]
        return max_abs, max_val

    def update_ap(ap_idx):
        ap_max_abs[ap_idx], ap_max_val[ap_idx] = compute_ap_stats(ap_idx)

    for ap_idx in range(num_aps):
        update_ap(ap_idx)
        
    current_max = np.max(ap_max_abs)
    best_max = current_max
    best_signs = signs.copy()
    
    rng = random.Random(42)
    temp = 1.0
    cooling = 0.999
    no_improve = 0
    
    while time.time() < deadline - 2:
        if current_max == 1:
            break
            
        worst_aps = np.where(ap_max_abs == current_max)[0]
        
        if len(worst_aps) > 10:
            worst_aps = rng.sample(list(worst_aps), 10)
            
        found_improvement = False
        
        for ap_idx in worst_aps:
            if time.time() > deadline - 2:
                break
                
            a, d = idx_to_ap[ap_idx]
            ap_vals = signs[a::d]
            max_val = ap_max_val[ap_idx]
            
            candidates = []
            if max_val > 0:
                for k, val in enumerate(ap_vals):
                    if val == 1:
                        candidates.append(a + k * d)
            else:
                for k, val in enumerate(ap_vals):
                    if val == -1:
                        candidates.append(a + k * d)
                        
            if not candidates:
                continue
                
            rng.shuffle(candidates)
            
            for idx in candidates:
                if time.time() > deadline - 2:
                    break
                
                old_sign = signs[idx]
                signs[idx] = -old_sign
                
                affected_aps = index_to_aps[idx]
                for aff_idx in affected_aps:
                    update_ap(aff_idx)
                
                new_max = np.max(ap_max_abs)
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                    found_improvement = True
                    no_improve = 0
                    break
                else:
                    delta = new_max - current_max
                    if delta > 0:
                        if rng.random() < math.exp(-delta / temp):
                            current_max = new_max
                            no_improve += 1
                            found_improvement = True
                            break
                    signs[idx] = old_sign
                    for aff_idx in affected_aps:
                        update_ap(aff_idx)
                    
            if found_improvement:
                break
                
        if not found_improvement:
            idx = rng.randint(0, n - 1)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            affected_aps = index_to_aps[idx]
            for aff_idx in affected_aps:
                update_ap(aff_idx)
                
            new_max = np.max(ap_max_abs)
            
            delta = new_max - current_max
            if delta < 0:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                no_improve = 0
            else:
                if delta == 0 or rng.random() < math.exp(-delta / temp):
                    current_max = new_max
                    no_improve += 1
                else:
                    signs[idx] = old_sign
                    for aff_idx in affected_aps:
                        update_ap(aff_idx)
                    no_improve += 1
                    
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
        if no_improve > 1000:
            temp = 1.0
            no_improve = 0
            for _ in range(5):
                idx = rng.randint(0, n-1)
                signs[idx] = -signs[idx]
                for aff_idx in index_to_aps[idx]:
                    update_ap(aff_idx)
            current_max = np.max(ap_max_abs)

    return best_signs.tolist()
