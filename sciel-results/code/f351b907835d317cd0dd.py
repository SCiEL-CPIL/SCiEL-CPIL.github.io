import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort()
    
    # Precompute indices for each modulus
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices
    
    best_uncovered = base
    best_classes = []
    
    # Strategy 1: Greedy with vectorized gain computation
    num_greedy_runs = 30
    
    for run in range(num_greedy_runs):
        if time.time() > deadline - 5:
            break
            
        random.seed(run * 7919 + 12345)
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = list(divisors)
        classes = []
        
        while available_moduli:
            if time.time() > deadline - 2:
                break
                
            best_gain = -1
            best_m = None
            best_r = None
            best_indices = None
            
            for m in available_moduli:
                res_indices_list = indices_by_m[m]
                uncovered_mask = (covered == 0)
                
                # Compute gains vectorized
                gains = np.array([np.sum(uncovered_mask[idx]) for idx in res_indices_list])
                
                if run > 0:
                    noise = np.random.uniform(0, 0.5, size=len(gains))
                    gains = gains + noise
                
                max_idx = np.argmax(gains)
                if gains[max_idx] > best_gain:
                    best_gain = gains[max_idx]
                    best_m = m
                    best_r = max_idx
                    best_indices = res_indices_list[max_idx]
            
            if best_gain <= 0:
                break
                
            classes.append([best_m, best_r])
            covered[best_indices] = 1
            available_moduli.remove(best_m)
            
            current_uncovered = np.sum(covered == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
            if best_uncovered == 0:
                return best_classes
    
    # Strategy 2: Local search polish
    if best_classes and time.time() < deadline - 10:
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered[indices_by_m[m][r]] = 1
        
        current_uncovered = np.sum(covered == 0)
        
        # Try swapping residues for each class
        for i, (m, r) in enumerate(best_classes):
            if time.time() > deadline - 5:
                break
            
            for r_new in range(m):
                if r_new == r:
                    continue
                
                old_idx = indices_by_m[m][r]
                new_idx = indices_by_m[m][r_new]
                
                # Efficiently compute the change
                # Positions in old_idx that are NOT in new_idx and are covered only by this class
                # Positions in new_idx that are NOT in old_idx and are currently uncovered
                
                # Create a set for fast lookup
                old_set = set(old_idx)
                new_set = set(new_idx)
                
                # Positions that lose coverage
                losing = old_set - new_set
                # Positions that gain coverage  
                gaining = new_set - old_set
                
                # Check if losing positions are covered by other classes
                # This is expensive, so let's just recompute for promising swaps
                # Only try if gaining has uncovered positions
                
                if len(gaining) == 0:
                    continue
                    
                # Quick check: how many gaining positions are currently uncovered?
                gain_count = np.sum(covered[list(gaining)] == 0)
                if gain_count == 0:
                    continue
                
                # Do the full recompute
                covered_temp = covered.copy()
                covered_temp[old_idx] = 0
                covered_temp[new_idx] = 1
                
                new_uncovered = np.sum(covered_temp == 0)
                
                if new_uncovered < current_uncovered:
                    covered = covered_temp
                    current_uncovered = new_uncovered
                    best_classes[i][1] = r_new
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c[:] for c in best_classes]
    
    return best_classes
