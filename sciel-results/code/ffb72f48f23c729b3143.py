import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0
    
    # 1. Precompute APs
    ap_list = []
    idx_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for start in range(n):
            indices = []
            curr = start
            while curr < n:
                indices.append(curr)
                curr += d
            if len(indices) > 0:
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    idx_to_aps[idx].append(ap_id)
    
    num_aps = len(ap_list)
    offset = n
    size = 2 * n + 1
    
    # 2. Greedy Construction
    signs = [0] * n
    sums = [0] * num_aps
    counts = [0] * size
    counts[0] = num_aps
    current_max = 0
    rng = random.Random(42)
    
    def calc_new_max(affected, delta, cur_sums, cur_counts, cur_max):
        # Simulate applying delta to affected APs
        # Returns new_max
        
        max_new_val = 0
        for ap_id in affected:
            new_s = cur_sums[ap_id] + delta
            abs_new = abs(new_s)
            if abs_new > max_new_val:
                max_new_val = abs_new
        
        if max_new_val > cur_max:
            return max_new_val
            
        # Check if cur_max bucket is emptied
        count_cur_max_before = 0
        count_cur_max_after = 0
        
        for ap_id in affected:
            old_s = cur_sums[ap_id]
            new_s = old_s + delta
            
            if abs(old_s) == cur_max:
                count_cur_max_before += 1
            if abs(new_s) == cur_max:
                count_cur_max_after += 1
        
        total_cur_max = cur_counts[cur_max + offset] + cur_counts[-cur_max + offset]
        new_total_cur_max = total_cur_max - count_cur_max_before + count_cur_max_after
        
        if new_total_cur_max > 0:
            return cur_max
        else:
            # Max dropped. Find next max.
            for d in range(cur_max - 1, -1, -1):
                virtual_count = cur_counts[d + offset] + cur_counts[-d + offset]
                for ap_id in affected:
                    old_s = cur_sums[ap_id]
                    new_s = old_s + delta
                    if abs(old_s) == d:
                        virtual_count -= 1
                    if abs(new_s) == d:
                        virtual_count += 1
                if virtual_count > 0:
                    return d
            return 0

    for i in range(n):
        affected = idx_to_aps[i]
        
        max_plus = calc_new_max(affected, 1, sums, counts, current_max)
        max_minus = calc_new_max(affected, -1, sums, counts, current_max)
        
        if max_plus <= max_minus:
            chosen_sign = 1
            new_max = max_plus
        else:
            chosen_sign = -1
            new_max = max_minus
            
        signs[i] = chosen_sign
        delta = chosen_sign
        
        for ap_id in affected:
            old_s = sums[ap_id]
            new_s = old_s + delta
            sums[ap_id] = new_s
            counts[old_s + offset] -= 1
            counts[new_s + offset] += 1
            
        current_max = new_max

    best_signs = signs[:]
    best_max = current_max
    best_sums = sums[:]
    best_counts = counts[:]

    # 3. Iterated Hill Climbing with constraint-directed targeting
    opt_signs = signs[:]
    opt_sums = sums[:]
    opt_counts = counts[:]
    opt_max = current_max

    def get_max_disc(c):
        for d_val in range(n, -1, -1):
            if c[d_val + offset] > 0 or c[-d_val + offset] > 0:
                return d_val
        return 0

    def find_worst_ap_indices():
        """Find indices that appear in the currently worst APs."""
        worst_aps = []
        for ap_id in range(num_aps):
            s = opt_sums[ap_id]
            if abs(s) == opt_max:
                worst_aps.append(ap_id)
        # Collect indices from worst APs, weighted by frequency
        idx_freq = defaultdict(int)
        for ap_id in worst_aps:
            for idx in ap_list[ap_id]:
                idx_freq[idx] += 1
        if not idx_freq:
            return list(range(n))
        # Return top indices by frequency
        sorted_idxs = sorted(idx_freq.keys(), key=lambda x: -idx_freq[x])
        return sorted_idxs[:50]

    def mincut_refinement(signs_arr, sums_arr, counts_arr, cur_max, max_iters=300):
        """
        Min-cut inspired refinement using surrogate objective.
        """
        if cur_max <= 0:
            return False
        
        improved = False
        worst_indices = find_worst_ap_indices()
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 2.0:
                break
            if not worst_indices:
                break
            
            i = rng.choice(worst_indices)
            old_sign = signs_arr[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = idx_to_aps[i]
            
            for ap_id in affected:
                old_s = sums_arr[ap_id]
                new_s = old_s + delta
                sums_arr[ap_id] = new_s
                counts_arr[old_s + offset] -= 1
                counts_arr[new_s + offset] += 1
            
            new_max = cur_max
            if counts_arr[cur_max + offset] == 0 and counts_arr[-cur_max + offset] == 0:
                while new_max > 0 and counts_arr[new_max + offset] == 0 and counts_arr[-new_max + offset] == 0:
                    new_max -= 1
            for ap_id in affected:
                s_val = abs(sums_arr[ap_id])
                if s_val > new_max:
                    new_max = s_val
            
            if new_max < cur_max:
                cur_max = new_max
                improved = True
                signs_arr[i] = new_sign
                worst_indices = find_worst_ap_indices()
            else:
                for ap_id in affected:
                    new_s = sums_arr[ap_id]
                    old_s = new_s - delta
                    sums_arr[ap_id] = old_s
                    counts_arr[new_s + offset] -= 1
                    counts_arr[old_s + offset] += 1
        
        return improved

    def polish_pass(signs_arr, sums_arr, counts_arr, cur_max, max_rounds=10):
        """
        High-precision polish: try all single flips and keep the best improvement.
        """
        improved = False
        for _ in range(max_rounds):
            if time.time() - start_time > time_limit - 1.0:
                break
            
            best_i = -1
            best_new_max = cur_max
            best_delta = 0
            best_affected = None
            
            for i in range(n):
                old_sign = signs_arr[i]
                new_sign = -old_sign
                delta = 2 * new_sign
                affected = idx_to_aps[i]
                
                new_max = cur_max
                for ap_id in affected:
                    old_s = sums_arr[ap_id]
                    new_s = old_s + delta
                    s_val = abs(new_s)
                    if s_val > new_max:
                        new_max = s_val
                
                if new_max >= cur_max:
                    count_at_max = 0
                    for ap_id in affected:
                        old_s = sums_arr[ap_id]
                        new_s = old_s + delta
                        if abs(old_s) == cur_max:
                            count_at_max -= 1
                        if abs(new_s) == cur_max:
                            count_at_max += 1
                    
                    total_at_max = counts_arr[cur_max + offset] + counts_arr[-cur_max + offset]
                    if total_at_max + count_at_max > 0:
                        new_max = cur_max
                    else:
                        for d in range(cur_max - 1, -1, -1):
                            virtual_count = counts_arr[d + offset] + counts_arr[-d + offset]
                            for ap_id in affected:
                                old_s = sums_arr[ap_id]
                                new_s = old_s + delta
                                if abs(old_s) == d:
                                    virtual_count -= 1
                                if abs(new_s) == d:
                                    virtual_count += 1
                            if virtual_count > 0:
                                new_max = d
                                break
                        else:
                            new_max = 0
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_i = i
                    best_delta = delta
                    best_affected = affected
            
            if best_i >= 0 and best_new_max < cur_max:
                signs_arr[best_i] *= -1
                for ap_id in best_affected:
                    old_s = sums_arr[ap_id]
                    new_s = old_s + best_delta
                    sums_arr[ap_id] = new_s
                    counts_arr[old_s + offset] -= 1
                    counts_arr[new_s + offset] += 1
                cur_max = best_new_max
                improved = True
            else:
                break
        
        return improved

    while time.time() - start_time < time_limit:
        # Phase 1: Constraint-directed hill climbing
        improved = True
        while improved and time.time() - start_time < time_limit - 3.0:
            improved = False
            
            # Pick index from worst APs (constraint-directed)
            if rng.random() < 0.7:
                worst_indices = find_worst_ap_indices()
                i = rng.choice(worst_indices)
            else:
                i = rng.randint(0, n - 1)
            
            old_sign = opt_signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = idx_to_aps[i]
            
            for ap_id in affected:
                old_s = opt_sums[ap_id]
                new_s = old_s + delta
                opt_sums[ap_id] = new_s
                opt_counts[old_s + offset] -= 1
                opt_counts[new_s + offset] += 1
            
            new_max = opt_max
            if opt_counts[opt_max + offset] == 0 and opt_counts[-opt_max + offset] == 0:
                while new_max > 0 and opt_counts[new_max + offset] == 0 and opt_counts[-new_max + offset] == 0:
                    new_max -= 1
            for ap_id in affected:
                s_val = abs(opt_sums[ap_id])
                if s_val > new_max:
                    new_max = s_val
            
            if new_max < opt_max:
                opt_max = new_max
                improved = True
                if opt_max < best_max:
                    best_max = opt_max
                    best_signs = opt_signs[:]
                    best_sums = opt_sums[:]
                    best_counts = opt_counts[:]
            else:
                for ap_id in affected:
                    new_s = opt_sums[ap_id]
                    old_s = new_s - delta
                    opt_sums[ap_id] = old_s
                    opt_counts[new_s + offset] -= 1
                    opt_counts[old_s + offset] += 1
        
        # Phase 2: Min-cut inspired refinement
        if time.time() - start_time < time_limit - 3.0:
            mincut_refinement(opt_signs, opt_sums, opt_counts, opt_max)
            if opt_max < best_max:
                best_max = opt_max
                best_signs = opt_signs[:]
                best_sums = opt_sums[:]
                best_counts = opt_counts[:]
        
        # Phase 3: Polish pass
        if time.time() - start_time < time_limit - 2.0:
            polish_pass(opt_signs, opt_sums, opt_counts, opt_max)
            if opt_max < best_max:
                best_max = opt_max
                best_signs = opt_signs[:]
                best_sums = opt_sums[:]
                best_counts = opt_counts[:]
        
        # Phase 4: Random restart (perturbation)
        if time.time() - start_time < time_limit - 3.0:
            k = rng.randint(5, 15)
            for _ in range(k):
                i = rng.randint(0, n - 1)
                opt_signs[i] *= -1
                delta = 2 * opt_signs[i]
                for ap_id in idx_to_aps[i]:
                    old_s = opt_sums[ap_id]
                    new_s = old_s + delta
                    opt_sums[ap_id] = new_s
                    opt_counts[old_s + offset] -= 1
                    opt_counts[new_s + offset] += 1
            
            opt_max = get_max_disc(opt_counts)

    return best_signs
