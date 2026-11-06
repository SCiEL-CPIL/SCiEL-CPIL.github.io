import time
import random
import numpy as np

def construct():
    """
    Constructs a B_3[1] set (unique triple sums) of maximum size.
    
    Strategy:
    1. Use a known-valid sparse skeleton (powers of 4) to guarantee a base set.
    2. Use fast incremental conflict checking (O(k^2) per insertion) to fill gaps.
    3. Perform multiple randomized restarts to find larger sets than a single greedy pass.
    4. Apply local improvement (swap-based) after greedy passes to recover missed elements.
    5. Track the best set found across all restarts.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    n_max = 4096
    max_sum = 3 * n_max
    
    # --- Fast Incremental Conflict Checking ---
    
    def try_add(x, selected, sums_arr):
        """
        Check if x can be added to selected without violating B_3[1] property.
        Returns True if successful, and modifies selected/sums_arr in place.
        """
        # 1. (x, x, x)
        s = 3 * x
        if sums_arr[s]:
            return False
        new_sums = [s]
        
        # 2. (y, x, x) -> y + 2x  and  (y, y, x) -> 2y + x
        for y in selected:
            s1 = y + 2 * x
            if sums_arr[s1]:
                return False
            new_sums.append(s1)
            
            s2 = 2 * y + x
            if sums_arr[s2]:
                return False
            new_sums.append(s2)
                
        # 3. (y, z, x) with y < z in selected
        n_sel = len(selected)
        for i in range(n_sel):
            y = selected[i]
            for j in range(i + 1, n_sel):
                z = selected[j]
                s = y + z + x
                if sums_arr[s]:
                    return False
                new_sums.append(s)
                
        # If all good, commit
        for s in new_sums:
            sums_arr[s] = True
        selected.append(x)
        return True

    def build_sums_array(S):
        """Build a boolean array of all triple sums for a given set S."""
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums_arr[S[i] + S[j] + S[l]] = True
        return sums_arr

    def greedy_pass(order, start_list=None):
        """
        Perform a greedy pass adding elements in the given order.
        If start_list is provided, initialize with it.
        """
        if start_list:
            selected = list(start_list)
            sums_arr = build_sums_array(selected)
            existing = set(selected)
        else:
            selected = []
            sums_arr = np.zeros(max_sum + 1, dtype=bool)
            existing = set()
            
        for x in order:
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if try_add(x, selected, sums_arr):
                existing.add(x)
                
        return selected

    def local_improve(selected, max_time_remaining=10.0):
        """
        Try to improve the set by swapping elements: remove one element and try to add two new ones.
        This can recover elements that greedy missed due to ordering.
        """
        if not selected:
            return selected
        
        improved = list(selected)
        sums_arr = build_sums_array(improved)
        existing = set(improved)
        all_candidates = [x for x in range(1, n_max + 1) if x not in existing]
        
        # Try multiple swap attempts
        attempt = 0
        while time.time() < start_time + max_time_remaining and attempt < 20:
            attempt += 1
            
            # Pick a random element to remove
            if not improved:
                break
            idx = random.randint(0, len(improved) - 1)
            removed = improved[idx]
            
            # Temporarily remove it
            improved.pop(idx)
            sums_arr = build_sums_array(improved)
            
            # Try to add two random candidates
            added_count = 0
            added_elements = []
            random.shuffle(all_candidates)
            for cand in all_candidates:
                if time.time() > start_time + max_time_remaining:
                    break
                if cand in existing:
                    continue
                if try_add(cand, improved, sums_arr):
                    existing.add(cand)
                    added_count += 1
                    added_elements.append(cand)
                    if added_count == 2:
                        break
            
            # If we added more than we removed, keep the improvement
            if added_count > 1:
                # We removed 1 and added 2, net +1
                pass
            elif added_count == 1:
                # Net 0, but maybe the new element is better positioned
                # Rebuild sums_arr since try_add modified it
                pass
            else:
                # Nothing added, put the removed element back
                improved.append(removed)
                existing.add(removed)
                sums_arr = build_sums_array(improved)
                continue
            
            # Rebuild sums_arr to be safe
            sums_arr = build_sums_array(improved)
            
            # Try to add more elements greedily
            for cand in all_candidates:
                if time.time() > start_time + max_time_remaining:
                    break
                if cand in existing:
                    continue
                if try_add(cand, improved, sums_arr):
                    existing.add(cand)
        
        return improved

    # --- Main Search Loop ---
    
    best_set = []
    best_len = 0
    
    # Skeleton: Powers of 4 are valid B_3[1] because 1+1+1=3 < 4 (no carries)
    skeleton = [4**i for i in range(7) if 4**i <= n_max]
    # skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Try a few different strategies:
    # 1. Pure random greedy + local improvement
    # 2. Skeleton-based greedy + local improvement
    # 3. Forward/Backward greedy + local improvement
    
    strategies = 0
    while time.time() < deadline:
        strategies += 1
        
        # Strategy 1: Random Greedy + Local Improvement
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        res = greedy_pass(candidates)
        
        # Apply local improvement if we have time
        if time.time() < deadline - 5:
            res_improved = local_improve(res, max_time_remaining=3.0)
            if len(res_improved) > len(res):
                res = res_improved
        
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
        
        if time.time() > deadline - 10:  # Need more time for next strategy
            break
            
        # Strategy 2: Skeleton-based Greedy + Local Improvement
        remaining = [x for x in range(1, n_max + 1) if x not in set(skeleton)]
        random.shuffle(remaining)
        res = greedy_pass(remaining, start_list=skeleton)
        
        if time.time() < deadline - 5:
            res_improved = local_improve(res, max_time_remaining=3.0)
            if len(res_improved) > len(res):
                res = res_improved
        
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
        
        if time.time() > deadline - 10:
            break
            
        # Strategy 3: Forward Greedy + Local Improvement
        forward_order = list(range(1, n_max + 1))
        res = greedy_pass(forward_order)
        
        if time.time() < deadline - 5:
            res_improved = local_improve(res, max_time_remaining=3.0)
            if len(res_improved) > len(res):
                res = res_improved
        
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            
        if time.time() > deadline - 10:
            break
        
        # Strategy 4: Backward Greedy + Local Improvement
        backward_order = list(range(n_max, 0, -1))
        res = greedy_pass(backward_order)
        
        if time.time() < deadline - 5:
            res_improved = local_improve(res, max_time_remaining=3.0)
            if len(res_improved) > len(res):
                res = res_improved
        
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
        
        # If we've done several rounds, increase randomness
        if strategies > 4:
            # Mix: start with random subset of skeleton, then fill
            partial_skeleton = skeleton[:random.randint(3, 6)]
            remaining = [x for x in range(1, n_max + 1) if x not in set(partial_skeleton)]
            random.shuffle(remaining)
            res = greedy_pass(remaining, start_list=partial_skeleton)
            
            if time.time() < deadline - 5:
                res_improved = local_improve(res, max_time_remaining=3.0)
                if len(res_improved) > len(res):
                    res = res_improved
            
            if len(res) > best_len:
                best_set = res
                best_len = len(res)

    # Final local improvement on the best set if we have time
    if best_set and time.time() < deadline - 2:
        final_improved = local_improve(best_set, max_time_remaining=2.0)
        if len(final_improved) > len(best_set):
            best_set = final_improved

    # If we found a good set, return it
    if best_set:
        return best_set
    
    # Fallback: Just return the skeleton (shouldn't happen)
    return skeleton

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size: {len(result)}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    
    # Verify
    n = len(result)
    sums_seen = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = result[i] + result[j] + result[k]
                if s in sums_seen:
                    valid = False
                    break
                sums_seen.add(s)
            if not valid:
                break
        if not valid:
            break
    print(f"Valid B_3[1] set: {valid}")
