import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    max_abs = 0
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            
            # Get the subsequence at positions r, r+d, r+2d, ...
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            
            if len(sub) == 0:
                continue
                
            cum = np.concatenate(([0], np.cumsum(sub)))
            
            min_prefix = np.minimum.accumulate(cum)
            max_prefix = np.maximum.accumulate(cum)
            
            # Max subarray sum
            max_sub = np.max(cum[1:] - min_prefix[:-1])
            # Min subarray sum
            min_sub = np.min(cum[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def _build_ap_structure(n):
    """
    Precompute all arithmetic progressions.
    Returns:
    aps_list: list of lists of indices
    aps_np: list of numpy arrays of indices
    ap_membership: list of lists of AP indices for each position
    ap_lengths: list of lengths
    """
    aps_list = []
    aps_np = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps_list)
                aps_list.append(sub_indices)
                aps_np.append(np.array(sub_indices, dtype=np.int32))
                ap_lengths.append(len(sub_indices))
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
                    
    return aps_list, aps_np, ap_membership, ap_lengths

def _update_ap_py(ap, signs_list, ap_idx, max_sums, min_sums):
    """
    Update extrema for a single AP using Python loop.
    Optimized for short APs.
    """
    max_end = 0
    max_sum = 0
    min_end = 0
    min_sum = 0
    
    # signs_list is a list of ints
    for idx in ap:
        v = signs_list[idx]
        
        max_end += v
        if max_end < 0:
            max_end = 0
        if max_end > max_sum:
            max_sum = max_end
            
        min_end -= v
        if min_end < 0:
            min_end = 0
        if min_end > min_sum:
            min_sum = min_end
            
    max_sums[ap_idx] = max_sum
    min_sums[ap_idx] = -min_sum

def _update_ap_numpy(ap_np, signs_np, ap_idx, max_sums, min_sums):
    """
    Update extrema for a single AP using NumPy.
    Optimized for long APs.
    """
    vals = signs_np[ap_np]
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _get_max_disc(max_sums, min_sums):
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Build structure
    aps_list, aps_np, ap_membership, ap_lengths = _build_ap_structure(n)
    n_aps = len(aps_list)
    
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    # Candidates for initialization
    candidates = []
    
    # 1. Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)
    
    # 2. Random with various seeds
    for seed in range(10):
        rng_c = random.Random(seed)
        candidates.append([rng_c.choice((-1, 1)) for _ in range(n)])
        
    # 3. Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    best_signs_list = candidates[0][:]
    best_disc = float('inf')
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline - 10:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs_list = c[:]
            
    # Initialize working signs
    signs_list = best_signs_list[:]
    signs_np = np.array(signs_list, dtype=np.int8)
    
    # Compute initial extrema
    for ap_idx in range(n_aps):
        if ap_lengths[ap_idx] > 30:
            _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
        else:
            _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
            
    current_disc = _get_max_disc(max_sums, min_sums)
    
    # Simulated Annealing
    temp = 5.0
    cool_rate = 0.995
    min_temp = 0.1
    
    while time.time() < deadline - 5 and temp > min_temp:
        improved = False
        
        # Number of moves per temperature step
        # Estimate: 1 move ~ 200-500us. 1s ~ 2000-5000 moves.
        # We have ~45s left. Let's do a fixed number of iterations per temp.
        moves_per_step = 1000
        
        for _ in range(moves_per_step):
            if time.time() > deadline - 5:
                break
                
            idx = rng.randint(0, n - 1)
            old_val = signs_list[idx]
            new_val = -old_val
            
            # Update signs
            signs_list[idx] = new_val
            signs_np[idx] = new_val
            
            # Update affected APs
            for ap_idx in ap_membership[idx]:
                if ap_lengths[ap_idx] > 30:
                    _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
                else:
                    _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
                    
            new_disc = _get_max_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs_list = signs_list[:]
            else:
                delta = new_disc - current_disc
                if delta > 0 and temp > min_temp:
                    accept_prob = math.exp(-delta / temp)
                    if rng.random() < accept_prob:
                        current_disc = new_disc
                    else:
                        # Revert
                        signs_list[idx] = old_val
                        signs_np[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            if ap_lengths[ap_idx] > 30:
                                _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
                            else:
                                _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
                else:
                    # Revert
                    signs_list[idx] = old_val
                    signs_np[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        if ap_lengths[ap_idx] > 30:
                            _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
                        else:
                            _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
                            
        temp *= cool_rate
        
    # Final Polish: Hill Climbing
    # Try flipping each sign, accept if it improves
    polish_improved = True
    while polish_improved and time.time() < deadline - 1:
        polish_improved = False
        for idx in range(n):
            if time.time() > deadline - 1:
                break
                
            old_val = signs_list[idx]
            new_val = -old_val
            
            signs_list[idx] = new_val
            signs_np[idx] = new_val
            
            for ap_idx in ap_membership[idx]:
                if ap_lengths[ap_idx] > 30:
                    _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
                else:
                    _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
                    
            new_disc = _get_max_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                polish_improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs_list = signs_list[:]
            else:
                # Revert
                signs_list[idx] = old_val
                signs_np[idx] = old_val
                for ap_idx in ap_membership[idx]:
                    if ap_lengths[ap_idx] > 30:
                        _update_ap_numpy(aps_np[ap_idx], signs_np, ap_idx, max_sums, min_sums)
                    else:
                        _update_ap_py(aps_list[ap_idx], signs_list, ap_idx, max_sums, min_sums)
                        
    # Final verification
    final_disc = _max_ap_discrepancy(best_signs_list)
    if final_disc < best_disc:
        best_disc = final_disc
        
    return best_signs_list
