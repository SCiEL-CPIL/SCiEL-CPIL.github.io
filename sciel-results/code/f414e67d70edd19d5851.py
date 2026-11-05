import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1..n]."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 1
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            m = (n - 1 - r) // d + 1
            if m <= 1:
                continue
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            cum = np.concatenate(([0], np.cumsum(sub)))
            running_min = 0
            running_max = 0
            max_sum = 0
            min_sum = 0
            for i in range(1, m + 1):
                val = cum[i]
                if val < running_min:
                    running_min = val
                if val > running_max:
                    running_max = val
                diff_max = val - running_min
                diff_min = val - running_max
                if diff_max > max_sum:
                    max_sum = diff_max
                if diff_min < min_sum:
                    min_sum = diff_min
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
    return max_abs

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2 and membership info."""
    aps = []
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    return aps, ap_membership

def _compute_ap_extrema(aps, signs):
    """Compute max and min subarray sums for each AP."""
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
        min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])
    return max_sums, min_sums

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """Update max/min subarray sum for a single AP in-place."""
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    max_sums[ap_idx] = np.max(cum_ext[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum_ext[1:] - max_prefix[:-1])

def _find_worst_aps(max_sums, min_sums, aps, k=10):
    """Find the k APs with the largest absolute discrepancy."""
    n_aps = len(aps)
    if n_aps == 0:
        return []
    scores = np.maximum(max_sums.astype(np.int64), -min_sums.astype(np.int64))
    top_k = np.argsort(scores)[::-1][:k]
    return [(int(i), int(scores[i])) for i in top_k if scores[i] > 1]

def _soft_max_disc(max_sums, min_sums, beta=2.0):
    """Soft surrogate for max discrepancy using log-sum-exp."""
    if len(max_sums) == 0:
        return 1.0
    # Use log-sum-exp for numerical stability
    pos = max_sums.astype(np.float64)
    neg = -min_sums.astype(np.float64)
    all_vals = np.concatenate([pos, neg])
    m = np.max(all_vals)
    if m < 1:
        return 1.0
    log_sum = np.log(np.sum(np.exp((all_vals - m) * beta))) / beta + m
    return log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    n_aps = len(aps)
    
    # Thue-Morse initialization
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    max_sums, min_sums = _compute_ap_extrema(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    if best_disc == 1:
        return best_signs
    
    # Iterated Local Search
    best_disc_il = best_disc
    best_signs_il = best_signs[:]
    
    num_restarts = 8
    restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        # If this is not the first restart, perturb from best
        if restart > 0:
            signs = best_signs_il[:]
            # Perturb with k random flips
            k_perturb = rng.randint(3, 8)
            for _ in range(k_perturb):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
                for ap_idx in ap_membership[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            current_disc = _get_max_disc(max_sums, min_sums)
        
        # Local search phase
        temp = 2.0
        cool_rate = 0.998
        no_improve = 0
        max_no_improve = 500
        beta_soft = 1.5
        
        while time.time() < deadline - 2:
            # Choose move: 70% random, 30% guided by worst APs
            if rng.random() < 0.30 and n_aps > 0:
                worst_aps = _find_worst_aps(max_sums, min_sums, aps, k=5)
                if worst_aps:
                    ap_idx, disc_val = rng.choice(worst_aps)
                    ap = aps[ap_idx]
                    # Pick a random element from the worst AP to flip
                    idx = rng.choice(ap)
                else:
                    idx = rng.randint(0, n - 1)
            else:
                idx = rng.randint(0, n - 1)
            
            old_val = signs[idx]
            new_val = -old_val
            signs[idx] = new_val
            
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
            
            new_disc = _get_max_disc(max_sums, min_sums)
            
            delta = new_disc - current_disc
            # Use soft surrogate for acceptance when delta >= 0
            if delta < 0:
                accept = True
            else:
                # Compute soft surrogate change for better gradient info
                soft_before = _soft_max_disc(
                    np.where(max_sums == current_disc, max_sums, max_sums),
                    np.where(min_sums == -current_disc, min_sums, min_sums),
                    beta_soft
                )
                accept = rng.random() < np.exp(-delta / max(temp, 0.01))
            
            if accept:
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
                    no_improve = 0
                    if new_disc < best_disc_il:
                        best_disc_il = new_disc
                        best_signs_il = signs[:]
                else:
                    no_improve += 1
            else:
                signs[idx] = old_val
                for ap_idx in ap_membership[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums)
                no_improve += 1
            
            temp *= cool_rate
            
            if best_disc_il == 1:
                return best_signs_il
            if no_improve > max_no_improve:
                break
    
    return best_signs_il
