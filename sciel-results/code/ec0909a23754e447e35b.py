import time
import random
import numpy as np

def _build_ap_structure(n):
    """Build list of all arithmetic progressions with length >= 2."""
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

def _compute_ap_sums(aps, signs):
    """Compute sum for each AP."""
    sums = np.zeros(len(aps), dtype=np.int32)
    for i, ap in enumerate(aps):
        sums[i] = np.sum(signs[ap])
    return sums

def _get_max_disc(sums):
    if len(sums) == 0:
        return 1
    return int(np.max(np.abs(sums)))

def _update_ap_sums(ap, signs, ap_idx, sums):
    """Update sum for a single AP."""
    # Recompute sum for this AP
    sums[ap_idx] = np.sum(signs[ap])

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 256
    rng = random.Random(42)
    
    # Build AP structure
    aps, ap_membership = _build_ap_structure(n)
    
    # Thue-Morse initialization
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    signs_arr = np.array(signs, dtype=np.int8)
    
    sums = _compute_ap_sums(aps, signs_arr)
    current_disc = _get_max_disc(sums)
    best_disc = current_disc
    best_signs = signs[:]
    
    # If we already have disc 1 or 2, we might be done.
    # Min disc is at least 2 for N>=4.
    if best_disc <= 2:
        return best_signs

    # Local Search
    temp = 1.0
    cool_rate = 0.995
    
    while time.time() < deadline:
        if best_disc <= 2:
            break
            
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        new_val = -old_val
        
        signs[idx] = new_val
        signs_arr[idx] = new_val
        
        # Update affected APs
        for ap_idx in ap_membership[idx]:
            _update_ap_sums(aps[ap_idx], signs_arr, ap_idx, sums)
        
        new_disc = _get_max_disc(sums)
        
        delta = new_disc - current_disc
        if delta < 0 or (delta == 0 and rng.random() < 0.5) or rng.random() < np.exp(-delta / max(temp, 0.01)):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
        else:
            signs[idx] = old_val
            signs_arr[idx] = old_val
            for ap_idx in ap_membership[idx]:
                _update_ap_sums(aps[ap_idx], signs_arr, ap_idx, sums)
            
        temp *= cool_rate

    return best_signs
