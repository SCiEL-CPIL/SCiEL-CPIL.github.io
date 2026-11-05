"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum on ANY arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1 (position i corresponds to index i+1).
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    max_abs = 1  # singletons always have abs sum 1
    
    # Iterate over step d from 1 to n-1
    for d in range(1, n):
        # For each residue class mod d
        for r in range(d):
            if r >= n:
                continue
            
            # Get the subsequence at positions r, r+d, r+2d, ...
            # Length m
            m = (n - 1 - r) // d + 1
            if m <= 1:
                continue
            
            # Extract subsequence
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            
            # Compute cumulative sum
            cum = np.concatenate(([0], np.cumsum(sub)))
            
            # Find max subarray sum and min subarray sum
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
    """Build list of all arithmetic progressions with length >= 2 and membership info.
    Also builds position-in-AP info for faster updates."""
    aps = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    # For each element, store list of (ap_idx, position_in_ap)
    ap_positions = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                ap_lengths.append(len(sub_indices))
                for pos, idx in enumerate(sub_indices):
                    ap_membership[idx].append(ap_idx)
                    ap_positions[idx].append((ap_idx, pos))
    return aps, ap_membership, ap_lengths, ap_positions

def _compute_ap_extrema(aps, signs, ap_lengths):
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

def _compute_ap_cumsums(aps, signs):
    """Precompute cumulative sums for all APs."""
    n_aps = len(aps)
    cumsums = []
    for ap in aps:
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        cumsums.append(cum_ext)
    return cumsums

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums, cumsums, pos_in_ap):
    """Update max/min subarray sum for a single AP using precomputed cumsums.
    pos_in_ap is the position of the flipped element in this AP."""
    # Update the cumulative sum from pos_in_ap onwards
    old_val = -signs[ap[pos_in_ap]]  # The element was flipped, so old_val is -current
    # Actually, we need to adjust: the cumsum at position p and beyond changes by 2*old_val
    # Since we flipped signs[elem], the change is -2*signs[elem] = 2*old_val
    delta = -2 * signs[ap[pos_in_ap]]
    
    # Update cumsums[ap_idx] from position pos_in_ap+1 onwards
    cum = cumsums[ap_idx]
    if pos_in_ap + 1 < len(cum):
        cum[pos_in_ap+1:] += delta
    
    # Recompute extrema
    min_prefix = np.minimum.accumulate(cum)
    max_prefix = np.maximum.accumulate(cum)
    
    max_sums[ap_idx] = np.max(cum[1:] - min_prefix[:-1])
    min_sums[ap_idx] = np.min(cum[1:] - max_prefix[:-1])

def _find_worst_ap_and_element(max_sums, min_sums, aps, signs, cumsums, ap_positions):
    """Find the AP with max discrepancy and the element whose flip would most reduce it.
    Returns (ap_idx, element_idx_in_aps, flip_direction) or (None, None, None)."""
    # Find worst AP
    worst_ap_idx = None
    worst_val = 0
    
    for i in range(len(max_sums)):
        if max_sums[i] > worst_val:
            worst_val = max_sums[i]
            worst_ap_idx = i
        if -min_sums[i] > worst_val:
            worst_val = -min_sums[i]
            worst_ap_idx = i
    
    if worst_ap_idx is None:
        return None, None, None
    
    ap = aps[worst_ap_idx]
    if len(ap) == 0:
        return None, None, None
    
    # Try flipping each element in this AP and see which gives best improvement
    best_elem = None
    best_improvement = 0
    
    # For efficiency, just try a few key positions
    positions_to_try = [0, len(ap)//2, len(ap)-1]
    if len(ap) >= 4:
        positions_to_try.append(len(ap)//4)
        positions_to_try.append(3*len(ap)//4)
    
    cum = cumsums[worst_ap_idx].copy()
    
    for pos in positions_to_try:
        elem_idx = ap[pos]
        old_val = signs[elem_idx]
        signs[elem_idx] = -old_val
        
        # Update cumsum from pos onwards
        delta = -2 * signs[elem_idx]
        if pos + 1 < len(cum):
            cum[pos+1:] += delta
        
        # Compute new discrepancy for this AP
        min_prefix = np.minimum.accumulate(cum)
        max_prefix = np.maximum.accumulate(cum)
        new_max = np.max(cum[1:] - min_prefix[:-1])
        new_min = np.min(cum[1:] - max_prefix[:-1])
        new_disc_this_ap = max(abs(new_max), abs(new_min))
        improvement = worst_val - new_disc_this_ap
        
        # Revert
        if pos + 1 < len(cum):
            cum[pos+1:] -= delta
        signs[elem_idx] = old_val
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_elem = elem_idx
    
    return worst_ap_idx, best_elem, best_improvement

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Build AP structure once
    aps, ap_membership, ap_lengths, ap_positions = _build_ap_structure(n)
    
    # Initialize with multiple strategies
    candidates = []
    
    # Thue-Morse
    signs_tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_alt)
    
    # Multiple random seeds
    for seed in range(20):
        rng_c = random.Random(seed)
        candidates.append([rng_c.choice((-1, 1)) for _ in range(n)])
    
    # Greedy-like: start with all +1
    candidates.append([1] * n)
    
    # Evaluate candidates
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 20:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    signs = best_signs[:]
    
    # Compute initial AP extrema and cumulative sums
    max_sums, min_sums = _compute_ap_extrema(aps, signs, ap_lengths)
    cumsums = _compute_ap_cumsums(aps, signs)
    current_disc = _get_max_disc(max_sums, min_sums)
    
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = signs[:]
    
    # Phase 1: Targeted descent - directly target worst APs
    phase1_deadline = deadline - 15
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < phase1_deadline and no_improve_count < max_no_improve:
        # Find worst AP and best element to flip
        worst_ap_idx, best_elem, improvement = _find_worst_ap_and_element(
            max_sums, min_sums, aps, signs, cumsums, ap_positions)
        
        if best_elem is None or improvement <= 0:
            no_improve_count += 1
            # Try random flips to escape
            for _ in range(3):
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx, pos in ap_positions[idx]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
                new_disc = _get_max_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs[:]
                        no_improve_count = 0
                    break
                else:
                    signs[idx] = old_val
                    for ap_idx, pos in ap_positions[idx]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
            continue
        
        # Flip the best element
        old_val = signs[best_elem]
        signs[best_elem] = -old_val
        
        # Update all affected APs
        for ap_idx, pos in ap_positions[best_elem]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        if new_disc < current_disc:
            current_disc = new_disc
            no_improve_count = 0
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
        else:
            # Revert
            signs[best_elem] = old_val
            for ap_idx, pos in ap_positions[best_elem]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
            no_improve_count += 1
    
    # Phase 2: Simulated annealing with incremental updates
    rng_sa = random.Random(123)
    temp = 2.0
    cool_rate = 0.99
    no_improve_sa = 0
    max_no_improve_sa = 50
    
    while time.time() < deadline - 5 and no_improve_sa < max_no_improve_sa:
        idx = rng_sa.randint(0, n - 1)
        old_val = signs[idx]
        new_val = -old_val
        
        signs[idx] = new_val
        
        # Update affected APs
        for ap_idx, pos in ap_positions[idx]:
            _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
        
        new_disc = _get_max_disc(max_sums, min_sums)
        
        delta = new_disc - current_disc
        if delta < 0 or rng_sa.random() < math.exp(-delta / max(temp, 0.01)):
            current_disc = new_disc
            no_improve_sa = 0
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
        else:
            signs[idx] = old_val
            for ap_idx, pos in ap_positions[idx]:
                _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
            no_improve_sa += 1
        
        temp *= cool_rate
    
    # Phase 3: Final local search
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old_val = signs[i]
                signs[i] = -old_val
                for ap_idx, pos in ap_positions[i]:
                    _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
                new_disc = _get_max_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs[:]
                else:
                    signs[i] = old_val
                    for ap_idx, pos in ap_positions[i]:
                        _update_single_ap(aps[ap_idx], signs, ap_idx, max_sums, min_sums, cumsums, pos)
    
    return best_signs
# EVOLVE-BLOCK-END
