import time
import random
import numpy as np
import math

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
    """Compute max and min subarray sums for each AP.
    Returns arrays max_sums and min_sums where:
    max_sums[i] = max subarray sum in AP i
    min_sums[i] = min subarray sum in AP i (will be <= 0)
    """
    n_aps = len(aps)
    max_sums = np.zeros(n_aps, dtype=np.int32)
    min_sums = np.zeros(n_aps, dtype=np.int32)
    
    for ap_idx, ap in enumerate(aps):
        vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        
        # Efficiently compute max/min subarray sums using prefix min/max
        # max_subarray = max_{j} (prefix[j] - min_{i<j} prefix[i])
        # min_subarray = min_{j} (prefix[j] - max_{i<j} prefix[i])
        
        # We can do this in a single pass or with vectorized ops
        # Vectorized:
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        
        # For max sum: max(cum_ext - min_prefix) but min_prefix[i] is min up to i.
        # Actually, standard Kadane's or prefix trick:
        # max_sum = max( cum[j] - min_{i<j} cum[i] )
        # The array `min_prefix` contains min_{i<=j} cum[i].
        # So we want max( cum[j] - min_prefix[j-1] ) for j=1..N?
        # Let's just do the loop for clarity and correctness in this small context, 
        # or use the property that max_subarray = max(cum) - min(cum) is WRONG (that's range).
        # Wait, max subarray sum is indeed max_j (S_j - min_{i<j} S_i).
        
        # Let's re-verify the vectorized logic.
        # min_prefix[j] = min(S_0, ..., S_j)
        # We need min(S_0, ..., S_{j-1}) to calculate S_j - min.
        # So shift min_prefix by 1.
        
        min_prefix_shifted = min_prefix[:-1]
        max_prefix_shifted = max_prefix[:-1]
        
        diffs_max = cum_ext[1:] - min_prefix_shifted
        diffs_min = cum_ext[1:] - max_prefix_shifted
        
        max_sums[ap_idx] = np.max(diffs_max) if len(diffs_max) > 0 else 0
        min_sums[ap_idx] = np.min(diffs_min) if len(diffs_min) > 0 else 0
        
    return max_sums, min_sums

def _update_single_ap(ap, signs, ap_idx, max_sums, min_sums):
    """Update max/min subarray sum for a single AP in-place."""
    vals = np.array([signs[idx] for idx in ap], dtype=np.int32)
    cum = np.cumsum(vals)
    cum_ext = np.concatenate(([0], cum))
    
    min_prefix = np.minimum.accumulate(cum_ext)
    max_prefix = np.maximum.accumulate(cum_ext)
    
    min_prefix_shifted = min_prefix[:-1]
    max_prefix_shifted = max_prefix[:-1]
    
    diffs_max = cum_ext[1:] - min_prefix_shifted
    diffs_min = cum_ext[1:] - max_prefix_shifted
    
    max_sums[ap_idx] = np.max(diffs_max) if len(diffs_max) > 0 else 0
    min_sums[ap_idx] = np.min(diffs_min) if len(diffs_min) > 0 else 0

def _get_max_disc(max_sums, min_sums):
    """Get the overall max discrepancy from per-AP extrema."""
    if len(max_sums) == 0:
        return 1
    max_pos = int(np.max(max_sums))
    min_neg = int(-np.min(min_sums))
    return max(max_pos, min_neg, 1)

def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]

def _rademacher_orthogonal(n):
    """Generate sign sequence using Walsh-Rademacher functions variant."""
    signs = []
    for i in range(n):
        x = i
        cnt = 0
        while x > 0:
            cnt += x & 1
            x >>= 1
        y = i ^ (i >> 3)
        cnt2 = 0
        while y > 0:
            cnt2 += y & 1
            y >>= 1
        signs.append(1 if (cnt + cnt2) % 2 == 0 else -1)
    return signs

def _local_search_fast(signs, aps, ap_membership, deadline, rng, n=256, initial_temp=1.0):
    """Simulated Annealing with fast incremental evaluation."""
    current_signs = signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, current_signs)
    current_val = _get_max_disc(max_sums, min_sums)
    best_signs = current_signs[:]
    best_val = current_val

    temp = initial_temp
    cool_rate = 0.9995
    stall_count = 0
    max_stall = 2000

    while time.time() < deadline:
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val

        # Update affected APs
        for ap_idx in ap_membership[idx]:
            _update_single_ap(aps[ap_idx], current_signs, ap_idx, max_sums, min_sums)
        
        new_val = _get_max_disc(max_sums, min_sums)
        delta = new_val - current_val

        if delta < 0 or (delta == 0 and rng.random() < 0.5) or \
           (delta > 0 and rng.random() < math.exp(-delta / max(temp, 0.01))):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs[:]
                stall_count = 0
            else:
                stall_count += 1
        else:
            current_signs[idx] = old_val
            for ap_idx in ap_membership[idx]:
                _update_single_ap(aps[ap_idx], current_signs, ap_idx, max_sums, min_sums)
            stall_count += 1

        temp *= cool_rate

        # Restart temperature if stuck
        if stall_count > max_stall:
            temp = max(temp, 1.5)
            stall_count = 0

        if best_val == 1:
            break

    return best_signs, best_val

def _greedy_repair_fast(signs, aps, ap_membership, deadline, n=256):
    """Greedy repair using fast evaluation."""
    current_signs = signs[:]
    max_sums, min_sums = _compute_ap_extrema(aps, current_signs)
    current_val = _get_max_disc(max_sums, min_sums)
    best_signs = current_signs[:]
    best_val = current_val

    for _ in range(20): # Limit rounds to ensure time
        if time.time() > deadline - 1:
            break

        # Find worst AP
        # max_sums contains max subarray sums, min_sums contains min subarray sums
        # We want the AP with max(max_sums, -min_sums)
        worst_idx = -1
        worst_disc = -1
        
        # Vectorized find of worst AP
        # Disc for AP i is max(max_sums[i], -min_sums[i])
        disc_array = np.maximum(max_sums, -min_sums)
        worst_idx = int(np.argmax(disc_array))
        worst_disc = int(disc_array[worst_idx])
        
        if worst_disc <= best_val:
            break
            
        # Get the indices of the worst AP
        ap_indices = aps[worst_idx]
        
        # Find the specific subarray causing the discrepancy
        # We need to identify which signs to flip.
        # Simple heuristic: Flip the sign at the position of the peak in the prefix sum.
        # Or just try flipping each element in the AP one by one and see if it reduces global max.
        # Since updating is fast, we can try a few.
        
        improved = False
        # Try flipping elements in the worst AP
        # Sort by impact? Just iterate.
        for idx in ap_indices:
            if time.time() > deadline - 1:
                break
            old_val = current_signs[idx]
            current_signs[idx] = -old_val
            
            for ap_i in ap_membership[idx]:
                _update_single_ap(aps[ap_i], current_signs, ap_i, max_sums, min_sums)
            
            new_val = _get_max_disc(max_sums, min_sums)
            
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs[:]
                improved = True
            else:
                # Revert
                current_signs[idx] = old_val
                for ap_i in ap_membership[idx]:
                    _update_single_ap(aps[ap_i], current_signs, ap_i, max_sums, min_sums)
        
        if not improved:
            break

    return best_signs, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    n = 256
    rng = random.Random(42)

    # Build structure once
    aps, ap_membership = _build_ap_structure(n)

    # Phase 1: Generate multiple initial candidates
    candidates = []
    candidates.append(_thue_morse(n))
    candidates.append(_rademacher_orthogonal(n))
    
    for seed in range(5):
        rng2 = random.Random(seed * 1000 + 7)
        candidates.append([rng2.choice((-1, 1)) for _ in range(n)])

    # Evaluate candidates
    scored = []
    for c in candidates:
        max_sums, min_sums = _compute_ap_extrema(aps, c)
        v = _get_max_disc(max_sums, min_sums)
        scored.append((v, c))
    scored.sort(key=lambda x: x[0])

    best_signs = scored[0][1][:]
    best_val = scored[0][0]

    # Phase 2: Local Search on best candidates
    # We have ~50s left. Allocate time to top 3 candidates.
    num_ls = min(3, len(scored))
    time_per_ls = (deadline - 10 - time.time()) / max(num_ls, 1)
    
    for i in range(num_ls):
        if time.time() > deadline - 10:
            break
        ls_deadline = min(deadline - 10, time.time() + time_per_ls)
        ls_signs, ls_val = _local_search_fast(scored[i][1], aps, ap_membership, ls_deadline, rng, n)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Greedy Repair
    if time.time() < deadline - 5 and best_val > 1:
        gr_signs, gr_val = _greedy_repair_fast(best_signs, aps, ap_membership, deadline, n)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    # Phase 4: ILS (Iterated Local Search) using remaining time
    if time.time() < deadline - 2 and best_val > 1:
        current_best = best_signs[:]
        current_best_val = best_val
        perturbation_strength = 5
        
        while time.time() < deadline - 2:
            # Perturb
            perturbed = current_best[:]
            for _ in range(perturbation_strength):
                idx = rng.randint(0, n - 1)
                perturbed[idx] = -perturbed[idx]
            
            # Local Search on perturbed
            ls_deadline = min(deadline - 2, time.time() + 5) # 5s per ILS iteration
            ls_signs, ls_val = _local_search_fast(perturbed, aps, ap_membership, ls_deadline, rng, n, initial_temp=0.5)
            
            if ls_val < current_best_val:
                current_best_val = ls_val
                current_best = ls_signs
                best_val = ls_val
                best_signs = ls_signs
            else:
                # If no improvement, increase perturbation slightly or keep same
                pass
                
    return best_signs

if __name__ == "__main__":
    # Simple self-test
    signs = construct()
    print(f"Constructed {len(signs)} signs")
    # Verify length and values
    assert len(signs) == 256
    assert all(s in [-1, 1] for s in signs)
    
    # Quick check discrepancy
    aps, _ = _build_ap_structure(256)
    ms, mins = _compute_ap_extrema(aps, signs)
    print(f"Discrepancy: {_get_max_disc(ms, mins)}")
