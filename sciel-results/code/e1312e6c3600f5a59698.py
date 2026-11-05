import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256

    # Build all APs with length >= 2
    aps = []
    ap_membership = [[] for _ in range(n)]
    ap_lengths = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                ap_lengths.append(len(sub_indices))
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)

    n_aps = len(aps)

    def _update_ap_fast(ap_idx, signs):
        """Compute max and min subarray sum for AP ap_idx using Python loop (Kadane)."""
        ap = aps[ap_idx]
        max_end = 0
        max_sum = 0
        min_end = 0
        min_sum = 0
        for idx in ap:
            v = signs[idx]
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
        return max_sum, -min_sum

    def _update_ap_numpy(ap_idx, signs_np):
        """Compute max and min subarray sum for AP ap_idx using NumPy."""
        ap = aps[ap_idx]
        vals = signs_np[ap]
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sub = int(np.max(cum_ext[1:] - min_prefix[:-1]))
        min_sub = int(np.min(cum_ext[1:] - max_prefix[:-1]))
        return max_sub, min_sub

    def _compute_all(signs, signs_np):
        """Compute max/min subarray sums for all APs."""
        max_sums = np.zeros(n_aps, dtype=np.int32)
        min_sums = np.zeros(n_aps, dtype=np.int32)
        for i in range(n_aps):
            if ap_lengths[i] > 20:
                max_sums[i], min_sums[i] = _update_ap_numpy(i, signs_np)
            else:
                max_sums[i], min_sums[i] = _update_ap_fast(i, signs)
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        if n_aps == 0:
            return 1
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _eval_full(signs):
        signs_np = np.array(signs, dtype=np.int8)
        max_sums, min_sums = _compute_all(signs, signs_np)
        return _get_disc(max_sums, min_sums)

    def _find_worst_ap_indices(max_sums, min_sums, k=5):
        abs_disc = np.maximum(max_sums, -min_sums)
        if len(abs_disc) == 0:
            return []
        k = min(k, len(abs_disc))
        worst_idx = np.argsort(-abs_disc)[:k]
        return list(worst_idx)

    def _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums):
        ap = aps[ap_idx]
        current_max = max_sums[ap_idx]
        current_min = min_sums[ap_idx]
        current_worst = max(current_max, -current_min)
        
        best_flip_pos = -1
        best_new_worst = current_worst + 1
        
        for pos in ap:
            old_val = signs[pos]
            signs[pos] = -old_val
            
            new_max, new_min = _update_ap_fast(ap_idx, signs)
            new_worst = max(new_max, -new_min)
            
            signs[pos] = old_val
            
            if new_worst < best_new_worst:
                best_new_worst = new_worst
                best_flip_pos = pos
        
        return best_flip_pos, best_new_worst

    def _find_best_multi_flip(ap_idx, signs, max_sums, min_sums, k_flips=2):
        """Try flipping up to k_flips positions in the AP to reduce its discrepancy."""
        ap = aps[ap_idx]
        current_max = max_sums[ap_idx]
        current_min = min_sums[ap_idx]
        current_worst = max(current_max, -current_min)
        
        if len(ap) < 2:
            return [], current_worst + 1
        
        # For 2-flip, try pairs
        if k_flips == 2 and len(ap) <= 20:
            best_flips = []
            best_new_worst = current_worst + 1
            for i in range(len(ap)):
                for j in range(i+1, len(ap)):
                    p1, p2 = ap[i], ap[j]
                    old1, old2 = signs[p1], signs[p2]
                    signs[p1] = -old1
                    signs[p2] = -old2
                    new_max, new_min = _update_ap_fast(ap_idx, signs)
                    new_worst = max(new_max, -new_min)
                    signs[p1] = old1
                    signs[p2] = old2
                    if new_worst < best_new_worst:
                        best_new_worst = new_worst
                        best_flips = [p1, p2]
            return best_flips, best_new_worst
        else:
            # Fall back to single flip
            best_flip_pos, best_new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
            if best_flip_pos >= 0:
                return [best_flip_pos], best_new_worst
            return [], current_worst + 1

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # All +1
    candidates.append([1] * n)

    # Block pattern
    direct = [0] * n
    for i in range(n):
        block = (i // 4) % 2
        pos = i % 4
        if block == 0:
            direct[i] = 1 if pos < 2 else -1
        else:
            direct[i] = -1 if pos < 2 else 1
    candidates.append(direct)

    # Van der Corput permuted Thue-Morse
    def radical_inverse(i, base=2):
        result = 0.0
        f = 1.0 / base
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    perm = sorted(range(n), key=lambda i: radical_inverse(i))
    tm_perm = [0] * n
    for orig, new in enumerate(perm):
        tm_perm[new] = 1 if bin(orig).count('1') % 2 == 0 else -1
    candidates.append(tm_perm)

    # Multiple random
    for s in range(20):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # Random with bias toward balance
    for s in range(10):
        r = random.Random(100 + s)
        signs = [r.choice((-1, 1)) for _ in range(n)]
        while sum(signs) > 0:
            idx = r.randint(0, n - 1)
            if signs[idx] == 1:
                signs[idx] = -1
                break
        candidates.append(signs)

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = candidates[0][:]
    best_disc = float('inf')

    for cand in candidates:
        if time.time() > deadline - 20:
            break
        d = _eval_full(cand)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 3: SA with multi-point moves ---
    signs = best_signs[:]
    signs_np = np.array(signs, dtype=np.int8)
    max_sums, min_sums = _compute_all(signs, signs_np)
    current_disc = _get_disc(max_sums, min_sums)

    rng = random.Random(12345)
    no_improve = 0
    max_no_improve = 400
    temp = 4.0
    cool = 0.997
    targeted_count = 0

    while time.time() < deadline - 10 and no_improve < max_no_improve:
        # Alternate between targeted and random moves
        use_targeted = (targeted_count % 2 == 0) or (current_disc > 4)
        
        if use_targeted and n_aps > 0:
            # Find worst APs and try to fix them with multi-flip
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
            flip_made = False
            
            for ap_idx in worst_aps:
                # Try 2-flip first for short APs, else single flip
                if ap_lengths[ap_idx] <= 15:
                    flip_positions, new_worst = _find_best_multi_flip(ap_idx, signs, max_sums, min_sums, k_flips=2)
                else:
                    flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                    if flip_pos >= 0:
                        flip_positions = [flip_pos]
                    else:
                        flip_positions = []
                
                if flip_positions:
                    # Make the flips
                    old_vals = [signs[p] for p in flip_positions]
                    for p in flip_positions:
                        signs[p] = -signs[p]
                        signs_np[p] = -signs_np[p]
                    
                    # Update all APs containing any flipped position
                    affected_aps = set()
                    for p in flip_positions:
                        for a_idx in ap_membership[p]:
                            affected_aps.add(a_idx)
                    
                    for a_idx in affected_aps:
                        if ap_lengths[a_idx] > 20:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                        else:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                    
                    new_disc = _get_disc(max_sums, min_sums)
                    
                    if new_disc < current_disc:
                        current_disc = new_disc
                        no_improve = 0
                        targeted_count += 1
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        # Revert
                        for p, ov in zip(flip_positions, old_vals):
                            signs[p] = ov
                            signs_np[p] = ov
                        for a_idx in affected_aps:
                            if ap_lengths[a_idx] > 20:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                            else:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                        no_improve += 1
                    
                    flip_made = True
                    break
            
            if not flip_made:
                # No targeted improvement, do random single flip
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                signs_np[idx] = -signs_np[idx]
                
                for ap_idx in ap_membership[idx]:
                    if ap_lengths[ap_idx] > 20:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                    else:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                
                new_disc = _get_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        if rng.random() < math.exp(-delta / temp):
                            current_disc = new_disc
                            no_improve = 1
                        else:
                            signs[idx] = old_val
                            signs_np[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                if ap_lengths[ap_idx] > 20:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                else:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                            no_improve += 1
                    else:
                        signs[idx] = old_val
                        signs_np[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            if ap_lengths[ap_idx] > 20:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                            else:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                        no_improve += 1
        else:
            # Random move: single or double flip
            if rng.random() < 0.3:
                # Double flip
                idx1 = rng.randint(0, n - 1)
                idx2 = rng.randint(0, n - 1)
                while idx2 == idx1:
                    idx2 = rng.randint(0, n - 1)
                old1, old2 = signs[idx1], signs[idx2]
                signs[idx1] = -old1
                signs[idx2] = -old2
                signs_np[idx1] = -signs_np[idx1]
                signs_np[idx2] = -signs_np[idx2]
                
                affected_aps = set(ap_membership[idx1]) | set(ap_membership[idx2])
                for ap_idx in affected_aps:
                    if ap_lengths[ap_idx] > 20:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                    else:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                
                new_disc = _get_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        if rng.random() < math.exp(-delta / temp):
                            current_disc = new_disc
                            no_improve = 1
                        else:
                            signs[idx1] = old1
                            signs[idx2] = old2
                            signs_np[idx1] = old1
                            signs_np[idx2] = old2
                            for ap_idx in affected_aps:
                                if ap_lengths[ap_idx] > 20:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                else:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                            no_improve += 1
                    else:
                        signs[idx1] = old1
                        signs[idx2] = old2
                        signs_np[idx1] = old1
                        signs_np[idx2] = old2
                        for ap_idx in affected_aps:
                            if ap_lengths[ap_idx] > 20:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                            else:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                        no_improve += 1
            else:
                # Single flip
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                signs_np[idx] = -signs_np[idx]
                
                for ap_idx in ap_membership[idx]:
                    if ap_lengths[ap_idx] > 20:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                    else:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                
                new_disc = _get_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        if rng.random() < math.exp(-delta / temp):
                            current_disc = new_disc
                            no_improve = 1
                        else:
                            signs[idx] = old_val
                            signs_np[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                if ap_lengths[ap_idx] > 20:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                else:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                            no_improve += 1
                    else:
                        signs[idx] = old_val
                        signs_np[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            if ap_lengths[ap_idx] > 20:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                            else:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                        no_improve += 1

        temp *= cool

    # --- Phase 4: Strict targeted hill climbing with multi-flip ---
    if time.time() < deadline - 5:
        signs = best_signs[:]
        signs_np = np.array(signs, dtype=np.int8)
        max_sums, min_sums = _compute_all(signs, signs_np)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # First try targeted: fix worst APs with multi-flip
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=5)
            for ap_idx in worst_aps:
                if ap_lengths[ap_idx] <= 15:
                    flip_positions, new_worst = _find_best_multi_flip(ap_idx, signs, max_sums, min_sums, k_flips=2)
                else:
                    flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                    if flip_pos >= 0:
                        flip_positions = [flip_pos]
                    else:
                        flip_positions = []
                
                if flip_positions:
                    old_vals = [signs[p] for p in flip_positions]
                    for p in flip_positions:
                        signs[p] = -signs[p]
                        signs_np[p] = -signs_np[p]
                    
                    affected_aps = set()
                    for p in flip_positions:
                        for a_idx in ap_membership[p]:
                            affected_aps.add(a_idx)
                    
                    for a_idx in affected_aps:
                        if ap_lengths[a_idx] > 20:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                        else:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                    
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        improved = True
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        for p, ov in zip(flip_positions, old_vals):
                            signs[p] = ov
                            signs_np[p] = ov
                        for a_idx in affected_aps:
                            if ap_lengths[a_idx] > 20:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                            else:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
            
            # Then try standard single-flip hill climb
            for idx in range(n):
                if not improved and time.time() > deadline - 5:
                    break
                old_val = signs[idx]
                signs[idx] = -old_val
                signs_np[idx] = -signs_np[idx]
                for ap_idx in ap_membership[idx]:
                    if ap_lengths[ap_idx] > 20:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                    else:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    signs_np[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        if ap_lengths[ap_idx] > 20:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                        else:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)

    # --- Phase 5: Additional restarts with multi-point ---
    if time.time() < deadline - 10:
        for restart in range(4):
            if time.time() > deadline - 5:
                break
            signs = best_signs[:]
            signs_np = np.array(signs, dtype=np.int8)
            rng2 = random.Random(9999 + restart)
            # Perturb with 5-10 random flips
            n_perturb = rng2.randint(5, 10)
            for _ in range(n_perturb):
                idx = rng2.randint(0, n - 1)
                signs[idx] = -signs[idx]
                signs_np[idx] = -signs_np[idx]

            max_sums, min_sums = _compute_all(signs, signs_np)
            current_disc = _get_disc(max_sums, min_sums)

            temp = 3.0
            cool = 0.996
            no_improve = 0
            max_no_improve = 200
            t_count = 0

            while time.time() < deadline - 5 and no_improve < max_no_improve:
                use_targeted = (t_count % 2 == 0) or (current_disc > 4)
                
                if use_targeted and n_aps > 0:
                    worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
                    flip_made = False
                    for ap_idx in worst_aps:
                        if ap_lengths[ap_idx] <= 15:
                            flip_positions, new_worst = _find_best_multi_flip(ap_idx, signs, max_sums, min_sums, k_flips=2)
                        else:
                            flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                            if flip_pos >= 0:
                                flip_positions = [flip_pos]
                            else:
                                flip_positions = []
                        
                        if flip_positions:
                            old_vals = [signs[p] for p in flip_positions]
                            for p in flip_positions:
                                signs[p] = -signs[p]
                                signs_np[p] = -signs_np[p]
                            
                            affected_aps = set()
                            for p in flip_positions:
                                for a_idx in ap_membership[p]:
                                    affected_aps.add(a_idx)
                            
                            for a_idx in affected_aps:
                                if ap_lengths[a_idx] > 20:
                                    max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                                else:
                                    max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                            
                            new_disc = _get_disc(max_sums, min_sums)
                            if new_disc < current_disc:
                                current_disc = new_disc
                                no_improve = 0
                                t_count += 1
                                if current_disc < best_disc:
                                    best_disc = current_disc
                                    best_signs = signs[:]
                            else:
                                for p, ov in zip(flip_positions, old_vals):
                                    signs[p] = ov
                                    signs_np[p] = ov
                                for a_idx in affected_aps:
                                    if ap_lengths[a_idx] > 20:
                                        max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                                    else:
                                        max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                                no_improve += 1
                            flip_made = True
                            break
                    if not flip_made:
                        idx = rng2.randint(0, n - 1)
                        old_val = signs[idx]
                        signs[idx] = -old_val
                        signs_np[idx] = -signs_np[idx]
                        for ap_idx in ap_membership[idx]:
                            if ap_lengths[ap_idx] > 20:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                            else:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                        new_disc = _get_disc(max_sums, min_sums)
                        if new_disc < current_disc:
                            current_disc = new_disc
                            no_improve = 0
                            if current_disc < best_disc:
                                best_disc = current_disc
                                best_signs = signs[:]
                        else:
                            delta = new_disc - current_disc
                            if delta > 0 and temp > 0.1:
                                if rng2.random() < math.exp(-delta / temp):
                                    current_disc = new_disc
                                    no_improve = 1
                                else:
                                    signs[idx] = old_val
                                    signs_np[idx] = old_val
                                    for ap_idx in ap_membership[idx]:
                                        if ap_lengths[ap_idx] > 20:
                                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                        else:
                                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                                    no_improve += 1
                            else:
                                signs[idx] = old_val
                                signs_np[idx] = old_val
                                for ap_idx in ap_membership[idx]:
                                    if ap_lengths[ap_idx] > 20:
                                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                    else:
                                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                                no_improve += 1
                else:
                    idx = rng2.randint(0, n - 1)
                    old_val = signs[idx]
                    signs[idx] = -old_val
                    signs_np[idx] = -signs_np[idx]
                    for ap_idx in ap_membership[idx]:
                        if ap_lengths[ap_idx] > 20:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                        else:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        no_improve = 0
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        delta = new_disc - current_disc
                        if delta > 0 and temp > 0.1:
                            if rng2.random() < math.exp(-delta / temp):
                                current_disc = new_disc
                                no_improve = 1
                            else:
                                signs[idx] = old_val
                                signs_np[idx] = old_val
                                for ap_idx in ap_membership[idx]:
                                    if ap_lengths[ap_idx] > 20:
                                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                    else:
                                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                                no_improve += 1
                        else:
                            signs[idx] = old_val
                            signs_np[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                if ap_lengths[ap_idx] > 20:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                                else:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                            no_improve += 1
                temp *= cool

    # --- Phase 6: Final polish ---
    if time.time() < deadline - 2:
        signs = best_signs[:]
        signs_np = np.array(signs, dtype=np.int8)
        max_sums, min_sums = _compute_all(signs, signs_np)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
            for ap_idx in worst_aps:
                if ap_lengths[ap_idx] <= 15:
                    flip_positions, new_worst = _find_best_multi_flip(ap_idx, signs, max_sums, min_sums, k_flips=2)
                else:
                    flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                    if flip_pos >= 0:
                        flip_positions = [flip_pos]
                    else:
                        flip_positions = []
                
                if flip_positions:
                    old_vals = [signs[p] for p in flip_positions]
                    for p in flip_positions:
                        signs[p] = -signs[p]
                        signs_np[p] = -signs_np[p]
                    
                    affected_aps = set()
                    for p in flip_positions:
                        for a_idx in ap_membership[p]:
                            affected_aps.add(a_idx)
                    
                    for a_idx in affected_aps:
                        if ap_lengths[a_idx] > 20:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                        else:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
                    
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        improved = True
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        for p, ov in zip(flip_positions, old_vals):
                            signs[p] = ov
                            signs_np[p] = ov
                        for a_idx in affected_aps:
                            if ap_lengths[a_idx] > 20:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_numpy(a_idx, signs_np)
                            else:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap_fast(a_idx, signs)
            for idx in range(n):
                if not improved and time.time() > deadline - 2:
                    break
                old_val = signs[idx]
                signs[idx] = -old_val
                signs_np[idx] = -signs_np[idx]
                for ap_idx in ap_membership[idx]:
                    if ap_lengths[ap_idx] > 20:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                    else:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    signs_np[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        if ap_lengths[ap_idx] > 20:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_numpy(ap_idx, signs_np)
                        else:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap_fast(ap_idx, signs)

    return best_signs
