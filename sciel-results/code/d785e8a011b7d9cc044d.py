"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s evaluator limit


def _time_ok(margin=2.0):
    """True if we still have time before the deadline (with margin)."""
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _greedy_fill(ow, ov, cap, skip_mask=None):
    """
    Total value of density-order greedy fill of (ow, ov) with starting capacity `cap`.
    If skip_mask is provided (a boolean array same length as ow), those positions are skipped.
    Items that don't fit are skipped (scan continues).
    """
    total = 0.0
    n = len(ow)
    if skip_mask is None:
        for k in range(n):
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
    else:
        for k in range(n):
            if skip_mask[k]:
                continue
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    C = float(remaining_capacity)
    if C <= 0:
        return -1
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    if n == 0:
        return -1

    # Feasible items: weight <= C
    feasible = w <= C
    if not np.any(feasible):
        return -1

    feas_idx = np.where(feasible)[0]
    fw = w[feas_idx]
    fv = v[feas_idx]
    m = len(fw)
    if m == 0:
        return -1

    # Density (guard zero/negative weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / np.where(fw > 0, fw, 1.0), np.inf)
    order = np.argsort(-dens, kind='stable')
    ow = fw[order]
    ov = fv[order]
    # pos_in_order: position in sorted order for each compact index
    pos = np.empty(m, dtype=int)
    pos[order] = np.arange(m)

    # If very few items, just do a simple check
    if m <= 2:
        # Pick the one with higher value that fits
        best = -1
        best_val = -np.inf
        for i in range(m):
            if fw[i] <= C:
                if fv[i] > best_val:
                    best_val = fv[i]
                    best = i
        if best == -1:
            return -1
        return int(feas_idx[best])

    # 1-step lookahead score for each compact candidate
    # score[i] = fv[i] + greedy(ow, ov, skip={pos[i]}, C - fw[i])
    scores1 = np.empty(m)
    for i in range(m):
        cap1 = C - fw[i]
        if cap1 < 0:
            scores1[i] = -np.inf
        else:
            # Create skip mask
            skip_mask = np.zeros(m, dtype=bool)
            skip_mask[pos[i]] = True
            scores1[i] = fv[i] + _greedy_fill(ow, ov, cap1, skip_mask)

    # Check time
    if not _time_ok(margin=5.0):
        best = int(np.argmax(scores1))
        return int(feas_idx[best])

    # Choose candidate set for 2-step
    if m <= 40:
        cand = list(range(m))
    else:
        # top B by 1-step score
        B = min(30, m)
        top = np.argsort(-scores1, kind='stable')[:B]
        cand = [int(t) for t in top]

    # 2-step lookahead for each candidate in cand
    best_score = -np.inf
    best_i = cand[0]
    
    for i in cand:
        cap1 = C - fw[i]
        if cap1 < 0:
            continue
        pi = pos[i]
        base = fv[i]
        
        # Skip mask for item i
        skip_mask_i = np.zeros(m, dtype=bool)
        skip_mask_i[pi] = True
        
        # Best with just item i + greedy
        best2 = base + _greedy_fill(ow, ov, cap1, skip_mask_i)
        
        # Try adding a second item j
        for j in range(m):
            if j == i:
                continue
            wj = fw[j]
            if wj > cap1:
                continue
            cap2 = cap1 - wj
            if cap2 < 0:
                continue
            # Skip mask for items i and j
            skip_mask_ij = np.zeros(m, dtype=bool)
            skip_mask_ij[pi] = True
            skip_mask_ij[pos[j]] = True
            val = base + fv[j] + _greedy_fill(ow, ov, cap2, skip_mask_ij)
            if val > best2:
                best2 = val
        
        if best2 > best_score:
            best_score = best2
            best_i = i
        
        if not _time_ok(margin=2.0):
            break

    return int(feas_idx[best_i])
# EVOLVE-BLOCK-END
