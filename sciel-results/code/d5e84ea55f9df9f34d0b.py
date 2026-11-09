# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

def _validate(points):
    """Exact-integer validity check. Returns (valid, k)."""
    if points.ndim != 2 or points.shape[1] != 11:
        return False, 0
    k = points.shape[0]
    if k < 2:
        return False, 0
    norms_sq = np.sum(points.astype(np.int64) ** 2, axis=1)
    if np.any(norms_sq == 0):
        return False, 0
    max_norm = int(np.max(norms_sq))
    # min pairwise squared distance
    min_dist = None
    for i in range(k):
        for j in range(i + 1, k):
            d = int(np.dot(points[i] - points[j], points[i] - points[j]))
            if min_dist is None or d < min_dist:
                min_dist = d
    if min_dist is None or min_dist < max_norm:
        return False, 0
    return True, k

def kissing_number11() -> np.ndarray:
    """
    Integer kissing configuration in dimension 11.
    Builds a large set of integer points with min pairwise squared distance
    >= max squared norm, maximizing the number of points k.
    Uses an E8 x D3 decomposition with mixed-type vectors, refined greedily.
    """
    start_time = time.time()
    deadline = start_time + 560  # safety margin before 630s kill

    d = 11
    fallback = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    # ---- E8 lattice vectors (norm_sq = 4) ----
    # Type 1: (±2, ±2, 0, ..., 0)
    e8_t1 = []
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(8, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                e8_t1.append(p)
    # Type 2: all coordinates ±1, even number of minus signs
    e8_t2 = []
    for mask in range(256):
        bits = [((mask >> b) & 1) for b in range(8)]
        nminus = sum(bits)
        if nminus % 2 == 0:
            p = np.array([(-1 if bits[b] else 1) for b in range(8)], dtype=np.int64)
            e8_t2.append(p)
    e8_vectors = e8_t1 + e8_t2  # 112 + 128 = 240

    # ---- D3 lattice vectors (norm_sq = 4) ----
    # (±2, ±2, 0) and (±2, 0, ±2) and (0, ±2, ±2)
    d3_vectors = []
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(3, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                d3_vectors.append(p)
    # 3 pairs * 4 = 12

    # ---- Build full 11-dim candidate pool ----
    candidates = []
    # Pure E8: E8 vector in first 8 coords, 0 in last 3
    for v in e8_vectors:
        p = np.zeros(11, dtype=np.int64)
        p[:8] = v
        candidates.append(p)
    # Pure D3: 0 in first 8, D3 vector in last 3
    for v in d3_vectors:
        p = np.zeros(11, dtype=np.int64)
        p[8:] = v
        candidates.append(p)
    # Mixed: non-zero E8 vector in first 8, non-zero D3 vector in last 3
    # (product gives norm_sq = 4 + 4 = 8)
    for a in e8_vectors:
        for b in d3_vectors:
            p = np.zeros(11, dtype=np.int64)
            p[:8] = a
            p[8:] = b
            candidates.append(p)

    # Deduplicate
    seen = set()
    uniq = []
    for p in candidates:
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    candidates = uniq
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)

    # ---- Greedy selection with vectorized distance check ----
    # We want min pairwise dist_sq >= max norm_sq.
    # Strategy: pick a target norm bound. We'll greedily add points,
    # tracking current max_norm and requiring new points to have
    # dist_sq >= max(current_max_norm, new_norm) to all existing points.

    # Precompute norms
    cand_norms = np.sum(cand_arr ** 2, axis=1)

    # We'll do a greedy: start empty, repeatedly add the candidate that
    # is "furthest" / most compatible. To keep it fast, use a heuristic:
    # add candidates in an order that tends to spread out.
    # Use a simple greedy: iterate, add if compatible with all current.

    selected_idx = []
    selected_arr = []
    current_max_norm = 0

    # Order candidates: prefer higher norm first? Actually we want to be able
    # to add many. Let's sort by norm descending so we establish a high bar
    # early, but that may block many. Instead, sort by norm ascending to build
    # a low-norm core then add.
    # Better: just do a greedy that tries all candidates in a fixed order.

    # To maximize count, we want many points. The constraint is min_dist >= max_norm.
    # If all selected points have the SAME norm N, then we need min_dist >= N.
    # Mixed-type vectors have norm 8; pure E8/D3 have norm 4.
    # If we include norm-8 points, max_norm = 8, so all pairs need dist >= 8.
    # If we only use norm-4 points, max_norm = 4, need dist >= 4. Easier!

    # So two strategies:
    #  A) Use only norm-4 points (pure E8 + pure D3): max_norm=4, need dist>=4.
    #  B) Use norm-8 points (mixed): max_norm=8, need dist>=8.

    # Strategy A likely yields more points. Let's do A first, then try to
    # add norm-8 points if compatible (which would raise max_norm to 8).

    # --- Strategy A: norm-4 points only ---
    norm4_mask = cand_norms == 4
    norm4_indices = np.where(norm4_mask)[0]
    norm4_arr = cand_arr[norm4_indices]

    sel_A = []
    sel_A_arr = []
    for idx in norm4_indices:
        if time.time() > deadline - 30:
            break
        v = cand_arr[idx]
        # check against all selected
        if len(sel_A_arr) == 0:
            sel_A.append(idx)
            sel_A_arr.append(v)
            continue
        arr = np.array(sel_A_arr)
        diffs = arr - v
        dists = np.sum(diffs ** 2, axis=1)
        if np.min(dists) >= 4:
            sel_A.append(idx)
            sel_A_arr.append(v)

    best_k = len(sel_A)
    best_points = np.array(sel_A_arr, dtype=np.int64) if sel_A_arr else fallback

    # --- Try Strategy B: add norm-8 points to a norm-4 core, or pure norm-8 ---
    # If we add norm-8 points, max_norm becomes 8, so ALL pairs (including
    # norm-4 vs norm-4) must have dist >= 8. That's very restrictive.
    # Better: build a set where ALL points have norm 8 (only mixed), need dist>=8.
    # Or: a set with max norm 8 where we verify the full constraint.

    # Let's try: greedy over ALL candidates but with the dynamic max_norm rule.
    # Start with a strong core, then add.
    # Use a different order: interleave.

    # Actually, let's try building with the rule: maintain max_norm, require
    # dist >= max_norm for all pairs. Try to maximize count.
    # We'll do a greedy that processes candidates and adds if compatible.

    # Process in an order: to get many, process lower-norm first.
    order = np.argsort(cand_norms)  # ascending norm
    sel_B = []
    sel_B_arr = []
    max_norm_B = 0
    for idx in order:
        if time.time() > deadline - 30:
            break
        v = cand_arr[idx]
        vn = int(cand_norms[idx])
        if vn == 0:
            continue
        new_max = max(max_norm_B, vn)
        if len(sel_B_arr) == 0:
            sel_B.append(idx)
            sel_B_arr.append(v)
            max_norm_B = vn
            continue
        arr = np.array(sel_B_arr)
        diffs = arr - v
        dists = np.sum(diffs ** 2, axis=1)
        if np.min(dists) >= new_max:
            sel_B.append(idx)
            sel_B_arr.append(v)
            max_norm_B = new_max

    k_B = len(sel_B)
    if k_B > best_k:
        best_k = k_B
        best_points = np.array(sel_B_arr, dtype=np.int64)

    # ---- Final validation and return ----
    if best_points.ndim == 1:
        best_points = best_points.reshape(1, -1)
    valid, k_final = _validate(best_points)
    if not valid or k_final < 2:
        return fallback

    best_points = np.around(best_points).astype(np.int64)
    return best_points

# EVOLVE-BLOCK-END
