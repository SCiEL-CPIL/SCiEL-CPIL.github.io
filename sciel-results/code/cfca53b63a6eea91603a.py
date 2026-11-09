import numpy as np
import time
import random

def _validate(pts):
    pts = np.asarray(pts, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 11:
        return False
    k = pts.shape[0]
    if k < 2:
        return False
    # no point is the zero vector
    if np.any((pts == 0).all(axis=1)):
        return False
    norms = np.einsum('ij,ij->i', pts, pts)
    maxn = int(norms.max())
    for i in range(k):
        for j in range(i+1, k):
            d = pts[i] - pts[j]
            dsq = int(np.dot(d, d))
            if dsq < maxn:
                return False
    return True

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 540

    # Construct E8 roots scaled by 2:
    # Type A: ±2e_i ± 2e_j for i<j, 4 sign combos -> 28*4 = 112
    # Type B: (±1, ±1, ..., ±1) with even number of minuses -> 128
    e8_roots = []
    # Type A
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_roots.append(v)
    # Type B: all sign vectors with even number of -1s
    for signs in range(256):
        s = [(signs >> i) & 1 for i in range(8)]
        # convert 0->+1, 1->-1
        v = np.array([1 if s[i]==0 else -1 for i in range(8)], dtype=np.int64)
        # count minuses
        if np.sum(v == -1) % 2 == 0:
            e8_roots.append(v)
    
    e8_roots = np.array(e8_roots)  # (240, 8)

    # Construct A2 roots scaled by 2:
    # A2 roots: ±(e_i - e_j) for i<j in {0,1,2} -> 3 pairs * 2 directions = 6
    # Scaled by 2: ±2(e_i - e_j)
    a2_roots = []
    for i in range(3):
        for j in range(i+1, 3):
            v = np.zeros(3, dtype=np.int64)
            v[i] = 2
            v[j] = -2
            a2_roots.append(v)
            a2_roots.append(-v)
    a2_roots = np.array(a2_roots)  # (6, 3)

    # Embed in Z^11:
    # P1: pad E8 roots with 3 zeros -> (240, 11)
    P1 = np.zeros((240, 11), dtype=np.int64)
    P1[:, :8] = e8_roots

    # P2: pad A2 roots with 8 zeros -> (6, 11)
    P2 = np.zeros((6, 11), dtype=np.int64)
    P2[:, 8:11] = a2_roots

    # Also consider just E8 embedded (without A2) as baseline
    pts_e8_only = P1.copy()
    
    # Combined
    pts_combined = np.vstack([P1, P2])

    # Now let's think about whether we can add more points.
    # The combined set has 246 points. All have norm^2 = 8 (for E8 type A: 4+4=8, type B: 1*8=8)
    # and A2 type: 4+4=8. So max norm = 8.
    # Min distance: within E8, min dist^2 = 8 (between distinct roots).
    # Between E8 and A2: E8 has zeros in last 3, A2 has zeros in first 8, so dist^2 = 8+8=16 >= 8.
    # So the combined set is valid with min dist^2 = 8 >= max norm^2 = 8.
    
    # Can we add more A2-like points? What about D3 = A3?
    # D3 roots: ±e_i ± e_j for i<j in {0,1,2} -> 3 pairs * 4 signs = 12
    # Scaled by 2: 12 vectors, each with norm^2 = 4+4=8.
    d3_roots = []
    for i in range(3):
        for j in range(i+1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    d3_roots.append(v)
    d3_roots = np.array(d3_roots)  # (12, 3)
    
    P2_d3 = np.zeros((12, 11), dtype=np.int64)
    P2_d3[:, 8:11] = d3_roots

    pts_combined_d3 = np.vstack([P1, P2_d3])  # 240 + 12 = 252 points

    # Validate
    if _validate(pts_combined_d3):
        best = pts_combined_d3
    elif _validate(pts_combined):
        best = pts_combined
    else:
        best = pts_e8_only

    # Try to add more points via greedy search
    # Candidate pool: all integer vectors with norm^2 = 8 in Z^11
    # These are:
    # - Type (±2, ±2, 0,...): choose 2 positions from 11, 4 signs -> C(11,2)*4 = 55*4 = 220
    # - Type (±1, ±1, ±1, ±1, 0,...): choose 4 positions from 11, 16 signs -> C(11,4)*16 = 330*16 = 5280
    # Total: 5500 candidates
    # But we need min dist^2 >= 8. Two vectors with norm^2=8 have dist^2 = 16 - 2*dot.
    # dist^2 >= 8 iff dot <= 4.
    
    from itertools import combinations
    
    candidates = []
    # Type 1: two ±2's
    for i in range(11):
        for j in range(i+1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    candidates.append(v)
    # Type 2: four ±1's
    for combo in combinations(range(11), 4):
        for signs in range(16):
            v = np.zeros(11, dtype=np.int64)
            for k, idx in enumerate(combo):
                v[idx] = 1 if (signs >> k) & 1 == 0 else -1
            candidates.append(v)
    
    candidates = np.array(candidates)  # (5500, 11)
    
    # Greedy add: start from best, try to add candidates one by one
    current = list(best)
    current_set = set(map(tuple, best))
    rng = random.Random(42)
    cand_idx = np.arange(len(candidates))
    
    # Precompute norms for efficiency
    # For each candidate, check if it can be added:
    # 1. Not already in set
    # 2. dot with every current point <= 4 (since both have norm^2=8, dist^2 = 16-2dot >= 8 iff dot <= 4)
    
    # Sort candidates by some heuristic (e.g., random order each iteration)
    # Time-bounded greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = rng.sample(range(len(candidates)), len(candidates))
        for ci in order:
            if time.time() > deadline - 10:
                break
            v = candidates[ci]
            vt = tuple(v)
            if vt in current_set:
                continue
            # Check dot <= 4 with all current points
            ok = True
            # Use numpy for speed
            dots = np.dot(current, v)
            if np.any(dots > 4):
                continue
            # Also check v is not zero (it isn't, norm^2=8)
            current.append(v)
            current_set.add(vt)
            improved = True
    
    if current:
        cand_arr = np.array(current)
        if _validate(cand_arr) and len(cand_arr) > len(best):
            best = cand_arr

    return best
