# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _regular_polygon(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def equidistant_points_polygon20() -> np.ndarray:
    """
    Coordinate ascent on radii with fixed regular angles, seeded from analytic profiles.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def radii_to_pts(r):
        return np.column_stack([r * cos_a, r * sin_a])
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        pts = radii_to_pts(r)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()

    # Seed 1: Regular
    try_candidate(np.ones(n))
    
    # Define analytic seeds
    seeds = []
    # Alternating
    seeds.append(1.0 + 0.1 * np.array([1, -1] * 10))
    seeds.append(1.0 + 0.05 * np.array([1, -1] * 10))
    seeds.append(1.0 - 0.1 * np.array([1, -1] * 10))
    # Sinusoidal
    for k in [2, 3, 5, 7]:
        for amp in [0.05, 0.1, 0.2]:
            seeds.append(1.0 + amp * np.sin(k * angles))
            seeds.append(1.0 - amp * np.sin(k * angles))
            seeds.append(1.0 + amp * np.cos(k * angles))
            seeds.append(1.0 - amp * np.cos(k * angles))
    # Polynomial (symmetric around 0)
    x = np.linspace(-1, 1, n)
    seeds.append(1.0 + 0.1 * x**2)
    seeds.append(1.0 - 0.1 * x**2)
    seeds.append(1.0 + 0.05 * x**4)
    
    # Try seeds
    for r in seeds:
        if time.time() > deadline:
            break
        # Clamp radii to be positive and reasonable
        r = np.clip(r, 0.5, 2.0)
        try_candidate(r)
        
        # Coordinate ascent from this seed
        current = r.copy()
        step = 0.05
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                # Try increasing
                new_r = current.copy()
                new_r[i] += step
                if new_r[i] > 2.5: continue
                s_new = _score_polygon(radii_to_pts(new_r))
                if s_new > best_score:
                    best_score = s_new
                    best_radii = new_r.copy()
                    current = new_r
                    improved = True
                    continue
                # Try decreasing
                new_r = current.copy()
                new_r[i] -= step
                if new_r[i] < 0.5: continue
                s_new = _score_polygon(radii_to_pts(new_r))
                if s_new > best_score:
                    best_score = s_new
                    best_radii = new_r.copy()
                    current = new_r
                    improved = True
            
            if improved:
                step *= 0.8
            else:
                # Reduce step if no improvement in full pass
                step *= 0.7
                if step < 0.001:
                    break

    # Final local search around global best
    if best_score < 1.0 and time.time() < deadline:
        current = best_radii.copy()
        step = 0.01
        for _ in range(500):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    new_r = current.copy()
                    new_r[i] += delta
                    if new_r[i] < 0.5 or new_r[i] > 2.5:
                        continue
                    s_new = _score_polygon(radii_to_pts(new_r))
                    if s_new > best_score:
                        best_score = s_new
                        best_radii = new_r.copy()
                        current = new_r
                        improved = True
            if not improved:
                step *= 0.8
                if step < 0.0001:
                    break

    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
