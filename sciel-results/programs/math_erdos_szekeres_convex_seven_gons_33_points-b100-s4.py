# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 590  # wall-clock deadline with safety margin
_BEST_POINTS = None
_BEST_SCORE = 0.0  # 1/(1+C), higher is better (fewer convex 7-gons)


def _check_valid(points: np.ndarray) -> bool:
    """Validate according to the task contract."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    n = 33
    diff = points[:, None, :] - points[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, 1.0)
    if np.any(sq_dists < 1e-12):
        return False
    # Collinearity check
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                cross = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
                if abs(cross) < 1e-9:
                    return False
    return True


def _count_convex_7(points: np.ndarray) -> int:
    """
    Count number of 7-point subsets in convex position using DP over convex chains.
    """
    n = 33
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    
    total = 0
    
    for i in range(n):
        later = list(range(i + 1, n))
        if len(later) < 6:
            continue
        
        pi = pts[i]
        
        def angle_key(j):
            return np.arctan2(pts[j, 1] - pi[1], pts[j, 0] - pi[0])
        
        later_sorted = sorted(later, key=angle_key)
        m = len(later_sorted)
        
        # Precompute cross products for all triples in later_sorted
        # cross[a][b][c] = (bx-ax)(cy-by) - (by-ay)(cx-bx) for a<b<c in angle order
        # Also need cross with pivot: cross_p[a][b] = cross(pivot, a, b)
        
        # dp[l][j][k] = number of convex chains of length l ending with ...->j->k
        # where j<k in angle order, l is number of vertices after pivot (l>=2)
        
        # l=2: dp2[j][k] = 1 if cross(pivot, pts[j], pts[k]) > 1e-9
        dp2 = np.zeros((m, m), dtype=np.int64)
        for j in range(m):
            for k in range(j + 1, m):
                a = pi
                b = pts[later_sorted[j]]
                c = pts[later_sorted[k]]
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if cross > 1e-9:
                    dp2[j][k] = 1
        
        # l=3 to l=6
        dp = [None, None, dp2]  # dp[2] = dp2
        for l in range(3, 7):
            dpl = np.zeros((m, m), dtype=np.int64)
            prev_dp = dp[l - 1]
            for j in range(l - 2, m):  # j must be at least l-2 (0-indexed, need l-1 points before)
                for k in range(j + 1, m):
                    b = pts[later_sorted[j]]
                    c = pts[later_sorted[k]]
                    s = 0
                    for i_idx in range(j):
                        if prev_dp[i_idx][j] == 0:
                            continue
                        a = pts[later_sorted[i_idx]]
                        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                        if cross > 1e-9:
                            s += prev_dp[i_idx][j]
                    dpl[j][k] = s
            dp.append(dpl)
        
        dp6 = dp[6]
        
        # Count 7-gons: chains of length 6 where closing turn (v5, v6, pivot) is convex
        for j in range(4, m):
            for k in range(j + 1, m):
                if dp6[j][k] == 0:
                    continue
                a = pts[later_sorted[j]]
                b = pts[later_sorted[k]]
                c = pi
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if cross > 1e-9:
                    total += dp6[j][k]
    
    return total


def _evaluate(points: np.ndarray):
    """Return (valid, score) where score = 1/(1+C)."""
    if not _check_valid(points):
        return False, 0.0
    C = _count_convex_7(points)
    return True, 1.0 / (1.0 + C)


def _generate_clustered(seed: int) -> np.ndarray:
    """Generate points in tight clusters to minimize convex 7-gons."""
    rng = np.random.RandomState(seed)
    # Use 4-5 clusters
    n_clusters = rng.randint(4, 6)
    centers = rng.rand(n_clusters, 2) * 0.8 + 0.1
    spread = 0.05
    
    pts = []
    for _ in range(33):
        c = centers[rng.randint(n_clusters)]
        p = c + rng.randn(2) * spread
        pts.append(p)
    pts = np.array(pts)
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_arc(seed: int) -> np.ndarray:
    """Generate points on/near a convex arc (fewer convex polygons)."""
    rng = np.random.RandomState(seed)
    # Points along a circular arc
    theta = np.linspace(0, np.pi * 0.7, 33) + rng.randn(33) * 0.02
    r = 0.4 + rng.randn(33) * 0.05
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    pts = np.column_stack([x, y])
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_concentric(seed: int) -> np.ndarray:
    """Generate points in concentric rings."""
    rng = np.random.RandomState(seed)
    n_rings = 4
    pts = []
    for ring in range(n_rings):
        r = 0.1 + ring * 0.15
        n_in_ring = 33 // n_rings if ring < n_rings - 1 else 33 - (33 // n_rings) * (n_rings - 1)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + rng.randn(n_in_ring) * 0.1
        x = 0.5 + r * np.cos(angles)
        y = 0.5 + r * np.sin(angles)
        pts.append(np.column_stack([x, y]))
    pts = np.vstack(pts)
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_grid_jittered(seed: int) -> np.ndarray:
    """Generate points on a jittered grid."""
    rng = np.random.RandomState(seed)
    n_pts = 6
    x = np.linspace(0.05, 0.95, n_pts)
    y = np.linspace(0.05, 0.95, n_pts)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    idx = rng.permutation(len(pts))[:33]
    pts = pts[idx]
    pts += rng.randn(33, 2) * 0.02
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def _generate_random(seed: int) -> np.ndarray:
    """Generate random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(33, 2)


def _perturb(points: np.ndarray, scale: float, rng: np.random.RandomState) -> np.ndarray:
    """Add small random perturbation and clip to [0,1]."""
    perturbed = points + rng.randn(33, 2) * scale
    perturbed = np.clip(perturbed, 0.0, 1.0)
    return perturbed


def erdos_szekeres_problem7() -> np.ndarray:
    global _BEST_POINTS, _BEST_SCORE, _DEADLINE
    
    start_time = time.time()
    
    # Phase 1: Try a variety of initial constructions
    candidates = []
    
    # Clustered
    for seed in range(30):
        candidates.append(_generate_clustered(seed))
    
    # Arc
    for seed in range(20):
        candidates.append(_generate_arc(seed))
    
    # Concentric
    for seed in range(20):
        candidates.append(_generate_concentric(seed))
    
    # Grid jittered
    for seed in range(20):
        candidates.append(_generate_grid_jittered(seed))
    
    # Random
    for seed in range(20):
        candidates.append(_generate_random(seed))
    
    # Evaluate all candidates
    for pts in candidates:
        if time.time() > _DEADLINE:
            break
        valid, score = _evaluate(pts)
        if valid and score > _BEST_SCORE:
            _BEST_SCORE = score
            _BEST_POINTS = pts.copy()
    
    # Phase 2: Local search (perturbation) around the best found
    if _BEST_POINTS is not None:
        current_best = _BEST_POINTS.copy()
        current_score = _BEST_SCORE
        rng = np.random.RandomState(12345)
        
        scales = [0.03, 0.01, 0.005, 0.002, 0.001]
        
        for scale in scales:
            if time.time() > _DEADLINE:
                break
            n_trials = 150
            for _ in range(n_trials):
                if time.time() > _DEADLINE:
                    break
                candidate = _perturb(current_best, scale, rng)
                valid, score = _evaluate(candidate)
                if valid and score > current_score:
                    current_score = score
                    current_best = candidate
                    _BEST_SCORE = score
                    _BEST_POINTS = candidate.copy()
    
    # Return the best found, or a default if nothing valid
    if _BEST_POINTS is not None:
        return _BEST_POINTS
    
    # Fallback: random points
    np.random.seed(0)
    return np.random.rand(33, 2)


# EVOLVE-BLOCK-END
