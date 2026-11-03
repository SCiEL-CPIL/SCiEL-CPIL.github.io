import numpy as np
from scipy.spatial.distance import cdist
import time

N = 200
HALF = 0.5
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill


def _score(pts):
    """Compute the maximum radius for a set of N points in [-0.5, 0.5]^2."""
    if pts.shape[0] != N or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -HALF) or np.any(pts > HALF):
        return 0.0
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    wall = np.min([
        np.min(pts[:, 0]) + HALF,
        HALF - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + HALF,
        HALF - np.max(pts[:, 1]),
    ])
    r = min(min_pair / 2.0, wall)
    return r if r > 0 else 0.0


def _soft_min_surrogate(pts, beta=50.0):
    """
    Compute a smooth approximation of the minimum of pairwise distances/2
    and wall distances using soft-min. Returns (value, gradient wrt pts).
    """
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Pairwise distances / 2
    pair_vals = D / 2.0
    # Wall distances
    wall_vals = np.array([
        np.min(pts[:, 0]) + HALF,
        HALF - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + HALF,
        HALF - np.max(pts[:, 1]),
    ])
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    # Numerically stable version
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / beta
    
    # Gradient of soft min wrt each value
    # d(soft_min)/d(val_k) = -exp(-beta*(val_k - min_val)) / sum(exp(-beta*(val_j - min_val)))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient wrt pts
    grad = np.zeros_like(pts)
    
    # Pairwise part
    pair_weights = weights[:N * N].reshape(N, N)
    # For pair (i,j), contribution to grad[i] and grad[j]
    # d(pair_val_ij)/d(pts[i]) = (pts[i] - pts[j]) / D[i,j]
    # d(pair_val_ij)/d(pts[j]) = (pts[j] - pts[i]) / D[i,j]
    for i in range(N):
        for j in range(N):
            if i != j:
                w = pair_weights[i, j]
                if w > 1e-12 and D[i, j] > 1e-12:
                    diff = pts[i] - pts[j]
                    grad[i] += w * diff / D[i, j]
                    grad[j] -= w * diff / D[i, j]
    
    # Wall part
    wall_weights = weights[N * N:]
    if wall_weights[0] > 1e-12:  # left wall: min(pts[:,0]) + HALF
        idx = np.argmin(pts[:, 0])
        grad[idx, 0] += wall_weights[0]
    if wall_weights[1] > 1e-12:  # right wall: HALF - max(pts[:,0])
        idx = np.argmax(pts[:, 0])
        grad[idx, 0] -= wall_weights[1]
    if wall_weights[2] > 1e-12:  # bottom wall
        idx = np.argmin(pts[:, 1])
        grad[idx, 1] += wall_weights[2]
    if wall_weights[3] > 1e-12:  # top wall
        idx = np.argmax(pts[:, 1])
        grad[idx, 1] -= wall_weights[3]
    
    return soft_min_val, grad


def _gradient_polish(pts, max_iters=500, beta=50.0, lr=0.002):
    """
    Gradient-based polish using L-BFGS-B style updates with bounds.
    Maximizes the soft-min surrogate of the minimum radius.
    """
    from scipy.optimize import minimize
    
    pts = pts.copy()
    
    def neg_objective(x):
        x = x.reshape(N, 2)
        val, _ = _soft_min_surrogate(x, beta=beta)
        return -val
    
    def neg_gradient(x):
        x = x.reshape(N, 2)
        _, grad = _soft_min_surrogate(x, beta=beta)
        return -grad.ravel()
    
    bounds = [(-HALF, HALF)] * (2 * N)
    x0 = pts.ravel()
    
    try:
        result = minimize(
            neg_objective, x0, jac=neg_gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iters, 'gtol': 1e-8, 'maxls': 20}
        )
        if result.success or (not result.success and result.fun < neg_objective(x0)):
            best_pts = result.x.reshape(N, 2)
            return best_pts
    except Exception:
        pass
    
    return pts


def _hex_points(s, ox, oy):
    """
    Generate hexagonal lattice points with spacing s and offset (ox, oy).
    Returns points inside [-0.5, 0.5]^2.
    """
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    
    pts = []
    for j in range(cols):
        x = -0.5 + j * dx + ox
        if x < -HALF - 1e-9 or x > HALF + 1e-9:
            continue
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i in range(rows):
            y = -0.5 + i * dy + oy + yoff
            if y < -HALF - 1e-9 or y > HALF + 1e-9:
                continue
            pts.append((x, y))
    
    arr = np.array(pts)
    if arr.shape[0] == 0:
        return arr
    keep = np.all(np.abs(arr) <= HALF + 1e-9, axis=1)
    return arr[keep]


def _greedy_select_fast(pts, N_target):
    """
    Faster greedy farthest-point selection using vectorized operations.
    """
    if len(pts) <= N_target:
        return pts.copy()
    
    n = len(pts)
    selected = []
    remaining_mask = np.ones(n, dtype=bool)
    
    # Start with point closest to origin
    d2_origin = np.sum(pts ** 2, axis=1)
    first = int(np.argmin(d2_origin))
    selected.append(first)
    remaining_mask[first] = False
    
    # min_dist for each point to the set of selected points
    min_dists = np.linalg.norm(pts - pts[first], axis=1)
    
    for _ in range(N_target - 1):
        # Among remaining, find max min_dist
        min_dists_sel = min_dists.copy()
        min_dists_sel[~remaining_mask] = -np.inf
        best_idx = int(np.argmax(min_dists_sel))
        
        if not remaining_mask[best_idx]:
            break
        
        selected.append(best_idx)
        remaining_mask[best_idx] = False
        
        # Update min_dists
        d = np.linalg.norm(pts - pts[best_idx], axis=1)
        min_dists = np.minimum(min_dists, d)
    
    return pts[selected]


def _coordinate_ascent_polish(c, iters=3000, step0=5e-3, seed=42):
    """
    Coordinate ascent polish: try small perturbations in each coordinate
    and keep if they improve the score.
    """
    c = c.copy()
    best = _score(c)
    rng = np.random.default_rng(seed)
    step = step0
    
    for it in range(iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Pick a random point and coordinate
        i = int(rng.integers(N))
        d = int(rng.integers(2))
        
        improved = False
        for sgn in (1.0, -1.0):
            for sc in (step, step * 0.5, step * 2.0):
                c2 = c.copy()
                c2[i, d] += sgn * sc
                if not np.all(np.abs(c2) <= HALF + 1e-12):
                    continue
                v = _score(c2)
                if v > best + 1e-12:
                    c = c2
                    best = v
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.85
        else:
            step = min(step * 1.05, 8e-3)
    
    return c, best


def _gradient_polish_wrapper(pts, max_iters=300, beta=50.0):
    """
    Wrapper for gradient polish that returns (pts, score).
    """
    if time.time() > _TIME_LIMIT - 10:
        return pts, _score(pts)
    
    polished = _gradient_polish(pts, max_iters=max_iters, beta=beta)
    polished = np.clip(polished, -HALF, HALF)
    r = _score(polished)
    return polished, r


def _local_refine(pts, n_iters=2000):
    """
    Local refinement: iteratively find the bottleneck (closest pair or wall)
    and adjust points to increase the minimum radius.
    """
    pts = pts.copy()
    
    for it in range(n_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find min pair
        min_pair_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_pair_idx
        d_pair = D[i, j]
        
        # Find min wall distance
        wall_dists = np.array([
            np.min(pts[:, 0]) + HALF,
            HALF - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + HALF,
            HALF - np.max(pts[:, 1]),
        ])
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        # Learning rate
        lr = 0.005 / (1 + it * 0.005)
        
        # Determine which constraint is limiting
        if d_pair / 2.0 <= min_wall_val:
            # Push pair apart
            diff = pts[j] - pts[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                pts[i, 0] += 0.001
                pts[j, 0] -= 0.001
            else:
                unit = diff / norm
                step = lr * min(1.0, 1.0 / norm)
                pts[i] -= step * unit
                pts[j] += step * unit
        else:
            # Push wall particle inward
            if min_wall_idx == 0:  # left wall
                idx = np.argmin(pts[:, 0])
                pts[idx, 0] += lr
            elif min_wall_idx == 1:  # right wall
                idx = np.argmax(pts[:, 0])
                pts[idx, 0] -= lr
            elif min_wall_idx == 2:  # bottom wall
                idx = np.argmin(pts[:, 1])
                pts[idx, 1] += lr
            else:  # top wall
                idx = np.argmax(pts[:, 1])
                pts[idx, 1] -= lr
        
        # Clamp to bounds
        pts[:, 0] = np.clip(pts[:, 0], -HALF + 1e-9, HALF - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -HALF + 1e-9, HALF - 1e-9)
    
    return pts


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    rng = np.random.default_rng(2024)
    
    # Target spacing estimate for hex packing of 200 circles in unit square
    # Area per circle ~ 1/200, hex spacing s ~ sqrt(1/(200 * sqrt(3)/2)) ~ 0.076
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Phase 1: Multiple hex-grid trials with random offsets and varied spacing
    n_trials = 40
    for trial in range(n_trials):
        if time.time() > _TIME_LIMIT - 40:
            break
        
        # Vary spacing slightly and use random offsets
        s = s0 * (0.97 + 0.06 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        
        cands = _hex_points(s, ox, oy)
        if cands.shape[0] < N:
            continue
        
        # Greedy select N points
        selected = _greedy_select_fast(cands, N)
        if selected.shape[0] < N:
            continue
        
        r = _score(selected)
        
        # Local refine
        refined = _local_refine(selected, n_iters=200)
        r_ref = _score(refined)
        
        if r_ref > best_r:
            best_r = r_ref
            best_pts = refined.copy()
        
        # Gradient polish on refined solution
        if time.time() < _TIME_LIMIT - 20:
            polished, r_pol = _gradient_polish_wrapper(refined, max_iters=200, beta=50.0)
            if r_pol > best_r:
                best_r = r_pol
                best_pts = polished.copy()
    
    # Phase 2: If we have time, do more gradient polish on the best solution
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 15:
        polished, r_pol = _gradient_polish_wrapper(best_pts, max_iters=300, beta=80.0)
        if r_pol > best_r:
            best_r = r_pol
            best_pts = polished.copy()
    
    # Phase 3: Coordinate ascent for fine-tuning
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        polished, r_pol = _coordinate_ascent_polish(best_pts, iters=1500, seed=999)
        if r_pol > best_r:
            best_r = r_pol
            best_pts = polished.copy()
    
    # Phase 4: Final local refinement
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        final_pts = _local_refine(best_pts, n_iters=300)
        r_final = _score(final_pts)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts
    
    if best_pts is None:
        # Fallback: simple grid
        rows, cols = 14, 15
        xs = np.linspace(-0.45, 0.45, cols)
        ys = np.linspace(-0.45, 0.45, rows)
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    # Final validity check and clamp
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
