"""Concentric-rings packing for 50 equal circles in unit disk."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _compute_score(points):
    """Return the feasible common radius for a set of centres."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    d = np.linalg.norm(points, axis=1)
    margin = 1.0 - d
    if np.min(margin) < -1e-12:
        return 0.0
    # pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, np.min(margin))
    return max(r, 0.0)


def _rings_to_points(params, ring_counts):
    """Convert param vector to (N,2) points.
    params layout: [ring0_r, ring0_theta0, ring0_theta1, ..., ring1_r, ring1_theta0, ...]
    For ring 0 (center), only radius is used (should be ~0).
    """
    pts = []
    idx = 0
    for ring_i, count in enumerate(ring_counts):
        r = params[idx]
        idx += 1
        if count == 0:
            continue
        if count == 1:
            # single point at center offset by r
            pts.append([r, 0.0])
        else:
            thetas = params[idx:idx + count]
            idx += count
            # thetas are absolute angles; base evenly spaced
            base = np.linspace(0, 2 * np.pi, count, endpoint=False)
            angles = base + thetas
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            pts.append(np.column_stack([xs, ys]))
    return np.vstack(pts)


def _neg_score(params, ring_counts):
    pts = _rings_to_points(params, ring_counts)
    if not np.all(np.isfinite(pts)):
        return 1e6
    d = np.linalg.norm(pts, axis=1)
    margin = 1.0 - d
    min_margin = np.min(margin)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, min_margin)
    # Penalize infeasibility heavily
    if min_margin < -1e-9:
        return 1e6 - r
    return -r


def _refine_local(points, max_iters=200, tol=1e-10):
    """Local refinement: iteratively push apart the most-violating pair / wall."""
    pts = points.copy()
    for _ in range(max_iters):
        # compute current r
        d = np.linalg.norm(pts, axis=1)
        margin = 1.0 - d
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_margin = np.min(margin)
        r = min(min_pair / 2.0, min_margin)
        
        # Find the most violated constraint
        if min_pair / 2.0 < min_margin - 1e-12:
            # pair constraint is tighter
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i == j:
                continue
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-14:
                continue
            di /= dist
            # push apart
            push = (2 * r - dist) / 2.0
            if push > 1e-14:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # wall constraint is tighter
            i = int(np.argmin(margin))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-14:
                continue
            di = pi / dist
            push = (1.0 - r - dist)
            if push > 1e-14:
                pts[i] += push * di
        # Check convergence
        if abs(r - _compute_score(pts)) < tol:
            break
    return pts


def construct():
    import time
    t0 = time.time()
    
    # Ring structure: 1 + 6 + 12 + 18 + 13 = 50
    ring_counts = [1, 6, 12, 18, 13]
    
    # Build parameter vector
    # For each ring: 1 radius param + (count-1) angle deviations (for count>1), or 0 for count==1
    # Actually for count==1, we just have radius (position on x-axis, but we'll allow offset)
    # Let me use: for each ring, radius + count angle params (absolute offsets from base)
    param_count = 0
    for c in ring_counts:
        param_count += 1  # radius
        if c > 1:
            param_count += c  # angle deviations
    # Total: 1 + (1+6) + (1+12) + (1+18) + (1+13) = 1+7+13+19+14 = 54
    
    # Initial guess: evenly spaced rings
    initial_params = np.zeros(param_count)
    idx = 0
    # Ring radii initial guesses (approximate)
    initial_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    for ring_i, (count, init_r) in enumerate(zip(ring_counts, initial_radii)):
        initial_params[idx] = init_r
        idx += 1
        if count > 1:
            # angles start at 0 deviation
            idx += count
    
    # Optimize
    best_score = -np.inf
    best_points = None
    
    deadline = t0 + 120.0
    remaining = deadline - time.time()
    
    try:
        result = minimize(
            _neg_score,
            initial_params,
            args=(ring_counts,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.success or result.fun < best_score:
            pts = _rings_to_points(result.x, ring_counts)
            s = _compute_score(pts)
            if s > best_score and np.all(np.isfinite(pts)):
                best_score = s
                best_points = pts
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        if best_points is not None:
            return best_points
        # Fallback
        return _fallback()
    
    # Local refinement
    if best_points is not None:
        refined = _refine_local(best_points, max_iters=500, tol=1e-12)
        s_ref = _compute_score(refined)
        if s_ref > best_score:
            best_score = s_ref
            best_points = refined
    
    # Try a few different initial guesses if time allows
    if time.time() < deadline - 15:
        for trial in range(5):
            # Perturb initial angles
            rng = np.random.RandomState(trial * 42 + 7)
            params = initial_params.copy()
            idx = 0
            for ring_i, (count, init_r) in enumerate(zip(ring_counts, initial_radii)):
                params[idx] = init_r * (0.9 + 0.2 * rng.random())
                idx += 1
                if count > 1:
                    params[idx:idx+count] = (rng.random(count) - 0.5) * 0.3
                    idx += count
            try:
                res = minimize(
                    _neg_score,
                    params,
                    args=(ring_counts,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts = _rings_to_points(res.x, ring_counts)
                s = _compute_score(pts)
                if s > best_score and np.all(np.isfinite(pts)):
                    best_score = s
                    best_points = pts
                    # Quick refine
                    refined = _refine_local(pts, max_iters=200, tol=1e-12)
                    s_ref = _compute_score(refined)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_points = refined
            except Exception:
                pass
            if time.time() > deadline - 10:
                break
    
    if best_points is None or not np.all(np.isfinite(best_points)):
        return _fallback()
    
    return best_points


def _fallback():
    """Hexagonal lattice fallback."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98
