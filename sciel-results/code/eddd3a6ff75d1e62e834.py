"""Packing 50 equal circles in a unit circle using ring-structured optimization with soft-min."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score(points):
    """Compute the max common radius for given centers."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _softmin_logsumexp(vals, beta):
    """Numerically stable log-sum-exp soft-min."""
    max_val = np.max(vals)
    return max_val - np.log(np.sum(np.exp(beta * (vals - max_val)))) / beta


def _smooth_objective(x, beta):
    """Smooth objective: negative soft-min of all constraints."""
    c = x.reshape(N, 2)
    # Boundary distances
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    # Pairwise distances
    ci = c[np.triu_indices(N, 1)[0]]
    cj = c[np.triu_indices(N, 1)[1]]
    d = np.linalg.norm(ci - cj, axis=1) / 2.0
    all_vals = np.concatenate([d, b])
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(rings):
    """Convert ring parameters to point coordinates.
    rings: array of shape (num_rings, 3) where each row is (radius, phase, count)
    The first ring (index 0) has count=1 at center.
    """
    points = np.zeros((N, 2))
    idx = 0
    for i, (r, phase, count) in enumerate(rings):
        if i == 0:
            # Center point
            points[idx] = [0.0, 0.0]
            idx += 1
        else:
            for j in range(int(count)):
                angle = phase + 2 * np.pi * j / count
                points[idx] = [r * np.cos(angle), r * np.sin(angle)]
                idx += 1
            if idx >= N:
                break
    return points


def _points_to_rings(points):
    """Convert point coordinates to ring parameters.
    Returns array of shape (num_rings, 3): (radius, phase, count)
    """
    points = points.reshape(-1, 2)
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance
    order = np.argsort(dists)
    sorted_points = points[order]
    sorted_dists = dists[order]
    
    # Group points into rings based on distance
    rings = []
    i = 0
    while i < len(sorted_points):
        if i == 0:
            # Center
            rings.append([0.0, 0.0, 1])
            i += 1
        else:
            # Find all points at approximately the same distance
            r = sorted_dists[i]
            j = i
            while j < len(sorted_points) and abs(sorted_dists[j] - r) < 0.005:
                j += 1
            count = j - i
            if count > 0:
                # Compute phase from the first point in this ring
                pts = sorted_points[i:j]
                angles = np.arctan2(pts[:, 1], pts[:, 0])
                # Average phase (handle wraparound)
                phase = np.mean(angles)
                rings.append([r, phase, float(count)])
            i = j
    
    # Pad or truncate to get consistent structure
    # We'll use a fixed number of rings for optimization
    return np.array(rings) if rings else np.array([[0.0, 0.0, 1.0]])


def _ring_initialization():
    """Generate a ring-structured initial configuration."""
    # Start with a reasonable ring structure
    # Center + several rings
    # Try to match the known optimal packing structure
    
    # Known best packing has approximately:
    # 1 center, then rings with counts like 6, 12, 18, etc.
    # Let's try several configurations
    
    best_points = None
    best_r = 0.0
    
    # Configuration 1: 1 + 6 + 12 + 12 + 9 = 40, need 10 more
    # Let's try: 1 + 6 + 12 + 12 + 12 + 7 = 50
    configs = [
        [1, 6, 12, 12, 12, 7],
        [1, 6, 12, 12, 10, 9],
        [1, 6, 12, 10, 12, 9],
        [1, 6, 10, 12, 12, 9],
        [1, 6, 12, 12, 12, 7],
        [1, 6, 12, 11, 12, 8],
        [1, 6, 11, 12, 12, 8],
        [1, 6, 12, 12, 11, 8],
        [1, 6, 12, 12, 12, 7],
        [1, 6, 10, 10, 12, 11],
        [1, 6, 10, 12, 10, 11],
    ]
    
    for counts in configs:
        if sum(counts) != N:
            continue
        
        # Estimate radii for each ring
        # The outermost ring should have radius close to 1 - r
        # Assume r ~ 0.125
        r_est = 0.125
        max_r = 1.0 - r_est
        
        # Distribute radii roughly evenly
        n_rings = len(counts)
        if n_rings == 1:
            radii = [0.0]
        else:
            # First ring at 0, rest distributed
            radii = [0.0]
            for i in range(1, n_rings):
                radii.append(max_r * i / (n_rings - 1))
        
        # Generate points
        points = np.zeros((N, 2))
        idx = 0
        for i, count in enumerate(counts):
            if i == 0:
                points[idx] = [0.0, 0.0]
                idx += 1
            else:
                r = radii[i]
                for j in range(count):
                    angle = 2 * np.pi * j / count + (np.pi / count if i % 2 == 0 else 0)
                    points[idx] = [r * np.cos(angle), r * np.sin(angle)]
                    idx += 1
            if idx >= N:
                break
        
        r = _score(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: hexagonal
        points = []
        s = 0.25
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            best_points = points[idx]
            best_r = _score(best_points)
        else:
            best_points = np.random.randn(N, 2) * 0.5
            best_r = _score(best_points)
    
    return best_points, best_r


def _optimize_from_init(points, t_start, deadline, beta_schedule=None):
    """Optimize from an initial configuration using smooth objective."""
    if beta_schedule is None:
        beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    
    best_c = points.copy()
    best_r = _score(best_c)
    
    for beta in beta_schedule:
        if time.time() - t_start >= deadline - 3:
            break
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        def callback(x):
            if time.time() - t_start >= deadline - 2:
                raise Exception("timeout")
        
        try:
            res = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14},
                callback=callback
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _de_ring_optimize(points, t_start, deadline):
    """Break ring symmetry by adding small perturbations and optimizing."""
    best_c = points.copy()
    best_r = _score(best_c)
    
    n_restarts = 0
    while time.time() - t_start < deadline - 5 and n_restarts < 10:
        # Add small random perturbation to break symmetry
        scale = 0.01 / (1.0 + 0.2 * n_restarts)
        x0 = best_c.flatten() + np.random.randn(2 * N) * scale
        
        c_opt, r_opt = _optimize_from_init(
            x0.reshape(N, 2), 
            t_start, 
            min(deadline - 5 - n_restarts * 2, time.time() + 10)
        )
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        n_restarts += 1
    
    return best_c, best_r


def _binary_search_polish(c_init, r_init, t_start, deadline):
    """Binary search for the maximum r by checking feasibility."""
    best_c = c_init.copy()
    best_r = r_init
    
    pair_idx = np.triu_indices(N, 1)
    
    def penalty_obj(x, r_target):
        c_local = x.reshape(N, 2)
        # Boundary penalty
        b = 1.0 - np.hypot(c_local[:, 0], c_local[:, 1])
        b_pen = np.sum(np.maximum(0, r_target - b) ** 2)
        # Pairwise penalty
        ci = c_local[pair_idx[0]]
        cj = c_local[pair_idx[1]]
        d = np.linalg.norm(ci - cj, axis=1)
        p_pen = np.sum(np.maximum(0, 2 * r_target - d) ** 2)
        return b_pen + p_pen
    
    # Try to increase r slightly
    for _ in range(10):
        if time.time() - t_start >= deadline - 2:
            break
        
        r_try = best_r + 1e-6 * (1 + np.random.rand())
        
        def obj(x):
            return penalty_obj(x, r_try)
        
        def callback(x):
            if time.time() - t_start >= deadline - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-16},
                callback=callback
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    t_start = time.time()
    np.random.seed(42)
    
    # Initial ring configuration
    points, init_score = _ring_initialization()
    best_points = points.copy()
    best_score = init_score
    
    # Stage 1: Optimize from ring init with progressive beta
    if time.time() - t_start < DEADLINE - 30:
        c_opt, r_opt = _optimize_from_init(
            points, 
            t_start, 
            min(time.time() + 20, DEADLINE - 25)
        )
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Stage 2: Break ring symmetry
    if time.time() - t_start < DEADLINE - 15:
        c_opt, r_opt = _de_ring_optimize(
            best_points, 
            t_start, 
            min(time.time() + 15, DEADLINE - 10)
        )
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Stage 3: Binary search polish
    if time.time() - t_start < DEADLINE - 5:
        c_opt, r_opt = _binary_search_polish(
            best_points, 
            best_score, 
            t_start, 
            min(time.time() + 10, DEADLINE - 3)
        )
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Final: Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    r_check = _score(c_final)
    if r_check < best_score:
        best_score = r_check
    
    return c_final
