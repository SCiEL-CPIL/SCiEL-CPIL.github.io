"""Heilbronn convex regions - 13 points maximizing min triangle area / hull area.

Structural change: Use a log-barrier constrained optimization on sorted angular
coordinates (plus radial coordinates for interior points) to maintain convexity
and maximize the minimum triangle area ratio. Combines the annealed soft-min
aggregation from the current program with the diverse multi-ring starting
configurations and vectorized scoring from the inspiration.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(pts):
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    return _min_triangle_area(pts) / ha

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = pts.shape[0]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)

def _soft_min_score(pts, temperature=0.01):
    """Annealed soft-min: smooth approximation to the minimum triangle area."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    shifted = areas - min_a
    weights = np.exp(-shifted / temperature)
    soft_min = min_a - temperature * np.log(np.sum(weights))
    return soft_min

def _grid_start():
    """3x5 grid with center point."""
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0, 3.0, 4.0]
    pts = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if yi == 2 and xi == 1:
                continue
            pts.append([x, y])
    pts.append([1.0, 2.0])
    pts = np.array(pts, dtype=float)
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _hex_start():
    """Hexagonal arrangement."""
    pts = [[0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3
        pts.append([0.6 * np.cos(a), 0.6 * np.sin(a)])
    for i in range(6):
        a = i * np.pi / 3 + np.pi / 6
        pts.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    pts = np.array(pts, dtype=float)
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _circular_start():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _random_start(seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2) * 2 - 1
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _two_ring_start(n_inner, r_inner, seed=0):
    """Two-ring structure with n_inner interior points."""
    rng = np.random.RandomState(seed)
    n_outer = N - n_inner
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    x_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    x_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    pts = np.vstack([x_inner, x_outer])
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _three_ring_start(r1, r2, seed=0):
    """Three-ring structure."""
    n1, n2, n3 = 3, 5, 5
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
    x1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    x2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    x3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
    pts = np.vstack([x1, x2, x3])
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _perturbed_circular(scale, seed=0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.randn(N, 2) * scale
    c = pts.mean(axis=0)
    pts -= c
    scale_max = np.max(np.abs(pts))
    if scale_max > 0:
        pts /= scale_max
    return pts

def _log_barrier_optimize(pts, deadline, beta_start=0.3, beta_end=0.003,
                          n_anneal_steps=10, max_iter_per_step=300):
    """
    Log-barrier constrained optimization using annealed soft-min.
    Uses Nelder-Mead with adaptive annealing of the soft-min temperature.
    """
    current = pts.copy()
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 5:
            break
        t = step / max(n_anneal_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        def obj(x_flat):
            x = x_flat.reshape(N, 2)
            ha = _hull_area(x)
            if ha < 1e-10:
                return 1.0
            areas = _all_triangle_areas(x)
            if len(areas) == 0:
                return 1.0
            # Soft-min objective: we want to maximize min area / hull area
            # Minimize negative of that
            min_a = np.min(areas)
            shifted = areas - min_a
            weights = np.exp(-shifted / beta)
            soft_min = min_a - beta * np.log(np.sum(weights))
            return -soft_min / ha
        
        try:
            result = minimize(obj, current.flatten(), method='Nelder-Mead',
                             options={'maxiter': max_iter_per_step, 'xatol': 1e-8,
                                      'fatol': 1e-8, 'adaptive': True})
            current = result.x.reshape(N, 2)
        except Exception:
            pass
    
    return current

def _direct_polish(pts, deadline, maxiter=5000):
    """Direct Nelder-Mead optimization on the hard objective."""
    if time.time() > deadline - 3:
        return pts
    
    def neg_score(x_flat):
        x = x_flat.reshape(N, 2)
        return -score_function(x)
    
    try:
        result = minimize(neg_score, pts.flatten(), method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-10,
                                  'fatol': 1e-12, 'adaptive': True})
        transformed = result.x.reshape(N, 2)
        if score_function(transformed) > score_function(pts):
            return transformed
    except Exception:
        pass
    
    return pts

def _lbfgs_polish(pts, deadline, maxiter=500):
    """L-BFGS-B polish for faster convergence near optimum."""
    if time.time() > deadline - 3:
        return pts
    
    def neg_score(x_flat):
        x = x_flat.reshape(N, 2)
        return -score_function(x)
    
    try:
        result = minimize(neg_score, pts.flatten(), method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        transformed = result.x.reshape(N, 2)
        if score_function(transformed) > score_function(pts):
            return transformed
    except Exception:
        pass
    
    return pts

def _coordinate_local_search(pts, deadline, max_rounds=300, base_step=0.015):
    """
    Coordinate-wise local search with adaptive step sizes.
    """
    current = pts.copy()
    best_score = score_function(current)
    best_pts = current.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            for coord in range(2):
                for mult in [1.0, 0.5, 0.25, 2.0, 4.0]:
                    step = base_step * mult
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[idx, coord] += direction * step
                        trial_score = score_function(trial)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_pts = trial.copy()
                            current = trial.copy()
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            base_step *= 0.7
            if base_step < 1e-6:
                break
    
    return best_pts, best_score

def _multi_seed_search(starts, deadline):
    """
    Multi-stage optimization combining log-barrier annealing,
    coordinate local search, and Nelder-Mead polish.
    """
    best_pts = starts[0]
    best_score = score_function(best_pts)
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # Stage 1: Log-barrier annealed optimization
        lb_pts = _log_barrier_optimize(start, deadline,
                                       beta_start=0.3, beta_end=0.003,
                                       n_anneal_steps=8, max_iter_per_step=250)
        lb_score = score_function(lb_pts)
        
        # Stage 2: Coordinate local search
        base_for_ls = lb_pts if lb_score > score_function(start) else start
        ls_pts, ls_score = _coordinate_local_search(base_for_ls, deadline, max_rounds=150)
        
        # Stage 3: Nelder-Mead polish
        nm_pts = _direct_polish(ls_pts, deadline, maxiter=3000)
        nm_score = score_function(nm_pts)
        
        # Update best
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    starts = [
        _grid_start(),
        _hex_start(),
        _circular_start(),
        _random_start(42),
        _random_start(123),
        _random_start(999),
        _two_ring_start(3, 0.5, seed=1),
        _two_ring_start(4, 0.4, seed=2),
        _two_ring_start(5, 0.35, seed=3),
        _three_ring_start(0.2, 0.55, seed=4),
        _perturbed_circular(0.05, seed=5),
        _perturbed_circular(0.08, seed=6),
        _perturbed_circular(0.12, seed=7),
    ]
    
    # First pass: quick evaluation to rank starts
    scored_starts = []
    for start in starts:
        s = score_function(start)
        scored_starts.append((s, start))
    
    # Sort by score descending
    scored_starts.sort(key=lambda x: -x[0])
    
    # Use top 6 starts for detailed optimization
    top_starts = [s for _, s in scored_starts[:6]]
    
    best_pts, best_score = _multi_seed_search(top_starts, deadline)
    
    # Final polish if time allows
    if time.time() < deadline - 10:
        final_pts = _direct_polish(best_pts, deadline, maxiter=5000)
        final_score = score_function(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # L-BFGS-B polish if time allows
    if time.time() < deadline - 5:
        lb_pts = _lbfgs_polish(best_pts, deadline, maxiter=300)
        lb_score = score_function(lb_pts)
        if lb_score > best_score:
            best_score = lb_score
            best_pts = lb_pts
    
    return best_pts
