# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def _score(points):
    """Compute R = (d_min/d_max)^2 from 120 pairwise distances."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _soft_min(d, beta=100.0):
    """Soft minimum using log-sum-exp trick."""
    d = np.asarray(d)
    if len(d) == 0:
        return 0.0
    min_d = d.min()
    if min_d <= 0:
        return 0.0
    scaled = -d / beta
    max_scaled = scaled.max()
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum * beta

def _soft_max(d, beta=100.0):
    """Soft maximum using log-sum-exp trick."""
    d = np.asarray(d)
    if len(d) == 0:
        return 0.0
    max_d = d.max()
    if max_d <= 0:
        return 0.0
    scaled = d / beta
    max_scaled = scaled.max()
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum * beta

def _smooth_objective_log(flat_pts, n=16, beta=100.0):
    """Minimize -2*log(soft_min/soft_max) for better conditioning."""
    pts = flat_pts.reshape(n, 2)
    d = pdist(pts)
    if len(d) == 0:
        return 0.0
    sm = _soft_min(d, beta)
    smx = _soft_max(d, beta)
    if sm <= 0 or smx <= 0:
        return 0.0
    return -2.0 * np.log(sm / smx)

def _neg_ratio_flat(flat_pts, n=16):
    """Nonsmooth objective for Powell: minimize -(d_min/d_max)^2."""
    pts = flat_pts.reshape(n, 2)
    d = pdist(pts)
    if len(d) == 0:
        return 0.0
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return -(d_min / d_max) ** 2

def _make_hexagonal_initial():
    """
    Generate a 16-point configuration based on a hexagonal lattice.
    Center + 6 (r=1) + 9 (r=sqrt(3)).
    """
    pts = []
    pts.append([0.0, 0.0])
    for k in range(6):
        angle = k * 2 * np.pi / 6
        pts.append([np.cos(angle), np.sin(angle)])
    r_outer = np.sqrt(3)
    for k in range(9):
        angle = k * 2 * np.pi / 9
        pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    pts = np.array(pts[:16])
    pts -= np.mean(pts, axis=0)
    return pts

def _make_grid_initial():
    """4x4 grid."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    pts -= np.mean(pts, axis=0)
    return pts

def _make_circle_initial():
    """16 points evenly spaced on a circle."""
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    pts -= np.mean(pts, axis=0)
    return pts

def _make_random_initial(seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(16, 2) * 2 - 1
    pts -= np.mean(pts, axis=0)
    return pts

def _improve_powell(pts, max_iter=1000, n=16):
    """Local optimization using Powell's method."""
    x0 = pts.flatten()
    res = minimize(
        _neg_ratio_flat,
        x0,
        args=(n,),
        method='Powell',
        options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-12}
    )
    return res.x.reshape(n, 2)

def _improve_lbfgs_smooth(pts, n=16, beta=100.0, max_iter=500):
    """Local optimization using L-BFGS-B with smooth surrogate."""
    x0 = pts.flatten()
    res = minimize(
        _smooth_objective_log,
        x0,
        args=(n, beta),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return res.x.reshape(n, 2)

def _final_polish(pts, n=16):
    """High-precision polish: alternate between smooth L-BFGS-B and exact Powell."""
    current = pts.copy()
    for _ in range(3):
        current = _improve_lbfgs_smooth(current, n=n, beta=50.0, max_iter=300)
        current = _improve_powell(current, max_iter=500, n=n)
    return current

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s

    best_pts = None
    best_val = -np.inf

    # Generate initial configurations - focus on high-quality starts
    initial_configs = [
        _make_hexagonal_initial(),
        _make_grid_initial(),
        _make_circle_initial(),
        _make_random_initial(1),
        _make_random_initial(42),
    ]

    # Add perturbed versions of structured configs
    rng = np.random.RandomState(0)
    base_hex = _make_hexagonal_initial()
    for noise_scale in [0.03, 0.08, 0.15]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    base_grid = _make_grid_initial()
    for noise_scale in [0.05, 0.1]:
        perturbed = base_grid + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    # Optimize each configuration
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break

        # 1. L-BFGS-B with smooth surrogate (fast)
        smooth_pts = _improve_lbfgs_smooth(init_pts, n=n, beta=100.0, max_iter=200)
        smooth_val = _score(smooth_pts)

        if smooth_val > best_val:
            best_val = smooth_val
            best_pts = smooth_pts

        # 2. Powell on the smooth result (more precise)
        if time.time() < deadline - 90:
            powell_pts = _improve_powell(smooth_pts, max_iter=300, n=n)
            powell_val = _score(powell_pts)

            if powell_val > best_val:
                best_val = powell_val
                best_pts = powell_pts

        # 3. If this is competitive, do final polish
        if best_pts is not None and powell_val > best_val - 0.005 and time.time() < deadline - 120:
            polished = _final_polish(powell_pts, n=n)
            polished_val = _score(polished)
            if polished_val > best_val:
                best_val = polished_val
                best_pts = polished

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 30:
        final_pts = _final_polish(best_pts, n=n)
        final_val = _score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val

    # If we still have time, try differential evolution
    if best_pts is not None and time.time() < deadline - 150:
        def de_obj(x):
            return _neg_ratio_flat(x, n=n)
        
        bounds = [(-5, 5)] * (n * 2)
        
        try:
            de_res = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=80,
                popsize=12,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            de_pts = de_res.x.reshape(n, 2)
            de_val = _score(de_pts)
            
            if de_val > best_val:
                best_val = de_val
                best_pts = de_pts
                
                if time.time() < deadline - 60:
                    polished_pts = _final_polish(de_pts, n=n)
                    polished_val = _score(polished_pts)
                    if polished_val > best_val:
                        best_val = polished_val
                        best_pts = polished_pts
        except Exception:
            pass

    # Ensure we have a valid result
    if best_pts is None:
        best_pts = _make_hexagonal_initial()

    # Final sanity check and shape enforcement
    best_pts = np.asarray(best_pts, dtype=np.float64).reshape(16, 2)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _make_hexagonal_initial()
        best_pts = np.asarray(best_pts, dtype=np.float64).reshape(16, 2)

    return best_pts

# EVOLVE-BLOCK-END
