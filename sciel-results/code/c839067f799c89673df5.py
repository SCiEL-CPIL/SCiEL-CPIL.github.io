"""Pack 50 equal circles in a unit circle; maximize the common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 142.0  # safety margin before 150s kill


def _score(pts):
    """Compute the packing radius for a set of centers."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check all centers are inside unit circle
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    # Minimum distance to boundary
    r_boundary = 1.0 - np.max(dists)
    # Minimum pairwise distance / 2 (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf to exclude self
    np.fill_diagonal(pair_dists, np.inf)
    r_pair = np.min(pair_dists) / 2.0
    return min(r_boundary, r_pair)


def _soft_min_objective(x, beta):
    """
    Smooth surrogate: maximize soft-min of (boundary distances, pairwise half-distances).
    We minimize the negative soft-min.
    x: flat array of 2N coordinates.
    """
    pts = x.reshape(N, 2)
    # Boundary terms: 1 - |c_i|
    dists = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - dists
    # Pairwise half-distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(pair_dists, np.inf)
    # Collect all terms
    all_terms = np.concatenate([boundary, pair_dists[np.triu_indices(N, k=1)]])
    # Soft-min via log-sum-exp (numerically stable)
    m = np.min(all_terms)
    shifted = all_terms - m
    log_sum = m + np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -log_sum / beta
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min_val


def _ring_init(n, inner_r=0.0, outer_r=1.0):
    """Generate N points in concentric rings with better spacing."""
    pts = np.zeros((N, 2))
    if N == 1:
        return pts
    
    # Use a hexagonal-like pattern: center + rings
    # For 50 circles, use rings of 1, 6, 12, 18, 13 (or similar)
    ring_sizes = [1, 6, 12, 18, 13]  # Total = 50
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    idx = 0
    for size, radius in zip(ring_sizes, ring_radii):
        if idx >= N:
            break
        if size == 1:
            pts[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
            # Offset alternate rings to improve packing
            if len([r for r in ring_radii[:ring_radii.index(radius)]]) % 2 == 1:
                angles += np.pi / size
            pts[idx:idx+size, 0] = radius * np.cos(angles)
            pts[idx:idx+size, 1] = radius * np.sin(angles)
            idx += size
    return pts[:N]


def _random_init():
    """Random initialization within unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _gradient_ascent_step(pts, step_size=0.01, iterations=500):
    """
    Simple gradient ascent on the soft-min objective using numerical gradients.
    """
    beta = 50.0
    for it in range(iterations):
        x = pts.flatten()
        f0 = _soft_min_objective(x, beta)
        # Numerical gradient
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            fp = _soft_min_objective(xp, beta)
            xm = x.copy()
            xm[i] -= eps
            fm = _soft_min_objective(xm, beta)
            grad[i] = (fp - fm) / (2 * eps)
        # Gradient ascent
        pts = (pts + step_size * grad.reshape(N, 2))
        # Project back: keep inside unit circle
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask][:, None] * 0.999
    return pts


def _violation_obj_and_grad(x, r_target):
    """
    Vectorized violation objective and gradient for radius r_target.
    Minimizes sum of squared violations of:
      |c_i| <= 1 - r_target
      |c_i - c_j| >= 2 * r_target
    """
    pts_x = x.reshape(N, 2)
    
    # Boundary violations
    dists = np.hypot(pts_x[:, 0], pts_x[:, 1])
    boundary_viol = np.maximum(0.0, dists - (1.0 - r_target))
    
    # Pairwise violations (vectorized)
    diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(pair_dists, np.inf)
    
    # Only consider pairs with distance < 2*r_target
    viol_mask = pair_dists < 2 * r_target
    pair_viol = np.where(viol_mask, (2 * r_target - pair_dists) ** 2, 0.0)
    
    total = np.sum(boundary_viol ** 2) + np.sum(pair_viol)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Boundary gradient
    bmask = boundary_viol > 0
    if np.any(bmask):
        for i in np.where(bmask)[0]:
            if dists[i] > 1e-10:
                grad[2*i] = 2 * boundary_viol[i] * pts_x[i, 0] / dists[i]
                grad[2*i+1] = 2 * boundary_viol[i] * pts_x[i, 1] / dists[i]
    
    # Pairwise gradient
    for i in range(N):
        for j in range(i + 1, N):
            if viol_mask[i, j]:
                d = pair_dists[i, j]
                if d > 1e-10:
                    viol = 2 * r_target - d
                    dx = (pts_x[i, 0] - pts_x[j, 0]) / d
                    dy = (pts_x[i, 1] - pts_x[j, 1]) / d
                    grad[2*i] += 2 * viol * dx
                    grad[2*i+1] += 2 * viol * dy
                    grad[2*j] -= 2 * viol * dx
                    grad[2*j+1] -= 2 * viol * dy
    
    return total, grad


def _check_feasible(pts, r_target):
    """Check if all constraints are satisfied for radius r_target."""
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0 - r_target + 1e-9):
        return False
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(pair_dists, np.inf)
    if np.any(pair_dists < 2 * r_target - 1e-9):
        return False
    return True


def construct():
    start_time = time.time()
    
    # Best solution found so far
    best_pts = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Initial solution
    pts = _ring_init(N)
    update_best(pts)
    
    # Binary search on radius with L-BFGS-B feasibility
    r_low = 0.10
    r_high = 0.15
    
    # Multiple restarts with binary search
    for restart in range(15):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        if restart == 0:
            current_pts = _ring_init(N)
        elif restart < 5:
            current_pts = _random_init()
        else:
            # Perturb the best solution
            current_pts = best_pts + np.random.normal(0, 0.005, (N, 2))
            dists = np.hypot(current_pts[:, 0], current_pts[:, 1])
            mask = dists > 1.0
            if np.any(mask):
                current_pts[mask] = current_pts[mask] / dists[mask][:, None] * 0.9999
        
        # Binary search on r
        for _ in range(25):
            if time.time() - start_time > DEADLINE - 25:
                break
            r_target = (r_low + r_high) / 2.0
            
            x0 = current_pts.flatten()
            try:
                res = minimize(_violation_obj_and_grad, x0, args=(r_target,), jac=True,
                               method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                opt_pts = res.x.reshape(N, 2)
                
                if _check_feasible(opt_pts, r_target):
                    r_low = r_target
                    current_pts = opt_pts
                    update_best(opt_pts)
                else:
                    r_high = r_target
            except Exception:
                r_high = r_target
        
        # Final optimization at r_low
        r_target = r_low
        x0 = current_pts.flatten()
        try:
            res = minimize(_violation_obj_and_grad, x0, args=(r_target,), jac=True,
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = res.x.reshape(N, 2)
            update_best(opt_pts)
            current_pts = opt_pts
        except Exception:
            pass
    
    # Soft-min optimization with increasing beta
    if time.time() - start_time < DEADLINE - 15 and best_pts is not None:
        for beta in [30, 60, 100, 150]:
            if time.time() - start_time > DEADLINE - 15:
                break
            x0 = best_pts.flatten()
            try:
                def obj(x):
                    return _soft_min_objective(x, beta)
                res = minimize(obj, x0, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
                opt_pts = res.x.reshape(N, 2)
                # Project inside
                dists = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
                mask = dists > 1.0
                if np.any(mask):
                    opt_pts[mask] = opt_pts[mask] / dists[mask][:, None] * 0.9999
                update_best(opt_pts)
            except Exception:
                pass
    
    # Final polish: local refinement with small perturbations
    if time.time() - start_time < DEADLINE - 5 and best_pts is not None:
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbed = best_pts + np.random.normal(0, 0.002, (N, 2))
            # Project inside
            dists = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = dists > 1.0
            if np.any(mask):
                perturbed[mask] = perturbed[mask] / dists[mask][:, None] * 0.9999
            
            r_target = _score(perturbed)
            if r_target <= 0:
                continue
            
            x0 = perturbed.flatten()
            try:
                res = minimize(_violation_obj_and_grad, x0, args=(r_target,), jac=True,
                               method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                opt_pts = res.x.reshape(N, 2)
                dists_f = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
                mask_f = dists_f > 1.0
                if np.any(mask_f):
                    opt_pts[mask_f] = opt_pts[mask_f] / dists_f[mask_f][:, None] * 0.99999
                update_best(opt_pts)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _ring_init(N)
    
    return best_pts
