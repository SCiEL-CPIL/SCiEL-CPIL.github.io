"""Combined packing: hexagonal lattice initialization + multi-stage smooth annealing + constraint push."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _hex_init(n):
    """Generate n points using a hexagonal lattice pattern, scaled to fit in unit circle."""
    if n <= 0:
        return np.empty((0, 2))
    if n == 1:
        return np.array([[0.0, 0.0]])
    
    # Generate hexagonal lattice points
    points = []
    # For a hexagonal packing with spacing s, points are at:
    # (i*s, j*s*sqrt(3)) and ((i+0.5)*s, (j+0.5)*s*sqrt(3)) for integers i, j
    # We'll generate a large hexagonal region and pick the n closest to origin
    
    max_radius = 2.0  # Generate points well beyond unit circle
    s = 0.25  # Initial spacing guess
    
    # Generate hexagonal grid
    i_max = int(max_radius / s) + 2
    j_max = int(max_radius / (s * np.sqrt(3))) + 2
    
    candidates = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            # Two sets of points for hexagonal packing
            x1 = i * s
            y1 = j * s * np.sqrt(3)
            x2 = (i + 0.5) * s
            y2 = (j + 0.5) * s * np.sqrt(3)
            
            r1 = np.hypot(x1, y1)
            r2 = np.hypot(x2, y2)
            
            if r1 < max_radius:
                candidates.append((r1, x1, y1))
            if r2 < max_radius:
                candidates.append((r2, x2, y2))
    
    # Sort by distance from origin and take n closest
    candidates.sort(key=lambda x: x[0])
    points = np.array([[c[1], c[2]] for c in candidates[:n]])
    
    # Scale to fit in unit circle with margin
    max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_r > 0.9:
        points *= 0.9 / max_r
    
    return points


def _score(points):
    """Compute the packing score: min of half-min-pairwise-dist and min-boundary-dist."""
    if len(points) != N or points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Compute soft-min using log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """
    Smooth approximation of the negative score.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalently, minimize -r.
    Use soft-min: -softmin(dists/2, boundary_dists, beta)
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii
    
    all_dists = np.concatenate([dists[np.triu_indices(N, k=1)], boundary])
    all_dists = all_dists[np.isfinite(all_dists)]
    
    if len(all_dists) == 0:
        return 1e10
    
    soft_min = _softmin_logsumexp(all_dists, beta)
    return -soft_min


def _smooth_grad(x, beta=50.0):
    """
    Compute gradient of the smooth objective w.r.t. points.
    Uses analytical gradient of the soft-min.
    """
    pts = x.reshape(N, 2)
    
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_half = dists / 2.0
    
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = pair_half[iu]
    
    all_dists = np.concatenate([pair_dists, boundary])
    all_dists = np.maximum(all_dists, -1.0)
    
    m = np.min(all_dists)
    if m < -0.5:
        return np.zeros_like(x)
    
    exp_terms = np.exp(-beta * (all_dists - m))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Boundary part gradient
    w_boundary = weights[N:]
    for i in range(N):
        if radii[i] > 1e-10:
            grad[i] -= w_boundary[i] * beta * pts[i] / radii[i]
    
    # Pair part gradient
    w_pair = weights[:len(pair_dists)]
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = w_pair[idx]
        if dists[i, j] > 1e-10:
            direction = (pts[i] - pts[j]) / dists[i, j]
            grad[i] += w * beta * 0.5 * direction
            grad[j] -= w * beta * 0.5 * direction
    
    return grad.flatten()


def _anneal_points(pts, deadline):
    """Multi-stage smooth annealing using L-BFGS-B."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Multi-stage annealing: start with low beta (smooth), increase to high beta (sharp)
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0, 1500.0, 3000.0, 5000.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                jac=_smooth_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            
            # Project back to unit circle
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            for i in range(N):
                if radii[i] > 0.999:
                    new_pts[i] *= 0.999 / radii[i]
            
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _constraint_push(pts, deadline, max_iters=300):
    """
    Constraint-based local search: iteratively push the most constrained points apart.
    """
    pts = pts.copy()
    current_score = _score(pts)
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_margins = dists / 2.0
        boundary_margins = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            # Push the closest pair apart
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            if i > j:
                i, j = j, i
            
            direction = pts[i] - pts[j]
            dist = np.linalg.norm(direction)
            if dist < 1e-15:
                break
            direction /= dist
            
            # Push apart by a small amount
            push = 0.002
            pts[i] += push * direction
            pts[j] -= push * direction
        else:
            # Push the point closest to boundary inward
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                direction = pi / dist
                push = 0.002
                pts[i] -= push * direction
        
        # Project to unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        for k in range(N):
            if radii[k] > 0.999:
                pts[k] *= 0.999 / radii[k]
        
        new_score = _score(pts)
        if new_score > current_score:
            current_score = new_score
        else:
            # Revert if score didn't improve
            break
    
    return pts, current_score


def construct():
    start_time = time.time()
    
    # Initial points using hexagonal lattice
    points = _hex_init(N).astype(np.float64)
    best_points = points.copy()
    best_score = _score(points)
    
    # Multi-stage smooth annealing
    if time.time() - start_time < DEADLINE - 30:
        points, score = _anneal_points(points, DEADLINE - 20)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Constraint-based local search
    if time.time() - start_time < DEADLINE - 10:
        points, score = _constraint_push(points, DEADLINE - 5)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final polish: a few more constraint push iterations
    if time.time() - start_time < DEADLINE - 3:
        points, score = _constraint_push(points, DEADLINE - 1, max_iters=100)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Ensure validity: no point outside unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    for i in range(N):
        if radii[i] > 0.9999:
            best_points[i] *= 0.9999 / radii[i]
    
    return best_points
