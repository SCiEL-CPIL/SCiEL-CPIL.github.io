"""Packing 50 equal circles in a unit circle using smooth optimization and simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score_vec(points):
    """Vectorized score computation."""
    points = np.asarray(points).reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration, picking best spacing."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _smooth_objective(params, b=50.0):
    """
    Smooth surrogate for maximizing the minimum of boundary and pairwise distances.
    Uses log-sum-exp to approximate the min function smoothly for L-BFGS-B.
    """
    pts = params.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise half-distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([boundary, pair_dists])
    
    # Smooth min approximation: min(x) ≈ -1/b * log(sum(exp(-b*x)))
    # To avoid overflow, shift by max
    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    if not np.isfinite(max_neg):
        return 1e10
    log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    smooth_min = -log_sum / b
    
    # We want to maximize smooth_min, so minimize -smooth_min
    return -smooth_min


def _local_optimize(points, maxiter=200, b=50.0):
    """
    Refine points using L-BFGS-B with the smooth objective.
    """
    x0 = points.flatten()
    
    def objective(x):
        return _smooth_objective(x, b=b)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        cand = res.x.reshape(-1, 2)
        # Ensure validity: project back if outside
        if np.all(np.isfinite(cand)):
            norms = np.linalg.norm(cand, axis=1)
            for i in range(len(cand)):
                if norms[i] > 1.0:
                    cand[i] = cand[i] / norms[i] * 0.999
            return cand
    except Exception:
        pass
    return points.copy()


def _anneal(points, start_time, deadline, T_start=0.02, T_min=0.001, cooling=0.999):
    """
    Simulated annealing to escape local minima.
    Uses the smooth objective for acceptance decisions? No, use exact score for best tracking,
    but maybe smooth for gradient-like moves? 
    Actually, standard SA uses the exact score (min radius) for the Metropolis criterion.
    However, the score is non-differentiable. 
    We will use the exact score for acceptance.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    current_score = best_score
    
    T = T_start
    step = 0
    
    while time.time() < deadline - 2.0:
        step += 1
        
        # Generate perturbation
        # Move a random subset of points
        n_move = np.random.randint(1, 6)
        idx = np.random.choice(N, n_move, replace=False)
        
        # Scale perturbation by temperature
        delta = np.random.normal(0, T * 0.1, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        # Hard constraint: keep inside circle for validity check in score
        # But for SA, we can allow temporary violations if penalized?
        # The score function returns min(...), so if a point is outside, 
        # boundary dist is negative, score drops. This is a natural penalty.
        
        new_score = _score_vec(new_pts)
        
        # Metropolis criterion
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            prob = np.exp((new_score - current_score) / T)
            if np.random.random() < prob:
                pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Periodic local optimization
        if step % 200 == 0 and time.time() < deadline - 10.0:
            opt_pts = _local_optimize(pts, maxiter=50, b=100.0)
            opt_score = _score_vec(opt_pts)
            if opt_score > current_score:
                pts = opt_pts
                current_score = opt_score
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
            # Reheat slightly
            T = max(T, 0.005)

    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # 1. Hexagonal initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # 2. Initial local optimization
    if time.time() < start_time + DEADLINE - 5:
        opt_points = _local_optimize(points, maxiter=200, b=50.0)
        opt_score = _score_vec(opt_points)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # 3. Simulated Annealing
    # We use the current best as the starting point for SA
    sa_points, sa_score = _anneal(best_points, start_time, start_time + DEADLINE - 5)
    if sa_score > best_score:
        best_score = sa_score
        best_points = sa_points.copy()
    
    # 4. Final multi-start local optimization with perturbations
    # Use remaining time for local refinements from the best point
    remaining_time = (start_time + DEADLINE - 2) - time.time()
    if remaining_time > 5:
        # Try a few restarts with small perturbations
        n_restarts = int(remaining_time / 5)  # estimate ~5s per restart
        n_restarts = max(1, min(n_restarts, 10))
        
        for i in range(n_restarts):
            if time.time() > start_time + DEADLINE - 3:
                break
            
            if i == 0:
                start_pts = best_points.copy()
            else:
                # Perturb the best point
                scale = 0.01 / (1 + i * 0.5)
                start_pts = best_points + np.random.randn(N, 2) * scale
            
            opt_pts = _local_optimize(start_pts, maxiter=100, b=100.0)
            opt_score = _score_vec(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
    
    # 5. Final high-precision polish
    if time.time() < start_time + DEADLINE - 1:
        final_pts = _local_optimize(best_points, maxiter=500, b=200.0)
        final_score = _score_vec(final_pts)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score
    
    return best_points
