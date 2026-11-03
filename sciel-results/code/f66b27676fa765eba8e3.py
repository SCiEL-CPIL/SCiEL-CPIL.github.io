"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + gradient polish + binary search radius."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if pts is None or len(pts) != N:
        return -np.inf
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -np.inf
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -np.inf
    return val


def _compute_score_fast(pts, moved_idx=None):
    """Faster score computation when only one point moved. Returns (score, min_pair, min_container)."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf, np.inf, np.inf
    
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_container, min_pair), min_pair, min_container


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    """Smooth objective for optimization: minimize negative soft-min of margins."""
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    # Combine all constraints
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
    
    return P


def _parametric_score(params, ring_counts):
    """Evaluate the parametric ring configuration."""
    P = _get_ring_initialization(params, ring_counts)
    return _compute_score(P)


def _optimize_parametric_ring(ring_counts, seed=42, maxiter=60, deadline=None):
    """
    Optimize ring radii and offsets using differential evolution.
    """
    k = len(ring_counts)
    # Bounds: radii in [0, 0.95], offsets in [0, 2*pi]
    bounds = [(0.0, 0.95)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        if deadline is not None and time.time() > deadline:
            return 0.0  # Infeasible
        s = _parametric_score(params, ring_counts)
        return -s
        
    try:
        res = differential_evolution(neg_score, bounds, maxiter=maxiter, popsize=15, 
                                     tol=1e-6, seed=seed, polish=False, 
                                     workers=1, init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        return best_P, -res.fun
        
    except Exception:
        # Fallback: try a few random configurations
        rng = np.random.default_rng(seed)
        best_P = None
        best_score = -np.inf
        for _ in range(20):
            params = np.array([rng.uniform(0.1, 0.9) for _ in range(k)] + 
                             [rng.uniform(0, 2*np.pi) for _ in range(k)])
            P = _get_ring_initialization(params, ring_counts)
            s = _compute_score(P)
            if s > best_score:
                best_score = s
                best_P = P
        if best_P is None:
            best_P = _get_hex_initial_points()
            best_score = _compute_score(best_P)
        return best_P, best_score


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    points = []
    # Generate hexagonal lattice
    scale = 0.12
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:  # Keep points well inside
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        # Last resort: grid
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * 0.2
                y = j * 0.2
                if x*x + y*y <= 0.85:
                    points.append([x, y])
        points = np.array(points[:N])
        return points
    
    # Sort by distance from center and take first 50
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _anneal_points(pts, deadline, max_betas=12):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _compute_score(current)
    
    # Progressive beta values for annealing
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    betas = betas[:max_betas]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000, 
                    'ftol': 1e-16, 
                    'gtol': 1e-14,
                    'maxfun': 30000
                }
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _compute_score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        
        if min_pair < min_bound - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _compute_score(current) - 1e-8
    high = _compute_score(current) + 1e-4
    
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _compute_score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    return current, _compute_score(current)


def _local_search(pts, n_iters=2000, seed=12345, deadline=None):
    """
    Local search: perturb one point at a time, accept if score improves.
    Uses adaptive step size and temperature annealing.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    step_size = 0.01
    temperature = 0.003
    no_improve_count = 0
    max_no_improve = 100
    best_step = step_size
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.995:
            new_pos *= 0.995 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score, _, _ = _compute_score_fast(test_pts, moved_idx=idx)
        
        delta = new_score - best_score
        if delta > 0 or (temperature > 1e-8 and rng.random() < np.exp(delta / temperature)):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
                no_improve_count = 0
                best_step = step_size
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Reset and adjust
            temperature = max(temperature * 0.5, 1e-8)
            step_size = min(step_size * 1.3, 0.04)
            no_improve_count = 0
        
        # Gentle annealing
        temperature *= 0.9998
        step_size = max(0.001, step_size * 0.99995)
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5s buffer before 120s limit
    
    # Define ring structures to try
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 17, 14],
        [1, 6, 11, 13, 19],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # Try each ring structure with parametric optimization
    for ring_counts in ring_structures:
        if time.time() > deadline - 40.0:
            break
        
        # 1. Global parametric search for ring radii and offsets
        P_candidate, score_candidate = _optimize_parametric_ring(
            ring_counts, seed=42, maxiter=50, deadline=deadline - 20.0)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Annealing refinement on the candidate
        if time.time() > deadline - 30.0:
            break
        
        P_annealed, score_annealed = _anneal_points(P_candidate, deadline=deadline - 15.0, max_betas=8)
        
        if score_annealed > score_best:
            P_best = P_annealed
            score_best = score_annealed
    
    # Final binary search polish with constraint projection
    if time.time() < deadline - 15.0:
        P_polished, score_polished = _binary_search_polish(P_best, deadline=deadline - 5.0)
        
        if score_polished > score_best:
            P_best = P_polished
            score_best = score_polished
    
    # Final local search refinement - use remaining time
    if time.time() < deadline - 10.0:
        remaining_time = deadline - 10.0 - time.time()
        remaining_iters = min(8000, int(remaining_time * 800))
        remaining_iters = max(1000, remaining_iters)
        P_best, score_best = _local_search(P_best, n_iters=remaining_iters, seed=99999, deadline=deadline - 5.0)
    
    # Final high-precision annealing polish
    if time.time() < deadline - 5.0:
        P_final, score_final = _anneal_points(P_best, deadline=deadline - 2.0, max_betas=6)
        
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
    
    # Validate and ensure all points are strictly inside
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    else:
        # Push any points that are too close to the boundary inward
        for i in range(N):
            r = radii[i]
            if r > 0.999:
                P_best[i] *= 0.999 / r
    
    return P_best
