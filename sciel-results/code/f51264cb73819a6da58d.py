"""Packing 50 equal circles in unit disk - Combined structural approach.

Strategy:
1. Global search: Differential Evolution on ring parameters (radii + per-point angles)
   This allows the structure to deviate from perfect symmetry if beneficial.
2. Local refinement: Multi-stage annealing with L-BFGS-B on full point coordinates
   using a smooth soft-min objective.
3. Final high-precision polish: Binary search radius with constraint projection.

Key structural change from parent: DE operates on individual angles (not just ring
offsets), giving much richer search space while maintaining ring structure for
good initialization.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_MARGIN = 15.0


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    return val if val > 0 else -1.0


def _softmin_logsumexp(values, beta):
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
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(params, ring_counts):
    """Convert DE parameters to point coordinates.
    
    params layout: [r1, r2, ..., r_{num_rings}, angle1_1, angle1_2, ..., angle_{last}_{last}]
    Ring 0 is always at center (1 point). Remaining rings have their radii and
    individual angles as parameters.
    """
    num_rings = len(ring_counts)
    # Radii for rings 1..num_rings-1
    radii = params[:num_rings - 1]
    # Angles for all points in rings 1..num_rings-1
    angles_flat = params[num_rings - 1:]
    
    points = [np.array([0.0, 0.0])]
    angle_idx = 0
    for i in range(1, num_rings):
        count = ring_counts[i]
        r = radii[i - 1]
        angles = angles_flat[angle_idx:angle_idx + count]
        angle_idx += count
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        for j in range(count):
            points.append(np.array([xs[j], ys[j]]))
    
    pts = np.array(points[:N])
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _anneal_points(pts, deadline, max_betas=10):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    
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
                options={'maxiter': 1200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 40000}
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=300):
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
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 3e-4
    
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    
    # Final L-BFGS polish at the best found radius
    if time.time() < deadline - 1:
        try:
            r_target = _score(current)
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(10000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 15000}
            )
            final_pts = res.x.reshape(N, 2)
            if np.all(np.isfinite(final_pts)):
                final_score = _score(final_pts)
                if final_score > _score(current):
                    current = final_pts
        except Exception:
            pass
    
    return current, _score(current)


def _de_search(ring_counts, deadline, pop_size=30, max_gen=40, F=0.8, CR=0.9, seed=42):
    """Differential evolution on ring parameters with individual angles."""
    num_rings = len(ring_counts)
    # Parameter count: (num_rings-1) radii + sum(ring_counts[1:]) angles
    num_angles = sum(ring_counts[1:])
    total_params = (num_rings - 1) + num_angles
    
    # Bounds
    bounds = []
    # Ring radii (rings 1..num_rings-1), roughly ordered
    for i in range(1, num_rings):
        # Estimate reasonable range based on ring index
        lo = 0.12 * i
        hi = 0.85 * (i / (num_rings - 1)) + 0.12
        hi = min(hi, 0.95)
        lo = max(lo, 0.04)
        bounds.append((lo, hi))
    # Angles
    for _ in range(num_angles):
        bounds.append((0.0, 2.0 * np.pi))
    
    rng = np.random.RandomState(seed)
    
    def eval_params(params):
        pts = _ring_to_points(params, ring_counts)
        return -_score(pts)
    
    # Initial population - mix of uniform and structured starts
    pop = np.zeros((pop_size, total_params))
    for i in range(pop_size):
        for j in range(total_params):
            if j < num_rings - 1:
                # Radii: sometimes use structured initialization
                if i % 3 == 0:
                    # Structured: evenly spaced
                    pop[i, j] = 0.85 * (j + 1) / (num_rings - 1) + rng.uniform(-0.05, 0.05)
                    pop[i, j] = np.clip(pop[i, j], bounds[j][0], bounds[j][1])
                else:
                    pop[i, j] = rng.uniform(bounds[j][0], bounds[j][1])
            else:
                # Angles: sometimes use symmetric initialization
                if i % 3 == 0:
                    ring_idx = j - (num_rings - 1)
                    # Find which ring this angle belongs to
                    cumulative = 0
                    for r in range(1, num_rings):
                        if ring_idx < cumulative + ring_counts[r]:
                            pos_in_ring = ring_idx - cumulative
                            count = ring_counts[r]
                            pop[i, j] = 2.0 * np.pi * pos_in_ring / count + rng.uniform(-0.1, 0.1)
                            break
                        cumulative += ring_counts[r]
                    pop[i, j] = np.clip(pop[i, j], bounds[j][0], bounds[j][1])
                else:
                    pop[i, j] = rng.uniform(bounds[j][0], bounds[j][1])
    
    scores = np.array([eval_params(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 10:
            break
        
        for i in range(pop_size):
            if time.time() > deadline - 10:
                break
            
            # Select three distinct random individuals
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = rng.choice(candidates, 3, replace=False)
            
            # Mutation with adaptive F
            f_adaptive = F * (1.0 - 0.3 * gen / max_gen)
            mutant = pop[a] + f_adaptive * (pop[b] - pop[c])
            
            # Clamp to bounds
            for k in range(total_params):
                mutant[k] = np.clip(mutant[k], bounds[k][0], bounds[k][1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(total_params)
            for j in range(total_params):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Selection
            trial_score = eval_params(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_params = trial.copy()
    
    best_pts = _ring_to_points(best_params, ring_counts)
    return best_pts, -best_score


def _hexagonal_init(scale=0.12):
    """Generate hexagonal lattice points and filter to fit in unit disk."""
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = -1.0
    
    # Standard ring topology for 50 circles: 1 + 6 + 12 + 18 + 13
    ring_counts = [1, 6, 12, 18, 13]
    
    # ---- Strategy 1: DE on ring parameters (individual angles) with multiple seeds ----
    if time.time() < deadline - 50:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 45:
                break
            de_pts, de_score = _de_search(ring_counts, deadline, seed=seed)
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts.copy()
    
    # ---- Strategy 2: Ring initialization with annealing - more diverse configs ----
    ring_configs = [
        [0.0, 0.25, 0.45, 0.65, 0.85],
        [0.0, 0.24, 0.44, 0.64, 0.84],
        [0.0, 0.26, 0.46, 0.66, 0.86],
        [0.0, 0.25, 0.44, 0.63, 0.83],
        [0.0, 0.245, 0.445, 0.645, 0.845],
        [0.0, 0.255, 0.455, 0.655, 0.855],
        [0.0, 0.23, 0.43, 0.63, 0.83],
        [0.0, 0.27, 0.47, 0.67, 0.87],
    ]
    
    for config in ring_configs:
        if time.time() > deadline - 35:
            break
        
        # Build points from ring config
        points = [np.array([0.0, 0.0])]
        for i in range(1, len(ring_counts)):
            count = ring_counts[i]
            r = config[i]
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            for j in range(count):
                points.append(np.array([xs[j], ys[j]]))
        
        pts = np.array(points[:N])
        refined_pts, refined_score = _anneal_points(pts, deadline)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # ---- Strategy 3: Hexagonal initialization with different scales ----
    if time.time() < deadline - 25:
        for scale in [0.115, 0.12, 0.125]:
            if time.time() > deadline - 20:
                break
            hex_pts = _hexagonal_init(scale=scale)
            if hex_pts is not None and len(hex_pts) == N:
                refined_pts, refined_score = _anneal_points(hex_pts, deadline)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_pts.copy()
    
    # ---- Strategy 4: Random restarts with more trials ----
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 8:
                break
            rng = np.random.RandomState(trial * 777 + 13)
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0, 0.9, N)
            pts_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            refined_pts, refined_score = _anneal_points(pts_rand, deadline)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    # ---- Final polish: binary search with constraint projection ----
    if best_points is not None and time.time() < deadline - 5:
        polished_pts, polished_score = _binary_search_polish(best_points, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
        
        # Additional polish pass if time permits
        if time.time() < deadline - 3:
            polished_pts2, polished_score2 = _binary_search_polish(best_points, deadline)
            if polished_score2 > best_score:
                best_score = polished_score2
                best_points = polished_pts2
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        points = [np.array([0.0, 0.0])]
        for i in range(1, len(ring_counts)):
            count = ring_counts[i]
            r = 0.25 + 0.2 * i
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            for j in range(count):
                points.append(np.array([xs[j], ys[j]]))
        best_points = np.array(points[:N])
        best_score = _score(best_points)
    
    return best_points
