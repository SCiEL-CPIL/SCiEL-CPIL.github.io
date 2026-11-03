import numpy as np
import time
from scipy.optimize import minimize

N = 50
SEED = 42
DEADLINE_MARGIN = 15.0

def _pairwise_distances(points):
    """Compute pairwise squared distances for an (N, 2) array."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=-1)

def _score(points):
    """Compute the packing radius for given points."""
    if len(points) < 2:
        return 0.0
    dist_sq = _pairwise_distances(points)
    # Mask diagonal
    mask = ~np.eye(len(points), dtype=bool)
    min_dist = np.sqrt(np.min(dist_sq[mask]))
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _softmin_logsumexp(values, beta):
    """Smooth approximation of min(values)."""
    m = np.max(values)
    if not np.isfinite(m):
        return -np.inf
    y = -beta * values
    m_y = np.max(y)
    if not np.isfinite(m_y):
        return -np.inf
    log_sum_exp = m_y + np.log(np.sum(np.exp(y - m_y)))
    return -log_sum_exp / beta

def _smooth_objective(x, beta):
    """
    Objective for local polish.
    Minimize negative smooth min of distances and boundary constraints.
    """
    points = x.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    dist_sq = _pairwise_distances(points)
    mask = ~np.eye(N, dtype=bool)
    # We need min of (distances/2) and (1 - radii)
    # dist_sq[mask] / 4 is the squared value of (dist/2)^2.
    # But smooth min is on the values themselves.
    # Let's use distances directly for the smooth min to be consistent with the radius definition.
    pair_dists = np.sqrt(dist_sq[mask]) / 2.0
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    r_smooth = _softmin_logsumexp(all_vals, beta)
    return -r_smooth

def _ring_to_points(params):
    """
    Convert 8D parametric ring model to (N, 2) points.
    params: [r1, theta1, r2, theta2, r3, theta3, r4, theta4]
    """
    r1, th1, r2, th2, r3, th3, r4, th4 = params
    points = []
    # Ring 1: 1 center
    points.append([0.0, 0.0])
    # Ring 2: 6 points
    for i in range(6):
        theta = th1 + i * (2 * np.pi / 6)
        points.append([r1 * np.cos(theta), r1 * np.sin(theta)])
    # Ring 3: 12 points
    for i in range(12):
        theta = th2 + i * (2 * np.pi / 12)
        points.append([r2 * np.cos(theta), r2 * np.sin(theta)])
    # Ring 4: 12 points
    for i in range(12):
        theta = th3 + i * (2 * np.pi / 12)
        points.append([r3 * np.cos(theta), r3 * np.sin(theta)])
    # Ring 5: 19 points (50 - 1 - 6 - 12 - 12 = 19)
    # The last ring is irregular. We distribute 19 points evenly.
    for i in range(19):
        theta = th4 + i * (2 * np.pi / 19)
        points.append([r4 * np.cos(theta), r4 * np.sin(theta)])
    
    return np.array(points)

def _hexagonal_init():
    """Generate a hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points_hex = np.array(rows)
    radius_hex = np.hypot(points_hex[:, 0], points_hex[:, 1])
    points_hex = points_hex[radius_hex <= 1.0 - spacing / 2.0]
    if len(points_hex) >= N:
        points_hex = points_hex[np.argsort(radius_hex, kind="stable")[:N]]
    else:
        points_hex = np.random.uniform(-0.8, 0.8, (N, 2))
    return points_hex

def _anneal_points(points, start_time, deadline_margin):
    """
    Perform simulated annealing on the points.
    """
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.995
    
    while T > T_min:
        if time.time() > time.time() - start_time + deadline_margin:
            break
            
        # Pick a random point and perturb it
        idx = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_point = current[idx] + delta
        
        # Keep within bounds (roughly)
        new_point = np.clip(new_point, -1.0, 1.0)
        
        new_points = current.copy()
        new_points[idx] = new_point
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta_score = new_score - current_score
            if delta_score < 0:
                accept_prob = np.exp(delta_score / T)
                if np.random.rand() < accept_prob:
                    current = new_points
                    current_score = new_score
        
        T *= cooling_rate
        
    return best, best_score

def _constraint_push(points):
    """
    Push points that are outside the unit circle or overlapping back inside/separated.
    This is a simple heuristic repair.
    """
    points = points.copy()
    for _ in range(100):
        # Check boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            # Move outside points towards center
            for i in np.where(outside)[0]:
                norm = np.linalg.norm(points[i])
                if norm > 0:
                    points[i] = points[i] * (1.0 / norm) * 0.99
        
        # Check overlaps (simplified: just check a few random pairs)
        # A full check is O(N^2), which is 2500, so it's fine.
        dist_sq = _pairwise_distances(points)
        mask = ~np.eye(N, dtype=bool)
        min_dist_sq = np.min(dist_sq[mask])
        # If min distance is too small (less than 2*r^2?), we push them apart.
        # We don't know r exactly, but we can push if dist < 0.2 (arbitrary threshold)
        # Better: push if dist < 2 * current_estimated_r
        # Let's just push if dist < 0.1 for now, or use a fixed small value.
        # Actually, let's just ensure no points are exactly coincident.
        # A more robust push:
        # For each pair with dist < 0.1, push them apart.
        
        # This is getting complex. Let's stick to boundary push and rely on optimizer for separation.
        # But if points are coincident, the gradient is undefined.
        # Let's add a small random jitter to coincident points.
        coincident = (dist_sq < 1e-8) & mask
        if np.any(coincident):
            indices = np.where(coincident)
            for i, j in zip(indices[0], indices[1]):
                if i < j:
                    # Push i and j apart
                    mid = (points[i] + points[j]) / 2.0
                    diff = points[i] - points[j]
                    if np.linalg.norm(diff) < 1e-6:
                        diff = np.random.normal(0, 1, 2)
                    diff = diff / np.linalg.norm(diff)
                    points[i] = mid + diff * 0.01
                    points[j] = mid - diff * 0.01
    
    return points

def _binary_search_polish(points, start_time, deadline_margin):
    """
    Binary search for the maximum radius r such that a feasible packing exists.
    We use the smooth objective to check feasibility for a given r.
    """
    # This is complex to implement correctly with the smooth objective.
    # Instead, we will just run the smooth polish with increasing beta.
    return points

def _optimize_from_init(points, start_time, deadline_margin):
    """
    Run local optimization (L-BFGS-B) with smooth objective.
    """
    current = points.flatten()
    best_score = _score(points)
    best_points = points.copy()
    
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > deadline_margin:
            break
            
        try:
            res = minimize(
                _smooth_objective,
                current,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -best_score:
                new_radius = -res.fun
                new_points = res.x.reshape(N, 2)
                exact_radius = _score(new_points)
                if exact_radius > best_score:
                    best_score = exact_radius
                    best_points = new_points
                    current = res.x
        except Exception:
            pass
    
    return best_points, best_score

def _de_ring_optimize(start_time, deadline_margin):
    """
    Use Differential Evolution on the 8D ring parameter space.
    """
    from scipy.optimize import differential_evolution
    
    bounds = [
        (0.0, 0.2),  # r1 (ring 2 radius)
        (0.0, 2 * np.pi),  # th1
        (0.0, 0.4),  # r2 (ring 3 radius)
        (0.0, 2 * np.pi),  # th2
        (0.0, 0.6),  # r3 (ring 4 radius)
        (0.0, 2 * np.pi),  # th3
        (0.0, 0.8),  # r4 (ring 5 radius)
        (0.0, 2 * np.pi),  # th4
    ]
    
    def objective(params):
        points = _ring_to_points(params)
        # Basic validity check
        if not np.all(np.isfinite(points)):
            return 1e10
        # Check if points are inside unit circle (loose check for DE)
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.max(radii) > 1.0:
            return 1e10 - (1.0 - np.max(radii)) * 100 # Penalty for being outside
        r = _score(points)
        return -r
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=SEED,
            popsize=20,
            maxiter=100,
            tol=0.01,
            polish=False
        )
        
        if de_result.success or de_result.fun < -1e10:
            best_params = de_result.x
            best_points = _ring_to_points(best_params)
            best_score = _score(best_points)
            return best_points, best_score
    except Exception:
        pass
    
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # 1. Hexagonal initialization
    points_hex = _hexagonal_init()
    best_points = points_hex.copy()
    best_score = _score(best_points)
    
    # 2. Ring-based DE
    if time.time() < deadline - 30:
        ring_points, ring_score = _de_ring_optimize(start_time, deadline)
        if ring_points is not None and ring_score > best_score:
            best_points = ring_points
            best_score = ring_score
    
    # 3. Local Polish (L-BFGS-B)
    if time.time() < deadline - 10:
        polished_points, polished_score = _optimize_from_init(best_points, start_time, deadline)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    # 4. Simulated Annealing (if time permits)
    if time.time() < deadline - 5:
        annealed_points, annealed_score = _anneal_points(best_points, start_time, deadline)
        if annealed_score > best_score:
            best_points = annealed_points
            best_score = annealed_score
    
    # 5. Final Polish
    if time.time() < deadline - 2:
        final_points, final_score = _optimize_from_init(best_points, start_time, deadline)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Ensure validity
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.max(radii) > 1.0:
        # Scale down to fit
        scale = 1.0 / np.max(radii) * 0.999
        best_points = best_points * scale
    
    return best_points
