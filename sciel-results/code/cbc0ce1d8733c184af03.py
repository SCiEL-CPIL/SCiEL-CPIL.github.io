"""Combined packing: hexagonal seed + L-BFGS-B potential optimization + simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 142.0  # safety margin under 150s wall-clock


def _radius(pos):
    """Score of a configuration (double precision)."""
    pos = np.asarray(pos, dtype=np.float64)
    # Wall margins: distance from each centre to nearest wall
    wall_margins = 0.5 - np.abs(pos)  # (N, 2)
    min_wall = wall_margins.min()
    # Pairwise distances
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    r = min(min_wall, 0.5 * min_pair)
    return float(r)


def _potential(x):
    """Potential to minimize: sum of inverse 4th powers of wall and pair distances."""
    centers = x.reshape(-1, 2)
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, 1e10)
    min_pair_dists = np.min(dists, axis=1)
    
    eps = 1e-6
    wall_safe = np.clip(wall_dists, eps, None)
    pair_safe = np.clip(min_pair_dists, eps, None)
    
    V = np.sum(1.0 / (wall_safe ** 4)) + np.sum(1.0 / (pair_safe ** 4))
    return V


def _hexagonal_seed(rng):
    """Generate a hexagonal lattice seed, clipped to the square."""
    # Estimate spacing for 50 circles in unit square
    d = 0.145
    pts = []
    range_x = int(np.ceil(1.0 / d)) + 2
    range_y = int(np.ceil(1.0 / (d * np.sqrt(3) / 2))) + 2
    
    for j in range(-range_y, range_y + 1):
        for i in range(-range_x, range_x + 1):
            x = i * d + (j % 2) * d / 2
            y = j * d * np.sqrt(3) / 2
            x -= 0.5
            y -= 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    
    pts = np.array(pts, dtype=np.float64)
    
    if len(pts) > N:
        dists_to_center = np.sum(pts ** 2, axis=1)
        idx = np.argsort(dists_to_center)
        pts = pts[idx[:N]]
    elif len(pts) < N:
        extra = rng.uniform(-0.45, 0.45, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    pts += rng.normal(0, 0.003, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts


def _lbfgs_optimize(pos, deadline):
    """Run L-BFGS-B on the potential."""
    x0 = pos.flatten()
    try:
        res = minimize(_potential, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16})
        x_opt = res.x
        centers = x_opt.reshape(-1, 2)
        centers = np.clip(centers, -0.5, 0.5)
        return centers
    except Exception:
        return pos


def _anneal(pos, iters, t0, tmin, deadline):
    """Simulated annealing on the radius score."""
    pos = np.array(pos, dtype=np.float64)
    r = _radius(pos)
    best = (r, pos.copy())
    temp = t0
    rng = np.random.default_rng()
    
    for it in range(iters):
        if time.time() > deadline - 2.0:
            break
        temp = max(tmin, temp * 0.999)
        delta = 0.003 * rng.standard_normal((N, 2))
        cand = pos + delta
        cand = np.clip(cand, -0.5, 0.5)
        rc = _radius(cand)
        if rc >= r or (temp > 0 and rng.random() < np.exp((rc - r) / max(temp, 1e-9))):
            pos, r = cand, rc
            if r > best[0]:
                best = (r, pos.copy())
    return best


def _hard_sphere_relax(positions, r_target, max_iters=2000, deadline=None):
    """
    Hard-sphere relaxation: push overlapping circles apart until all are separated
    by at least 2*r_target, while keeping them inside the square.
    Returns (positions, converged).
    """
    positions = np.array(positions, dtype=np.float64)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        max_overlap = 0.0
        # For each pair, if overlapping, push apart
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                min_dist = 2 * r_target
                if dist < min_dist and dist > 1e-12:
                    push = (min_dist - dist) / (2 * dist)
                    positions[i] += push * d
                    positions[j] -= push * d
                    max_overlap = max(max_overlap, min_dist - dist)
                elif dist <= 1e-12:
                    positions[i][0] += 1e-4
                    positions[j][0] -= 1e-4
                    max_overlap = max(max_overlap, min_dist)
        # Keep inside box
        for i in range(N):
            for k in range(2):
                if positions[i][k] < -0.5 + r_target:
                    positions[i][k] = -0.5 + r_target
                elif positions[i][k] > 0.5 - r_target:
                    positions[i][k] = 0.5 - r_target
        if max_overlap < 1e-8:
            return positions, True
    return positions, False


def _grow_radius(positions, r_start, deadline):
    """
    Gradually increase radius and relax positions to find a feasible packing.
    Returns (best_positions, best_r).
    """
    best_r = r_start
    best_pos = positions.copy()
    r = r_start
    max_grow_iters = 200
    
    for grow_iter in range(max_grow_iters):
        if time.time() > deadline - 5.0:
            break
        # Try to increase r slightly
        r_new = r * 1.003 + 1e-7
        pos_new, converged = _hard_sphere_relax(positions, r_new, max_iters=1000, deadline=deadline)
        if converged:
            r = r_new
            positions = pos_new
            best_r = r
            best_pos = positions.copy()
        else:
            # Try smaller increment
            r_new = r * 1.001 + 1e-8
            pos_new, converged = _hard_sphere_relax(positions, r_new, max_iters=500, deadline=deadline)
            if converged:
                r = r_new
                positions = pos_new
                best_r = r
                best_pos = positions.copy()
            else:
                break
    return best_pos, best_r


def _local_polish(positions, r, max_iters=500, deadline=None):
    """
    Local polish: for each circle, try small perturbations to increase the minimum distance.
    """
    positions = np.array(positions, dtype=np.float64)
    rng = np.random.default_rng()
    
    for polish_iter in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            # Compute current min distance for circle i
            min_dist_i = np.inf
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist_i = min(min_dist_i, dist)
            for k in range(2):
                min_dist_i = min(min_dist_i, 0.5 - abs(positions[i][k]))
            
            # Try small perturbations
            for _ in range(20):
                if deadline is not None and time.time() > deadline - 2.0:
                    break
                new_pos = positions[i].copy()
                new_pos += rng.normal(0, 0.005, 2)
                # Check boundary
                valid = True
                for k in range(2):
                    if new_pos[k] < -0.5 or new_pos[k] > 0.5:
                        valid = False
                        break
                if not valid:
                    continue
                # Check min distance from new position
                new_min_dist = np.inf
                for j in range(N):
                    if i != j:
                        dist = np.linalg.norm(new_pos - positions[j])
                        new_min_dist = min(new_min_dist, dist)
                for k in range(2):
                    new_min_dist = min(new_min_dist, 0.5 - abs(new_pos[k]))
                if new_min_dist > min_dist_i * 1.0005:
                    positions[i] = new_pos
                    improved = True
                    break
        if not improved and polish_iter > 100:
            break
    
    return positions


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best = (0.0, None)
    rng = np.random.default_rng(42)
    
    # Phase 1: Hexagonal seed + L-BFGS + Annealing
    pos = _hexagonal_seed(rng)
    r = _radius(pos)
    best = (r, pos.copy())
    
    pos_opt = _lbfgs_optimize(pos, deadline)
    r_opt = _radius(pos_opt)
    if r_opt > best[0]:
        best = (r_opt, pos_opt.copy())
    
    pos_a = _anneal(pos_opt, iters=500, t0=0.008, tmin=0.0003, deadline=deadline)
    if pos_a[0] > best[0]:
        best = pos_a
    
    # Phase 2: Hard-sphere radius growth from current best
    pos_grow, r_grow = _grow_radius(best[1], best[0] * 0.95, deadline)
    if r_grow > best[0]:
        best = (r_grow, pos_grow.copy())
    
    # Phase 3: Local polish
    pos_polish = _local_polish(best[1], best[0], max_iters=300, deadline=deadline)
    r_polish = _radius(pos_polish)
    if r_polish > best[0]:
        best = (r_polish, pos_polish.copy())
    
    # Phase 4: Multiple restarts with different strategies
    for attempt in range(25):
        if time.time() > deadline - 10.0:
            break
        
        # Mix of fresh hex seeds, perturbed best, and random
        if attempt % 3 == 0:
            pos = _hexagonal_seed(rng)
        elif attempt % 3 == 1:
            pos = best[1] + 0.01 * rng.standard_normal((N, 2))
            pos = np.clip(pos, -0.5, 0.5)
        else:
            pos = rng.uniform(-0.45, 0.45, size=(N, 2))
        
        # L-BFGS
        pos_opt = _lbfgs_optimize(pos, deadline)
        r_opt = _radius(pos_opt)
        
        # Anneal
        pos_a = _anneal(pos_opt, iters=200, t0=0.006, tmin=0.0002, deadline=deadline)
        
        # Hard-sphere growth
        pos_grow, r_grow = _grow_radius(pos_a[1], pos_a[0] * 0.95, deadline)
        
        # Local polish
        pos_polish = _local_polish(pos_grow, r_grow, max_iters=100, deadline=deadline)
        r_polish = _radius(pos_polish)
        
        cand = max((r_opt, pos_opt), pos_a, (r_grow, pos_grow), (r_polish, pos_polish),
                   key=lambda t: t[0])
        if cand[0] > best[0]:
            best = cand
        
        # Early stop if near best known
        if best[0] >= 0.07137:
            break
    
    # Final polish on best
    pos_final = _local_polish(best[1], best[0], max_iters=200, deadline=deadline)
    r_final = _radius(pos_final)
    if r_final > best[0]:
        best = (r_final, pos_final.copy())
    
    return np.asarray(best[1], dtype=np.float64)
