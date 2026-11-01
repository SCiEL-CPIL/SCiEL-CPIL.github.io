"""Epigraph-based sphere packing solver with simulated annealing and adaptive local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _generate_fcc_initial():
    """Generate initial configuration from FCC lattice."""
    points = []
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])

    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()

    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _generate_grid_initial(seed=123):
    """Generate perturbed grid initial positions."""
    n_per_side = 4
    grid = np.array([[i, j, k] for i in range(n_per_side)
                     for j in range(n_per_side)
                     for k in range(n_per_side)], dtype=float)
    grid = (grid / (n_per_side - 1) - 0.5) * 0.85

    rng = np.random.default_rng(seed)
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)

    return grid[:N].copy()


def _generate_bcc_initial():
    """Generate initial positions from BCC lattice."""
    points = []
    max_idx = 4
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])

    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()

    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _lloyd_optimize(c, deadline, max_damps=8):
    """Voronoi-Lloyd equalization."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    damps = (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03)
    for di in range(min(max_damps, len(damps))):
        if time.time() > deadline:
            break
        damp = damps[di]
        steps = max(15, int(120 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                tri = Delaunay(c)
            except Exception:
                d = _pair_dist_matrix(c)
                inv = 1.0 / np.maximum(d, 1e-6) ** 2
                diff = c[:, None, :] - c[None, :, :]
                F = np.sum(diff * inv[:, :, None], axis=1)
                c = np.clip(c + 0.005 * F, -0.49, 0.49)
                continue

            simplices = tri.simplices
            cell_centroids = np.zeros((N, 3))
            cell_counts = np.zeros(N)

            for si in simplices:
                v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
                A = np.array([v1 - v0, v2 - v0, v3 - v0])
                b = np.array([
                    np.dot(v1, v1) - np.dot(v0, v0),
                    np.dot(v2, v2) - np.dot(v0, v0),
                    np.dot(v3, v3) - np.dot(v0, v0)
                ])
                try:
                    cc = np.linalg.solve(2.0 * A, b)
                except np.linalg.LinAlgError:
                    cc = np.mean([v0, v1, v2, v3])

                for vi in si:
                    cell_centroids[vi] += cc
                    cell_counts[vi] += 1

            valid = cell_counts > 0
            if np.any(valid):
                targets = cell_centroids[valid] / cell_counts[valid, None]
                c[valid] = c[valid] + damp * (targets - c[valid])

            c = np.clip(c, -0.49, 0.49)

        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _epigraph_objective(x, t=0.0):
    """Epigraph formulation: minimize -r + penalty for violated constraints.
    
    x = [r, c_1x, c_1y, c_1z, c_2x, ...]
    Constraints: |c_i - c_j| >= 2r for all i<j, and 0.5 - |c_ik| >= r for all i,k.
    We minimize -r + lambda * sum(max(0, violation)^2)
    """
    r = x[0]
    c = x[1:].reshape(N, 3)
    
    # Pair distance violations
    ii, jj = np.triu_indices(N, 1)
    diff = c[ii] - c[jj]
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    pair_viol = np.maximum(0.0, 2.0 * r - dists)
    
    # Wall violations
    wall = 0.5 - np.abs(c)
    wall_viol = np.maximum(0.0, r - wall)
    
    # Objective: -r + penalty
    pen = np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)
    value = -r + 10.0 * pen
    
    # Gradient
    grad = np.zeros_like(x)
    grad[0] = -1.0
    
    # Pair gradient contributions
    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        v = pair_viol[k]
        if v > 0:
            d = dists[k]
            if d > 1e-12:
                factor = 20.0 * v / d
                grad[1 + 3*i] += factor * diff[k]
                grad[1 + 3*j] -= factor * diff[k]
            # Gradient of -r term through constraint
            grad[0] += 20.0 * 2.0 * v
    
    # Wall gradient contributions
    for i in range(N):
        for k in range(3):
            v = wall_viol[i, k]
            if v > 0:
                sign = np.sign(c[i, k])
                if sign == 0:
                    sign = 1.0
                grad[1 + 3*i + k] -= 20.0 * v * sign
                grad[0] += 20.0 * v
    
    return value, grad


def _epigraph_polish(c, r_init, deadline, maxiter=500):
    """Polish using epigraph formulation with L-BFGS-B."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    
    x0 = np.concatenate([[r_init], c.flatten()])
    
    # Bounds: r in [0, 0.5], coordinates in [-0.49, 0.49]
    bounds = [(0.0, 0.5)] + [(-0.49, 0.49)] * (N * 3)
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30},
        )
        r_opt = res.x[0]
        c_opt = np.clip(res.x[1:].reshape(N, 3), -0.49, 0.49)
        r_actual = _compute_radius(c_opt)
        if r_actual > best_r:
            best_r = r_actual
            best = c_opt.copy()
    except Exception:
        pass
    
    return best, best_r


def _simulated_annealing(c, deadline, steps=3000, T0=0.02, Tmin=0.001):
    """Simulated annealing for global exploration."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(12345)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        frac = it / max(steps - 1, 1)
        T = T0 * (Tmin / T0) ** frac
        
        # Adaptive step size based on current radius
        step_scale = 0.015 * (best_r / 0.135 + 0.5)
        step_scale = np.clip(step_scale, 0.005, 0.03)
        
        # Move 1-3 random spheres
        n_move = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_move, replace=False)
        
        c_new = c.copy()
        for idx in idxs:
            delta = rng.normal(0, step_scale, size=3)
            c_new[idx] += delta
        c_new = np.clip(c_new, -0.49, 0.49)
        
        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)
        
        if r_new >= r_cur:
            c = c_new
        else:
            delta_r = r_new - r_cur
            prob = np.exp(delta_r / max(T, 1e-10))
            if rng.random() < prob:
                c = c_new
        
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r


def _adaptive_local_search(c, deadline, iters=2000):
    """Adaptive local search focusing on most violated constraints."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(999)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        wall = 0.5 - np.abs(c)
        
        current_r = min(np.min(d) / 2.0, np.min(wall))
        
        # Find the most violated constraints
        pair_threshold = current_r * 1.15
        wall_threshold = current_r * 1.15
        
        # Collect candidate moves
        moves = []
        
        # Near-binding pairs
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] / 2.0 <= pair_threshold:
                    direction = c[i] - c[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        direction /= norm
                        # Priority based on how close to binding
                        priority = pair_threshold - d[i, j] / 2.0
                        moves.append((priority, 'pair', i, j, direction))
        
        # Near-binding walls
        for i in range(N):
            for k in range(3):
                if wall[i, k] <= wall_threshold:
                    axis = k
                    if c[i, axis] > 0:
                        direction = np.zeros(3)
                        direction[axis] = -1.0
                    else:
                        direction = np.zeros(3)
                        direction[axis] = 1.0
                    priority = wall_threshold - wall[i, k]
                    moves.append((priority, 'wall', i, -1, direction))
        
        if not moves:
            i = rng.integers(0, N)
            delta = rng.normal(0, 0.002, size=3)
            c[i] += delta
        else:
            # Weighted random selection favoring more violated constraints
            priorities = np.array([m[0] for m in moves])
            probs = priorities / np.sum(priorities)
            chosen_idx = rng.choice(len(moves), p=probs)
            _, mtype, i, j, direction = moves[chosen_idx]
            
            base_step = 0.0025 * (best_r / 0.135 + 0.3)
            base_step = np.clip(base_step, 0.001, 0.005)
            step = base_step * rng.uniform(0.5, 1.8)
            
            if mtype == 'pair':
                c[i] += step * direction
                c[j] -= step * direction
            else:
                c[i] += step * direction
        
        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c, best_r


def _repulsion_anneal(c, deadline, steps=800):
    """Vectorized repulsion annealing."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)

    return c


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # Phase 1: Multi-start initialization (allocate ~10% of time)
    phase1_deadline = start_time + 13.5
    
    best_pts = _generate_fcc_initial()
    best_r = _compute_radius(best_pts)
    
    grid_pts = _generate_grid_initial(123)
    grid_r = _compute_radius(grid_pts)
    if grid_r > best_r:
        best_r = grid_r
        best_pts = grid_pts
    
    bcc_pts = _generate_bcc_initial()
    bcc_r = _compute_radius(bcc_pts)
    if bcc_r > best_r:
        best_r = bcc_r
        best_pts = bcc_pts
    
    # Try a few more random grid seeds
    for seed in [42, 77, 2024]:
        if time.time() > phase1_deadline:
            break
        pts = _generate_grid_initial(seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 2: Lloyd + repulsion to get a good baseline (~20% of time)
    phase2_deadline = start_time + 27.0
    
    if time.time() < phase2_deadline:
        pts, r = _lloyd_optimize(best_pts, phase2_deadline, max_damps=5)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if time.time() < phase2_deadline:
        pts = _repulsion_anneal(best_pts, phase2_deadline, steps=500)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 3: Epigraph polish (~15% of time)
    phase3_deadline = start_time + 47.25
    
    if time.time() < phase3_deadline:
        pts, r = _epigraph_polish(best_pts, best_r, phase3_deadline, maxiter=300)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 4: Simulated annealing for global exploration (~25% of time)
    phase4_deadline = start_time + 81.0
    
    if time.time() < phase4_deadline:
        pts, r = _simulated_annealing(best_pts, phase4_deadline, steps=2500, T0=0.02, Tmin=0.001)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 5: Adaptive local search (~15% of time)
    phase5_deadline = start_time + 101.25
    
    if time.time() < phase5_deadline:
        pts, r = _adaptive_local_search(best_pts, phase5_deadline, iters=1500)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 6: Final polish and refinement (~15% of time)
    phase6_deadline = deadline
    
    if time.time() < phase6_deadline:
        pts, r = _epigraph_polish(best_pts, best_r, phase6_deadline, maxiter=200)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if time.time() < phase6_deadline:
        pts, r = _adaptive_local_search(best_pts, phase6_deadline, iters=800)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if time.time() < phase6_deadline:
        pts, r = _epigraph_polish(best_pts, best_r, phase6_deadline, maxiter=100)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Final validation
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts
# EVOLVE-BLOCK-END
