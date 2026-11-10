import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_score(x):
    """Compute the true radius (score) for a configuration."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(x)
    min_wall = np.min(np.min(wall_dists, axis=1))
    
    return min(min_pair / 2.0, min_wall)

def _generate_fcc_init():
    """Generate initial points from FCC lattice structure, scaled to fit 50 points."""
    # FCC lattice points in a unit cell (conventional cubic cell)
    # FCC has atoms at corners and face centers
    # For packing 50 spheres, we use a 4x4x4 grid of FCC unit cells
    # Each FCC unit cell has 4 atoms
    # 4x4x4 = 64 unit cells * 4 atoms = 256 atoms, we pick 50
    
    # FCC positions in one unit cell (fractional coordinates)
    fcc_positions = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ])
    
    # Generate a 4x4x4 grid of FCC unit cells
    # Each unit cell has side length s, so the cube side is 4*s
    # We want the cube to be [-0.5, 0.5], so side = 1, s = 0.25
    s = 0.25
    m = 4  # number of unit cells in each direction
    
    all_points = []
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                for fp in fcc_positions:
                    p = np.array([ix, iy, iz]) * s + fp * s
                    all_points.append(p)
    
    all_points = np.array(all_points)
    
    # The FCC lattice spans [0, 4*s] = [0, 1] in each dimension
    # Center it: shift by -0.5
    all_points = all_points - 0.5
    
    # Now we have 256 points in [-0.5, 0.5]^3
    # We need to select 50 points that are well-distributed
    # Strategy: select points with maximum minimum distance (greedy)
    
    # For speed, use a simpler approach: take every 5th point from a shuffled list
    # But better: use a space-filling selection
    
    # Simple approach: divide into 50 regions and pick one from each
    # Or: just take the first 50 from a deterministic ordering that spreads them out
    
    # Better: use a jittered selection
    # Let's use a simple approach: take points on a 3x3x3 grid of FCC cells
    # 3x3x3 = 27 cells, each with 4 points = 108, pick 50
    
    # Actually, let's just use a 4x4x4 grid and pick 50 well-spread points
    # Use a greedy farthest-point sampling
    
    rng = np.random.RandomState(42)
    order = rng.permutation(len(all_points))
    
    # Greedy: start with a random point, then repeatedly add the point
    # that maximizes the minimum distance to already-selected points
    selected = [order[0]]
    selected_set = set(selected)
    
    for _ in range(N - 1):
        if len(selected) >= N:
            break
        # Compute min distance from each unselected point to selected points
        sel = all_points[selected]
        # dists[i] = min distance from point i to any selected point
        dists = np.min(np.sqrt(np.sum((all_points[:, np.newaxis, :] - sel[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        # Only consider unselected points
        unselected_mask = np.ones(len(all_points), dtype=bool)
        for idx in selected:
            unselected_mask[idx] = False
        dists[~unselected_mask] = -np.inf
        # Pick the farthest
        next_idx = np.argmax(dists)
        selected.append(next_idx)
        selected_set.add(next_idx)
    
    points = all_points[selected]
    
    # Add small jitter to break symmetry
    jitter = rng.normal(0, 0.02, points.shape)
    points = points + jitter
    
    # Clamp to cube interior
    points = np.clip(points, -0.48, 0.48)
    
    return points

def _generate_jittered_grid_init(seed=0):
    """Generate initial points from a jittered grid."""
    rng = np.random.RandomState(seed)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    jitter = rng.normal(0, 0.05, grid.shape)
    points = grid + jitter
    points = np.clip(points, -0.45, 0.45)
    
    if len(points) > N:
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    else:
        while len(points) < N:
            p = rng.uniform(-0.45, 0.45, 3)
            points = np.vstack([points, p])
    
    return points

def _soft_min_objective_and_grad(x_flat, beta=100.0):
    """Smooth objective using log-sum-exp to approximate min distance."""
    x = x_flat.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq + 1e-12)
    
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(x)
    min_wall_dists = np.min(wall_dists, axis=1)
    
    all_dists = np.concatenate([pair_dists, min_wall_dists])
    
    vals = -beta * all_dists
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    obj = -log_sum_exp / beta
    
    # Gradient
    exp_vals = np.exp(vals - max_val)
    S = np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    W = np.zeros((N, N))
    W[mask] = exp_vals[:N*(N-1)] / S
    unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall gradient
    wall_exp = exp_vals[N*(N-1):]
    weights_wall = wall_exp / S
    grad_wall = np.zeros_like(x)
    min_dims = np.argmin(wall_dists, axis=1)
    
    for i in range(N):
        k = min_dims[i]
        grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
    
    grad = grad_pair + grad_wall
    
    return obj, grad

def _energy_objective_and_grad(x_flat, target_r, penalty=1000.0):
    """Energy-based objective: penalize pairs closer than 2*target_r and points too close to walls."""
    x = x_flat.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq + 1e-12)
    
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(x)
    min_wall_dists = np.min(wall_dists, axis=1)
    
    # Energy: sum of squared violations
    # Pair violation: max(0, 2*target_r - d_ij)^2
    pair_viol = np.maximum(0, 2 * target_r - pair_dists)
    pair_energy = np.sum(pair_viol ** 2)
    
    # Wall violation: max(0, target_r - w_i)^2
    wall_viol = np.maximum(0, target_r - min_wall_dists)
    wall_energy = np.sum(wall_viol ** 2)
    
    energy = penalty * (pair_energy + wall_energy)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pair gradient
    # d/dx_i sum_j max(0, 2r - d_ij)^2
    # = sum_j 2*max(0, 2r - d_ij) * (-1) * (x_i - x_j)/d_ij
    # = -2 * sum_j max(0, 2r - d_ij) * (x_i - x_j)/d_ij
    
    pair_viol_full = np.maximum(0, 2 * target_r - dists)
    np.fill_diagonal(pair_viol_full, 0)
    
    unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad_pair = -2 * penalty * np.sum(pair_viol_full[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall gradient
    # d/dx_ik max(0, r - (0.5 - |x_ik|))^2
    # For the min wall dimension:
    # d/dx_ik (r - 0.5 + |x_ik|)^2 * 2 * sign(x_ik) if violation
    wall_viol_full = np.maximum(0, target_r - wall_dists)
    min_dims = np.argmin(wall_dists, axis=1)
    
    for i in range(N):
        k = min_dims[i]
        if wall_viol_full[i, k] > 0:
            # d/dx_ik (r - 0.5 + |x_ik|)^2 = 2*(r - 0.5 + |x_ik|) * sign(x_ik)
            # But we need the gradient of the squared violation
            # violation = r - (0.5 - |x_ik|) = r - 0.5 + |x_ik|
            # d(violation^2)/dx_ik = 2*violation * d(violation)/dx_ik = 2*violation * sign(x_ik)
            grad_wall_val = 2 * penalty * wall_viol_full[i, k] * np.sign(x[i, k])
            grad[i, k] += grad_wall_val
    
    grad = grad_pair + grad
    
    return energy, grad

def _resolve_overlaps(x, target_r, max_iters=1000):
    """Iteratively resolve overlaps by pushing apart spheres that are too close."""
    x = x.copy()
    
    for it in range(max_iters):
        # Find all pairs that are too close
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find violations
        mask = dists < 2 * target_r
        if not np.any(mask):
            break
        
        # Compute pushes
        pushes = np.zeros_like(x)
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dists[i, j]
                    if d < 1e-10:
                        # Random push
                        push = np.random.randn(3) * 0.01
                    else:
                        push = (x[i] - x[j]) / d
                    overlap = (2 * target_r - d) / 2
                    pushes[i] += push * overlap
                    pushes[j] -= push * overlap
        
        # Also push away from walls
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(wall_dists, axis=1)
        for i in range(N):
            if min_wall[i] < target_r:
                min_dim = np.argmin(wall_dists[i])
                # Push away from wall
                if x[i, min_dim] > 0:
                    pushes[i, min_dim] -= (target_r - wall_dists[i, min_dim])
                else:
                    pushes[i, min_dim] += (target_r - wall_dists[i, min_dim])
        
        # Apply pushes with damping
        x += 0.5 * pushes
        
        # Clamp to cube
        x = np.clip(x, -0.5 + target_r, 0.5 - target_r)
    
    return x

def _binary_search_refinement(x_start, time_limit, deadline):
    """Binary search on target radius with energy minimization."""
    x = x_start.copy()
    current_r = _compute_score(x)
    
    # Upper bound: theoretical maximum
    # Volume argument: 50 * (4/3) pi r^3 <= 1 => r <= (3/(50*4*pi))^(1/3) ≈ 0.173
    # But packing fraction limits this. Use 0.16 as upper bound.
    r_low = current_r
    r_high = 0.16
    
    best_x = x
    best_r = current_r
    
    # Binary search
    for _ in range(30):  # Max 30 iterations
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find a configuration with radius >= r_mid
        # Use energy minimization
        x_trial = x.copy()
        
        try:
            res = minimize(
                lambda xf: _energy_objective_and_grad(xf, r_mid, penalty=1000.0),
                x_trial.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            x_trial = res.x.reshape(N, 3)
        except Exception:
            x_trial = x.copy()
        
        # Resolve any remaining overlaps
        x_trial = _resolve_overlaps(x_trial, r_mid, max_iters=200)
        
        actual_r = _compute_score(x_trial)
        
        if actual_r >= r_mid - 1e-9:
            # Feasible, try higher
            r_low = r_mid
            if actual_r > best_r:
                best_r = actual_r
                best_x = x_trial.copy()
        else:
            # Not feasible, try lower
            r_high = r_mid
        
        # Update x for next iteration
        if actual_r > current_r:
            x = x_trial.copy()
            current_r = actual_r
    
    return best_x, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_score = -np.inf
    
    # Phase 1: Try multiple initializations with soft-min optimization
    initializations = []
    
    # FCC-based initialization
    initializations.append(_generate_fcc_init())
    
    # Jittered grid initializations with different seeds
    for seed in range(5):
        initializations.append(_generate_jittered_grid_init(seed))
    
    # Random initializations
    rng = np.random.RandomState(123)
    for _ in range(3):
        initializations.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    for x_init in initializations:
        if time.time() > deadline - 30:
            break
        
        # Soft-min optimization
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                x_init.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            x_opt = res.x.reshape(N, 3)
        except Exception:
            continue
        
        score = _compute_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt
        
        # Phase 2: Binary search refinement on the best solution found so far
        if time.time() > deadline - 20:
            break
        
        try:
            x_refined, r_refined = _binary_search_refinement(x_opt, deadline - 10, deadline)
            if r_refined > best_score:
                best_score = r_refined
                best_x = x_refined
        except Exception:
            pass
    
    # Phase 3: Final polish on the best solution
    if best_x is not None and time.time() < deadline - 5:
        try:
            # Tighter optimization
            res = minimize(
                _soft_min_objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            x_polished = res.x.reshape(N, 3)
            score_polished = _compute_score(x_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished
        except Exception:
            pass
        
        # Final binary search refinement
        if time.time() < deadline - 3:
            try:
                x_final, r_final = _binary_search_refinement(best_x, deadline - 1, deadline)
                if r_final > best_score:
                    best_score = r_final
                    best_x = x_final
            except Exception:
                pass
    
    if best_x is None:
        # Fallback
        best_x = _generate_fcc_init()
    
    return best_x
