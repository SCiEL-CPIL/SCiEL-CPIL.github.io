import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return -np.inf
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via negative log-sum-exp."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints) + boundary penalty.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints (smooth approximation to the true min)
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _initial_grid():
    """4x4x4 grid initialization, pick 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """FCC-like layered hexagonal initialization."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            idx = layer * per_layer + i
            if idx < N:
                centers[idx] = [x, y, z]
    
    # Fill any remaining
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def _lloyd_step(centers, n_iters=20, alpha=0.3):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell,
    clipped to the cube. This equalizes the influence regions.
    """
    centers = centers.copy()
    for _ in range(n_iters):
        # Compute Voronoi centroids using a simple nearest-neighbor partition
        # For efficiency, use scipy's Voronoi
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(centers)
            
            # For each site, compute the centroid of its Voronoi cell vertices
            # that lie within the cube [-0.5, 0.5]^3
            new_centers = centers.copy()
            for i in range(N):
                # Get vertices of the Voronoi cell for site i
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    continue
                verts = vor.vertices[region]
                # Filter to cube
                mask = (np.all(verts >= -0.5, axis=1) & np.all(verts <= 0.5, axis=1))
                if np.sum(mask) > 0:
                    centroid = np.mean(verts[mask], axis=0)
                    new_centers[i] = (1 - alpha) * centers[i] + alpha * centroid
            centers = new_centers
        except Exception:
            # Fallback: simple relaxation - move away from nearest neighbor
            for i in range(N):
                dists = np.linalg.norm(centers[i] - centers, axis=1)
                dists[i] = np.inf
                j = np.argmin(dists)
                if dists[j] > 1e-10:
                    direction = centers[i] - centers[j]
                    direction /= np.linalg.norm(direction)
                    new_pos = centers[i] + alpha * 0.01 * direction
                    if np.all(np.abs(new_pos) <= 0.5):
                        centers[i] = new_pos
            # Push away from walls
            for i in range(N):
                for k in range(3):
                    if centers[i, k] > 0.4:
                        centers[i, k] -= alpha * 0.01
                    elif centers[i, k] < -0.4:
                        centers[i, k] += alpha * 0.01
    
    return centers

def _polish_coordinate_descent(centers, deadline, max_rounds=200):
    """
    Final polish via coordinate descent on the exact objective.
    """
    best_score = _score(centers)
    step = 0.001
    
    for _ in range(max_rounds):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 3:
                break
            for coord in range(3):
                for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2]:
                    new_c = centers.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > best_score + 1e-12:
                            centers = new_c
                            best_score = s
                            improved = True
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    
    return centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
    ]
    
    # Add a perturbed grid
    rng = np.random.RandomState(123)
    perturbed = _initial_grid() + rng.normal(0, 0.02, (N, 3))
    perturbed = np.clip(perturbed, -0.5, 0.5)
    inits.append(("perturbed", perturbed))
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 25:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
        
        # Phase 1: Lloyd's algorithm for structural improvement
        if time.time() < deadline - 20:
            lloyd_centers = _lloyd_step(cur, n_iters=15, alpha=0.25)
            lloyd_score = _score(lloyd_centers)
            if lloyd_score > cur_score:
                cur = lloyd_centers
                cur_score = lloyd_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_centers = cur.copy()
        
        # Phase 2: Smooth surrogate optimization with annealed beta
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Phase 3: Coordinate descent polish
        if time.time() < deadline - 5:
            cur = best_centers.copy()
            polished, polished_score = _polish_coordinate_descent(cur, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
    
    if best_centers is None:
        return _initial_grid()
    return best_centers
