"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, QhullError
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _compute_radius_fast(c):
    """Fast radius computation using squared distances where possible."""
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    # Wall
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _initial_config(n):
    """
    Generate a good initial configuration using a body-centered cubic (BCC) inspired
    arrangement or a deformed 4x4x4 grid with 50 points selected for density.
    """
    # Start with a 4x4x4 grid (64 points) and remove 14 points.
    # To maximize density, we should keep points that are well-separated.
    # A better initial guess: take a 5x5x5 grid (125 points) and select 50?
    # Or use a known good packing seed?
    # Let's use a 4x4x4 grid but perturb it significantly to escape local minima,
    # or use a 3x4x4 grid (48 points) + 2 random?
    # Actually, 50 points in a cube.
    # A 3x4x4 grid has 48 points. Spacing 1/3 in x, 1/3 in y, 1/3 in z?
    # No, 3 points in x: spacing 1/2? No.
    # Let's stick to the 4x4x4 grid but select the 50 "best" points or just randomize.
    # For Lloyd's algorithm, a random start is often better to avoid symmetry traps.
    # But a structured start helps convergence.
    
    # Let's try a 5x5x5 grid selection.
    # Or simply: 4x4x4 grid, remove 14 points randomly? No, deterministic is better for reproducibility if seed is fixed.
    # Let's use a 4x4x4 grid, keep all 64, then merge/remove? No, N is fixed.
    
    # Strategy: Use a 5x5x5 grid (125 points). Select 50 points that maximize minimum distance?
    # That's expensive.
    
    # Simpler: 4x4x4 grid (64 points). Remove 14 points from the "corners" or "edges" to allow expansion?
    # Actually, for packing, corners are good for radius.
    # Let's just take the first 50 of a 4x4x4 grid, but add noise.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64. We need 50.
    # Randomly remove 14? Or remove specific ones?
    # Let's remove points from the center of faces?
    # To be safe, let's just use a random subset of the 64 grid points, or better:
    # Use a 3x4x4 grid (48 points) and add 2 points?
    # Let's use a 4x4x4 grid and keep 50.
    # We'll keep the ones with the highest "corner-ness" or just random.
    # Since we will run Lloyd's, the initial structure matters less than having no overlaps.
    # The 4x4x4 grid has spacing 1/3. Radius approx 1/6 = 0.166.
    # Wait, 4x4x4 in unit cube.
    # Coords: -0.5, -1/6, 1/6, 0.5.
    # Spacing is 1/3.
    # Nearest neighbor distance is 1/3.
    # Radius is 1/6 = 0.1666.
    # This is already better than 0.1359.
    # Why did the previous program score 0.92?
    # Ah, the previous program used `grid[:50]`.
    # The grid is ordered by z, then y, then x.
    # Taking the first 50 might bias the distribution.
    # Let's randomize the selection of 50 from 64.
    
    np.random.seed(42)
    indices = np.random.choice(64, 50, replace=False)
    c = grid[indices].copy()
    
    # Add small noise to break symmetry and avoid degenerate Voronoi cells
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    c = np.clip(c, -0.49, 0.49)
    
    return c


def _lloyd_step(c, max_iter=50, tol=1e-6):
    """
    Perform Lloyd's algorithm steps.
    Move each point to the centroid of its Voronoi cell clipped to the cube.
    """
    c = c.copy()
    n = len(c)
    
    # Add boundary points to ensure Voronoi cells are bounded by the cube
    # We add points far outside the cube to clip the cells.
    # Or, we can compute the Voronoi and then manually clip the centroid?
    # SciPy's Voronoi gives unbounded regions.
    # To get the centroid of the clipped cell, we need to integrate or approximate.
    # Approximation: Add dummy points at the corners and faces of a slightly larger cube
    # so that the Voronoi cells of the real points are bounded within the real cube.
    
    # A robust way: Add points at (-0.5 - eps, ...) etc?
    # Actually, a common trick is to add points at the 8 corners of the cube,
    # and points at the centers of the 6 faces, extended slightly?
    # No, the standard trick for "Voronoi in a box" is to add dummy points outside.
    
    # Let's add dummy points:
    # 8 corners of the cube: (-0.5, -0.5, -0.5) etc.
    # But we want the cells to be clipped BY the cube.
    # If we add dummy points just outside the cube, the Voronoi boundaries will pass through the cube.
    # The centroid of the Voronoi cell of a real point, computed including the dummy points,
    # will be pulled towards the dummy points if the cell is open.
    # This is not exactly the centroid of the clipped cell.
    
    # Alternative: Use the property that for a point in a box, the "centroid" can be approximated
    # by the mean of the vertices of the Voronoi cell that lie inside the box.
    # This is complex to implement from scratch.
    
    # Simpler Approximation:
    # 1. Compute Voronoi of the points + dummy points (corners of a larger box).
    # 2. For each real point, get its region.
    # 3. Intersect the region with the cube [-0.5, 0.5]^3.
    # 4. Compute the centroid of the intersection.
    
    # SciPy does not have direct "centroid of clipped Voronoi cell".
    # However, we can use a simpler heuristic:
    # Move each point towards the center of the cube if it's near a wall,
    # and away from neighbors if it's crowded.
    
    # Let's try a different "Lloyd-like" step:
    # For each point i, calculate the "center of mass" of the empty space around it.
    # This is hard.
    
    # Let's stick to a simpler "equal clearance" heuristic:
    # For each point i, find its nearest neighbor j and nearest wall.
    # If dist(i,j) < 2*dist(i,wall), move i slightly away from j.
    # If dist(i,wall) < dist(i,j)/2, move i slightly away from wall.
    # This is essentially the repulsion relaxation but with a specific target.
    
    # Actually, let's implement the Dummy Point Voronoi approach and just take the
    # Voronoi vertex centroid, ignoring the clipping error for now, or use a
    # "clipped" approximation by averaging the vertices that are inside the cube.
    
    # Let's try the dummy point method first.
    # Add 8 points at the corners of a box slightly larger than the cube.
    # Box: [-0.5 - d, 0.5 + d]^3
    # If d is small, the Voronoi cells of interior points will be clipped by the dummy points' cells.
    # The centroid of the full Voronoi cell (including the part outside the cube) will be slightly biased.
    # But it's a good gradient.
    
    d = 0.1
    corners = np.array([
        [-0.5-d, -0.5-d, -0.5-d],
        [-0.5-d, -0.5-d, 0.5+d],
        [-0.5-d, 0.5+d, -0.5-d],
        [-0.5-d, 0.5+d, 0.5+d],
        [0.5+d, -0.5-d, -0.5-d],
        [0.5+d, -0.5-d, 0.5+d],
        [0.5+d, 0.5+d, -0.5-d],
        [0.5+d, 0.5+d, 0.5+d],
    ])
    
    # Also add face centers? No, corners might be enough to bound the cells.
    # Actually, we need to bound the cells in all directions.
    # Adding points at the centers of the faces of the larger box might help.
    # Let's just use the corners for now.
    
    all_points = np.vstack([c, corners])
    
    for _ in range(max_iter):
        try:
            vor = Voronoi(all_points)
        except QhullError:
            # Degenerate case, add small noise
            c += np.random.uniform(-1e-5, 1e-5, size=c.shape)
            continue
            
        new_c = c.copy()
        
        for i in range(n):
            region_id = vor.point_region[i]
            region = vor.regions[region_id]
            
            if -1 in region:
                # Unbounded region. This shouldn't happen if we have enough dummy points.
                # If it does, keep the point or move it towards center.
                continue
                
            vertices = vor.vertices[region]
            
            # Filter vertices that are inside the cube [-0.5, 0.5]^3
            # This is an approximation of the clipped cell centroid.
            inside_mask = np.all((vertices >= -0.5) & (vertices <= 0.5), axis=1)
            
            if np.sum(inside_mask) > 0:
                centroid = np.mean(vertices[inside_mask], axis=0)
            else:
                # Fallback: mean of all vertices (unclipped)
                centroid = np.mean(vertices, axis=0)
                
            new_c[i] = centroid
            
        # Clip to cube
        new_c = np.clip(new_c, -0.5, 0.5)
        
        # Check convergence
        delta = np.linalg.norm(new_c - c)
        c = new_c
        
        if delta < tol:
            break
            
    return c


def _softmin_radius(c, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    r_soft = -1/beta * log( sum_{i<j} exp(-beta * dist_ij/2) + sum_{i,k} exp(-beta * wall_dist_ik) )
    Wait, we want to maximize the MIN distance.
    min(x) = -1/beta * log(sum(exp(-beta * x)))
    Here x are the "clearances": d_ij/2 and d_ik.
    So we want to maximize this soft-min.
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise clearances: dists / 2
    pair_clears = dists / 2.0
    
    # Wall clearances
    wall_clears = 0.5 - np.abs(c)
    
    # Combine all clearances into one array
    # Shape: (N*(N-1)/2 + N*3)
    # To avoid huge memory, we can compute the log-sum-exp in parts.
    
    # Log-sum-exp trick:
    # log(sum(exp(-beta * x))) = -beta * ( -1/beta * log(sum(exp(-beta * x))) )
    # We want to maximize: -1/beta * log(sum(exp(-beta * x)))
    # Let f = sum(exp(-beta * x))
    # We want to maximize -log(f)/beta
    
    # Compute log-sum-exp for pairs and walls separately, then combine.
    # log(sum(exp(-beta * x_pairs)) + sum(exp(-beta * x_walls)))
    # = log(exp(log_sum_pairs) + exp(log_sum_walls))
    # = max(log_sum_pairs, log_sum_walls) + log(1 + exp(-abs(diff)))
    
    # Pairwise
    # We only need i < j to avoid double counting?
    # The formula for min over all i<j is symmetric.
    # sum_{i<j} exp(-beta * d_ij/2)
    # Let's use the full matrix but divide by 2? No, just sum unique pairs.
    
    # For efficiency, let's just compute the vector of all clearances.
    # N=50, pairs = 1225, walls = 150. Total 1375 values. Small.
    
    pair_idx = np.triu_indices(N, k=1)
    pair_clears = pair_clears[pair_idx]
    
    wall_clears_flat = wall_clears.flatten()
    
    all_clears = np.concatenate([pair_clears, wall_clears_flat])
    
    # Soft-min
    # r = -1/beta * logsumexp(-beta * all_clears)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -beta * all_clears
    m = np.max(x)
    log_sum = m + np.log(np.sum(np.exp(x - m)))
    
    r_soft = -log_sum / beta
    
    return r_soft


def _softmin_grad(c, beta=100.0):
    """
    Gradient of the soft-min radius w.r.t. c.
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    pair_clears = dists / 2.0
    wall_clears = 0.5 - np.abs(c)
    
    pair_idx = np.triu_indices(N, k=1)
    pair_clears_flat = pair_clears[pair_idx]
    wall_clears_flat = wall_clears.flatten()
    
    all_clears = np.concatenate([pair_clears_flat, wall_clears_flat])
    
    x = -beta * all_clears
    m = np.max(x)
    exp_x = np.exp(x - m)
    sum_exp = np.sum(exp_x)
    weights = exp_x / sum_exp  # Softmax weights
    
    # Gradient of r_soft = -1/beta * log(sum(exp(-beta * c_k)))
    # dr/dc_i = -1/beta * (1/sum) * sum_k (d/dc_i exp(-beta * c_k))
    # = -1/beta * (1/sum) * sum_k exp(-beta * c_k) * (-beta * dc_k/dc_i)
    # = sum_k weights_k * dc_k/dc_i
    
    # So grad = sum_k weights_k * grad(c_k)
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    # c_k = dist(i,j)/2
    # d(dist)/d(c_i) = (c_i - c_j) / dist
    # d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist)
    
    n_pairs = len(pair_idx[0])
    pair_weights = weights[:n_pairs]
    
    # Vectorize the pairwise gradient accumulation
    # This is tricky to vectorize fully without loops over pairs for accumulation.
    # N=50, 1225 pairs. A loop is fast enough.
    for k in range(n_pairs):
        i, j = pair_idx[0][k], pair_idx[1][k]
        w = pair_weights[k]
        if w < 1e-12:
            continue
        d_ij = c[i] - c[j]
        dist = np.linalg.norm(d_ij)
        if dist < 1e-10:
            continue
        grad_pair = d_ij / (2.0 * dist)
        grad_c[i] += w * grad_pair
        grad_c[j] -= w * grad_pair  # d(c_j)/d(c_j) = -d_ij/dist / 2
        
    # Wall part
    # c_k = 0.5 - |c_ik|
    # d(c_k)/d(c_ik) = -sign(c_ik)
    
    wall_weights = weights[n_pairs:]
    # wall_clears is (N, 3)
    # wall_weights is (N*3,)
    # grad_c[i, k] += wall_weights[i*3+k] * (-sign(c[i,k]))
    
    grad_wall = -np.sign(c) * wall_weights.reshape(N, 3)
    grad_c += grad_wall
    
    return grad_c


def _polish(c, time_limit=20.0):
    """
    Use L-BFGS-B to maximize the soft-min radius.
    """
    t_start = time.time()
    
    # Start with a moderate beta, increase it for precision
    # If beta is too high, it's non-differentiable.
    # If beta is too low, it's too smooth.
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() - t_start > time_limit - 2:
            break
            
        def objective(x):
            c_temp = x.reshape(N, 3)
            # Clip to valid range to avoid numerical issues in softmin
            c_temp = np.clip(c_temp, -0.5 + 1e-9, 0.5 - 1e-9)
            r_soft = _softmin_radius(c_temp, beta)
            return -r_soft  # Minimize negative radius
        
        def gradient(x):
            c_temp = x.reshape(N, 3)
            c_temp = np.clip(c_temp, -0.5 + 1e-9, 0.5 - 1e-9)
            g = _softmin_grad(c_temp, beta)
            return g.flatten()
        
        try:
            res = minimize(
                objective,
                c.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    # 1. Initial Configuration
    c = _initial_config(N)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # 2. Lloyd's Algorithm (Voronoi Centroid Moves)
    # Run for a while to equalize clearance
    if time.time() < deadline - 30.0:
        c = _lloyd_step(c, max_iter=50, tol=1e-6)
        c = np.clip(c, -0.5, 0.5)
        r_lloyd = _compute_radius(c)
        
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c.copy()
            
    # 3. Multi-scale Soft-Min Polish
    # Use the result from Lloyd's as the starting point for the gradient polish
    if time.time() < deadline - 10.0:
        time_left = deadline - time.time() - 5.0
        c_polished, r_polished = _polish(best_c, time_limit=time_left)
        
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
            
    # 4. Final check and return
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
