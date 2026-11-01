"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min(values, beta):
    """
    Smooth approximation of min(values) using log-sum-exp.
    As beta -> inf, soft_min -> min.
    """
    m = np.min(values)
    shifted = values - m
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return m - log_sum / beta

def _compute_pair_dists(centers):
    """Compute pairwise distances."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu[0], iu[1]]
    pair_dists = np.sqrt(pair_dists_sq)
    return pair_dists, iu

def _compute_wall_dists(centers):
    """Compute wall distances."""
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dists = np.min(wall_dists, axis=1)
    return min_wall_dists

def _softmin_obj_and_grad(centers_flat, beta):
    """
    Compute soft-min objective and gradient.
    Objective = -soft_min([pair_dists/2, wall_dists])
    """
    centers = centers_flat.reshape(-1, 3)
    pair_dists, iu = _compute_pair_dists(centers)
    min_wall_dists = _compute_wall_dists(centers)
    
    vals = np.concatenate([pair_dists / 2.0, min_wall_dists])
    
    # Compute soft_min
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    r_smooth = m - np.log(sum_exp) / beta
    obj = -r_smooth
    
    # Compute gradient
    grad = np.zeros_like(centers)
    
    w_pairs = weights[:N*(N-1)//2]
    w_walls = weights[N*(N-1)//2:]
    
    i_idx = iu[0]
    j_idx = iu[1]
    
    diff = centers[i_idx] - centers[j_idx]
    dists = pair_dists.copy()
    dists[dists < 1e-12] = 1e-12
    
    coeff = w_pairs / (2.0 * dists)
    
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * diff)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * diff)
    
    # Wall gradient
    abs_c = np.abs(centers)
    active_axis = np.argmax(abs_c, axis=1)
    active_sign = np.sign(centers)
    
    grad_wall = np.zeros_like(centers)
    active_sign_vals = active_sign[np.arange(N), active_axis]
    grad_wall[np.arange(N), active_axis] = -w_walls * active_sign_vals
    
    grad += grad_wall
    
    return obj, grad.flatten()

def _delaunay_adjacency(centers):
    """
    Compute Delaunay adjacency graph.
    Returns list of neighbor indices for each vertex.
    """
    try:
        tri = Delaunay(centers)
        neighbors = [[] for _ in range(N)]
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    if b not in neighbors[a]:
                        neighbors[a].append(b)
                    if a not in neighbors[b]:
                        neighbors[b].append(a)
        return neighbors
    except Exception:
        # Fallback: k-nearest neighbors
        tree = cKDTree(centers)
        neighbors = []
        for i in range(N):
            dists, idxs = tree.query(centers[i], k=8)
            neighbors.append(list(idxs[1:]))
        return neighbors

def _voronoi_centroid_step(centers, step_size=0.1):
    """
    Move each center towards the centroid of its Voronoi cell (approximated).
    This helps equalize cell volumes.
    """
    neighbors = _delaunay_adjacency(centers)
    new_centers = centers.copy()
    
    for i in range(N):
        if len(neighbors[i]) == 0:
            continue
        
        # Compute centroid of neighbors
        neigh_pts = centers[neighbors[i]]
        centroid = np.mean(neigh_pts, axis=0)
        
        # Also consider wall constraints: pull towards center if near wall
        for d in range(3):
            if centers[i, d] > 0.3:
                centroid[d] = min(centroid[d], 0.25)
            elif centers[i, d] < -0.3:
                centroid[d] = max(centroid[d], -0.25)
        
        # Move towards centroid
        new_centers[i] = centers[i] + step_size * (centroid - centers[i])
    
    # Clip to valid region
    new_centers = np.clip(new_centers, -0.49, 0.49)
    
    return new_centers

def _force_relaxation(centers, num_iters=100, k_repel=1.0, k_wall=0.5):
    """
    Force-based relaxation: repulsion between spheres, attraction to center from walls.
    """
    centers = centers.copy()
    
    for it in range(num_iters):
        forces = np.zeros_like(centers)
        
        # Repulsion between all pairs
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(3) * 1e-10
                    dist = np.linalg.norm(diff)
                
                # Inverse square repulsion
                f = k_repel / (dist * dist)
                unit = diff / dist
                forces[i] += f * unit
                forces[j] -= f * unit
        
        # Wall repulsion
        for i in range(N):
            for d in range(3):
                if centers[i, d] > 0:
                    wall_dist = 0.5 - centers[i, d]
                    if wall_dist < 0.1:
                        forces[i, d] -= k_wall / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 + centers[i, d]
                    if wall_dist < 0.1:
                        forces[i, d] += k_wall / (wall_dist + 1e-6)
        
        # Apply forces with adaptive step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            step = 0.01 / (1 + max_force)
            centers += step * forces
        
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _metropolis_mc(centers, num_iters=5000, temp_start=0.02, temp_end=0.001):
    """
    Metropolis Monte Carlo: random perturbations accepted if they improve score.
    """
    centers = centers.copy()
    current_score = _score(centers)
    
    rng = np.random.default_rng(42)
    
    for it in range(num_iters):
        # Annealing schedule
        progress = it / num_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, temp, 3)
        
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = _score(new_centers)
        
        # Accept if better, or with probability exp((new_score - current_score) / temp)
        if new_score >= current_score:
            centers = new_centers
            current_score = new_score
        else:
            prob = np.exp((new_score - current_score) / (temp + 1e-10))
            if rng.random() < prob:
                centers = new_centers
                current_score = new_score
    
    return centers

def _seed_fcc():
    """Generate FCC-like initial configuration."""
    points = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                p = n1 * np.array([0, 0.5, 0.5]) + \
                    n2 * np.array([0.5, 0, 0.5]) + \
                    n3 * np.array([0.5, 0.5, 0])
                if np.all(p >= -0.5) and np.all(p <= 0.5):
                    points.append(p)
    
    points = np.array(points)
    
    if len(points) >= N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]
    else:
        rng = np.random.default_rng(42)
        while len(points) < N:
            p = rng.uniform(-0.45, 0.45, 3)
            points = np.vstack([points, p])
        return points[:N]

def _seed_grid():
    """Generate grid initial configuration."""
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(grid))[:N]
    return grid[idx]

def _softmin_polish(centers, beta_start=20.0, beta_end=500.0, num_stages=4):
    """
    Multi-stage soft-min polish using L-BFGS.
    """
    centers = centers.copy()
    
    for stage in range(num_stages):
        beta = beta_start * (beta_end / beta_start) ** (stage / max(num_stages - 1, 1))
        
        try:
            res = minimize(
                lambda x: _softmin_obj_and_grad(x, beta)[0],
                centers.flatten(),
                jac=lambda x: _softmin_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            centers = res.x.reshape(-1, 3)
            centers = np.clip(centers, -0.5, 0.5)
        except Exception:
            pass
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. FCC-like
    fcc = _seed_fcc()
    inits.append(fcc + rng.normal(0, 0.01, fcc.shape))
    
    # 2. Grid
    grid = _seed_grid()
    inits.append(grid + rng.normal(0, 0.005, grid.shape))
    
    # 3. Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # 4. Dense grid
    ticks_fine = np.linspace(-0.45, 0.45, 5)
    grid_fine = np.array([(x, y, z) for z in ticks_fine for y in ticks_fine for x in ticks_fine])
    d_center = np.linalg.norm(grid_fine, axis=1)
    idx_sorted = np.argsort(d_center)
    idx = idx_sorted[:N]
    centers_fine = grid_fine[idx].copy()
    inits.append(centers_fine + rng.normal(0, 0.01, centers_fine.shape))
    
    # 5. Another FCC with different perturbation
    inits.append(fcc + rng.normal(0, 0.02, fcc.shape))
    
    # Optimize each initial configuration
    for init_idx, centers in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = np.clip(centers, -0.49, 0.49)
        current_centers = centers.copy()
        
        # Stage 1: Force relaxation
        if time.time() < deadline - 30:
            current_centers = _force_relaxation(current_centers, num_iters=50)
        
        # Stage 2: Voronoi centroid step
        if time.time() < deadline - 25:
            for _ in range(10):
                current_centers = _voronoi_centroid_step(current_centers, step_size=0.05)
        
        # Stage 3: Soft-min polish
        if time.time() < deadline - 20:
            current_centers = _softmin_polish(current_centers)
        
        # Stage 4: Metropolis MC
        if time.time() < deadline - 15:
            current_centers = _metropolis_mc(current_centers, num_iters=2000)
        
        # Stage 5: Final soft-min polish
        if time.time() < deadline - 10:
            current_centers = _softmin_polish(current_centers, beta_start=50.0, beta_end=1000.0, num_stages=3)
        
        # Stage 6: Final MC polish
        if time.time() < deadline - 5:
            current_centers = _metropolis_mc(current_centers, num_iters=1000, temp_start=0.01, temp_end=0.0005)
        
        score = _score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
    
    # If time remains, do additional refinement on best candidate
    if best_centers is not None:
        remaining = deadline - time.time()
        
        if remaining > 15:
            # More force relaxation
            best_centers = _force_relaxation(best_centers, num_iters=30)
            score = _score(best_centers)
            best_score = score
        
        if deadline - time.time() > 10:
            # More Voronoi steps
            for _ in range(5):
                best_centers = _voronoi_centroid_step(best_centers, step_size=0.03)
            score = _score(best_centers)
            best_score = score
        
        if deadline - time.time() > 8:
            # Final soft-min
            best_centers = _softmin_polish(best_centers, beta_start=100.0, beta_end=2000.0, num_stages=3)
            score = _score(best_centers)
            best_score = score
        
        if deadline - time.time() > 5:
            # Final MC
            best_centers = _metropolis_mc(best_centers, num_iters=500, temp_start=0.005, temp_end=0.0001)
            score = _score(best_centers)
            best_score = score
    
    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
