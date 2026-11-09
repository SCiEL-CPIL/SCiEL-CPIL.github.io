import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

def compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def soft_min_score(centres_flat, beta=100.0):
    """
    Smooth approximation of the negative radius.
    Objective to minimize: -soft_min(r_i)
    where r_i are the constraints (pairwise/2, boundary).
    """
    centres = centres_flat.reshape(N, 3)
    
    # Boundary terms: 0.5 - |c_ik|
    boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
    
    # Pairwise distance terms: |c_i - c_j| / 2
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    
    all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * x)))
    # To be stable, shift by max
    max_term = np.max(all_terms)
    exp_terms = np.exp(-beta * (all_terms - max_term))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return 1e10
    soft_min_val = max_term - np.log(sum_exp) / beta
    
    return -soft_min_val

def neg_score(centres_flat):
    """Negative exact score for direct methods."""
    centres = centres_flat.reshape(N, 3)
    if not np.all(np.isfinite(centres)):
        return 1e10
    return -compute_score(centres)

def fibonacci_lattice(n_points, seed=0):
    """Generate n_points using Fibonacci lattice in the unit cube."""
    rng = np.random.default_rng(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n_points):
        # Fibonacci sphere parameter
        t = i / n_points
        z = 1 - 2 * t  # z in [-1, 1]
        radius = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        # Scale to fit in [-0.5, 0.5]^3
        point = np.array([x, y, z]) * 0.45
        # Add small jitter
        point += rng.normal(0, 0.01, size=(3,))
        point = np.clip(point, -0.5, 0.5)
        points.append(point)
    return np.array(points)

def halton_sequence(n, base):
    """Generate n points in the Halton sequence with given base."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_val = i
        while i_val > 0:
            f /= base
            x += f * (i_val % base)
            i_val //= base
        result.append(x)
    return np.array(result)

def generate_jittered_grid(n_points, seed=0):
    """Generate a jittered grid of n_points in [-0.5, 0.5]^3."""
    rng = np.random.default_rng(seed)
    # Use a grid of 4x4x4 = 64 points, select 50
    grid_size = 4
    coords = np.linspace(-0.45, 0.45, grid_size)
    grid_points = np.array([(x, y, z) for x in coords for y in coords for z in coords])
    
    # Add jitter
    jitter = rng.uniform(-0.03, 0.03, size=(len(grid_points), 3))
    grid_points = grid_points + jitter
    
    # Select n_points points (shuffle and take first n)
    indices = rng.permutation(len(grid_points))[:n_points]
    return grid_points[indices]

def generate_random_points(n_points, seed=0):
    """Generate random points in [-0.45, 0.45]^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(n_points, 3))

def optimize_repulsion(centers, max_iter=1000, base_lr=0.05, decay=0.999):
    """Fast vectorized repulsion optimization."""
    centers = centers.copy().astype(float)
    lr = base_lr
    
    for it in range(max_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        boundary_dists = 0.5 - np.abs(centers)
        boundary_min = np.min(boundary_dists, axis=1)
        
        r_min = min(np.min(dists) / 2, np.min(boundary_min))
        if r_min <= 0:
            centers = np.clip(centers, -0.49, 0.49)
            continue
        
        # Pairwise forces (vectorized)
        push_mask = (dists < 2 * r_min) & ~np.eye(N, dtype=bool)
        mag_factor = np.zeros((N, N))
        mag_factor[push_mask] = (2 * r_min - dists[push_mask]) * 0.5 / dists[push_mask]
        forces = np.sum(mag_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        for i in range(N):
            for k in range(3):
                bd = boundary_dists[i, k]
                if bd < r_min:
                    sign = 1 if centers[i, k] > 0 else -1
                    forces[i, k] += sign * (r_min - bd)
        
        centers += lr * forces
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        lr *= decay
    
    return centers

def polish_lbfgs(centers, max_iter=200, ftol=1e-14):
    """Polish using L-BFGS-B with soft-min objective."""
    try:
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Use increasing beta for annealing
        for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
            def obj(x):
                return soft_min_score(x, beta=beta)
            
            res = minimize(obj, centers.flatten(), method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter // len([20,50,100,200,500]), 
                                   'ftol': ftol, 'gtol': 1e-8})
            if res.success or res.fun < obj(centers.flatten()):
                centers = res.x.reshape(N, 3)
                centers = np.clip(centers, -0.5 + 1e-12, 0.5 - 1e-12)
        return centers
    except:
        return centers

def local_search(centers, max_iter=50, step_size=0.01):
    """Local search: perturb individual points to escape local minima."""
    centers = centers.copy()
    current_score = compute_score(centers)
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        # Pick a random point to perturb
        idx = rng.integers(0, N)
        original = centers[idx].copy()
        
        # Try random perturbations
        for _ in range(20):
            centers[idx] = original + rng.normal(0, step_size, size=3)
            centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
            new_score = compute_score(centers)
            if new_score > current_score:
                current_score = new_score
                break
        else:
            # Revert if no improvement
            centers[idx] = original
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_r = -np.inf
    best_centers = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Fibonacci lattice
    inits.append(fibonacci_lattice(N, seed=1))
    
    # 2. Halton sequences
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    scale = 0.9
    initial2 = np.column_stack((
        h1 * scale - scale/2,
        h2 * scale - scale/2,
        h3 * scale - scale/2
    ))
    rng_main = np.random.default_rng(42)
    initial2 += rng_main.normal(0, 0.01, size=(N, 3))
    initial2 = np.clip(initial2, -0.5, 0.5)
    inits.append(initial2)
    
    # 3. Jittered grid
    inits.append(generate_jittered_grid(N, seed=1))
    
    # 4. Another Halton with different bases
    h1b = halton_sequence(N, 3)
    h2b = halton_sequence(N, 5)
    h3b = halton_sequence(N, 7)
    initial4 = np.column_stack((
        h1b * scale - scale/2,
        h2b * scale - scale/2,
        h3b * scale - scale/2
    ))
    initial4 += rng_main.normal(0, 0.01, size=(N, 3))
    initial4 = np.clip(initial4, -0.5, 0.5)
    inits.append(initial4)
    
    # 5. Random
    inits.append(generate_random_points(N, seed=2))
    
    # 6. Another jittered grid
    inits.append(generate_jittered_grid(N, seed=3))
    
    # 7. Another Fibonacci lattice
    inits.append(fibonacci_lattice(N, seed=7))
    
    # 8. Jittered 4x4x4 grid (original)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 9. Various random inits
    for seed in range(5):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 10. Jittered grids with different jitter
    for jit in [0.02, 0.05, 0.1]:
        rng = np.random.RandomState(100 + int(jit*100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # Optimize each init
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Stage 1: Repulsion pre-optimization
        result = optimize_repulsion(init, max_iter=500, base_lr=0.05, decay=0.9995)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # Stage 2: L-BFGS-B polish with soft-min
        if time.time() < deadline - 30 and r > 0.12:
            polished = polish_lbfgs(result, max_iter=150, ftol=1e-14)
            r_polished = compute_score(polished)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = polished.copy()
        
        # Stage 3: Local search
        if time.time() < deadline - 15 and best_r > 0.13:
            local = local_search(best_centers, max_iter=30, step_size=0.005)
            r_local = compute_score(local)
            if r_local > best_r:
                best_r = r_local
                best_centers = local.copy()
                
                # Final polish after local search
                if time.time() < deadline - 10:
                    final_polished = polish_lbfgs(best_centers, max_iter=100, ftol=1e-15)
                    r_final = compute_score(final_polished)
                    if r_final > best_r:
                        best_r = r_final
                        best_centers = final_polished.copy()
    
    return best_centers
