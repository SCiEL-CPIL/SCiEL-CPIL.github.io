import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    # centers shape (N, 3)
    # return the actual radius
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    # The score definition: min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    # Note: min_k (1/2 - |c_ik|) is 0.5 - max_k |c_ik|
    
    # Distances between pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distances
    max_abs = np.max(np.abs(centers), axis=1)
    boundary_dists = 0.5 - max_abs
    min_boundary = np.min(boundary_dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def md_step(centers, velocities, dt, r_target):
    """
    Perform one MD step with hard-sphere collisions and wall reflections.
    Returns updated centers and velocities.
    """
    N = len(centers)
    
    # Update positions
    new_centers = centers + velocities * dt
    
    # Handle wall collisions: reflect velocity if outside bounds
    for i in range(N):
        for k in range(3):
            if new_centers[i, k] < -0.5 + r_target:
                new_centers[i, k] = -0.5 + r_target + (-0.5 + r_target - new_centers[i, k])
                velocities[i, k] = -velocities[i, k]
            elif new_centers[i, k] > 0.5 - r_target:
                new_centers[i, k] = 0.5 - r_target - (new_centers[i, k] - (0.5 - r_target))
                velocities[i, k] = -velocities[i, k]
    
    # Handle sphere-sphere collisions
    # For each pair, check if they overlap and resolve
    for i in range(N):
        for j in range(i+1, N):
            diff = new_centers[j] - new_centers[i]
            dist = np.linalg.norm(diff)
            min_dist = 2 * r_target
            
            if dist < min_dist and dist > 1e-12:
                # Normal direction
                n = diff / dist
                # Relative velocity along normal
                rel_v = np.dot(velocities[j] - velocities[i], n)
                
                if rel_v < 0:  # Moving toward each other
                    # Elastic collision: swap velocity components along normal
                    impulse = rel_v
                    velocities[i] += impulse * n
                    velocities[j] -= impulse * n
                
                # Separate spheres
                overlap = min_dist - dist
                new_centers[i] -= n * (overlap / 2)
                new_centers[j] += n * (overlap / 2)
    
    return new_centers, velocities

def run_md(initial_centers, r_target, n_steps=10000, dt=0.001, seed=42):
    """
    Run MD simulation with hard spheres of radius r_target.
    """
    np.random.seed(seed)
    centers = initial_centers.copy()
    velocities = np.random.normal(0, 0.01, (N, 3))
    
    for step in range(n_steps):
        centers, velocities = md_step(centers, velocities, dt, r_target)
        
        # Add small random perturbation to escape local minima
        if step % 100 == 0:
            velocities += np.random.normal(0, 0.001, (N, 3))
        
        # Dampen velocities slightly to help convergence
        velocities *= 0.999
    
    return centers

def polish_gradient(centers, r_target, max_iter=500):
    """
    Final gradient-based polish to maximize minimum distance.
    """
    def objective(centers_flat):
        centers = centers_flat.reshape(N, 3)
        
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Soft min of pairwise distances
        eps = 0.01
        d_soft = -eps * np.log(np.sum(np.exp(-dists / eps)))
        
        # Boundary constraint
        max_abs = np.max(np.abs(centers), axis=1)
        b = 0.5 - max_abs
        b_soft = -eps * np.log(np.sum(np.exp(-b / eps)))
        
        # Combined soft radius
        A = d_soft / 2.0
        B = b_soft
        r_soft = -eps * np.log(np.exp(-A / eps) + np.exp(-B / eps))
        
        return -r_soft
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x.reshape(N, 3)
    except:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_centers = None
    best_score = 0.0
    
    # Generate initial configurations
    np.random.seed(42)
    
    # Strategy 1: Grid-based initialization
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(64, 50, replace=False)
    init1 = grid[idx].copy()
    
    # Strategy 2: Random uniform
    init2 = np.random.uniform(-0.4, 0.4, (50, 3))
    
    # Strategy 3: Jittered grid
    init3 = init1 + np.random.normal(0, 0.05, (50, 3))
    init3 = np.clip(init3, -0.45, 0.45)
    
    inits = [init1, init2, init3]
    
    # Try different target radii with MD
    target_radii = [0.13, 0.135, 0.14, 0.145]
    
    for r_target in target_radii:
        if time.time() > deadline - 20:  # Reserve 20s for polish
            break
            
        for init_idx, init in enumerate(inits):
            if time.time() > deadline - 20:
                break
                
            try:
                # Run MD
                md_centers = run_md(init, r_target, n_steps=5000, dt=0.001, seed=42 + init_idx)
                
                # Check if MD succeeded (spheres don't overlap too much)
                score_md = compute_score(md_centers)
                
                if score_md > best_score:
                    best_score = score_md
                    best_centers = md_centers.copy()
                    
            except Exception as e:
                pass
    
    # Final polish on best configuration
    if best_centers is not None and time.time() < deadline - 5:
        polished = polish_gradient(best_centers, 0.135, max_iter=300)
        score_polished = compute_score(polished)
        
        if score_polished > best_score:
            best_score = score_polished
            best_centers = polished
    
    # If no optimization ran or failed, use best init
    if best_centers is None:
        best_centers = init1
        best_score = compute_score(best_centers)
    else:
        # Check if best init was better
        for init in inits:
            s = compute_score(init)
            if s > best_score:
                best_score = s
                best_centers = init.copy()

    return best_centers
