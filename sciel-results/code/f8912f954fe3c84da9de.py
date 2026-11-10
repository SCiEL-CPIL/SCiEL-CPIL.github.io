"""Improved packing of 50 equal spheres in a unit cube using gradient-based optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    # Compute all pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    a = min_pair / 2.0
    b = min_wall
    m = min(a, b)
    return m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta

def _objective_and_grad(x, beta=50.0):
    """
    Objective and gradient for L-BFGS-B.
    We minimize -soft_min_score.
    The soft min is: min(min_pair/2, min_wall) approximated smoothly.
    
    Actually, let's use a different approach: minimize a potential that penalizes
    distances below a threshold. We'll use the sum of (r_target - d)^2 for all pairs
    where d < r_target, plus wall violations.
    
    Better: use the log-sum-exp of inverse distances to approximate min distance.
    """
    centers = x.reshape(N, 3)
    centers_clipped = np.clip(centers, -0.5, 0.5)
    
    # Wall distances
    coords = np.abs(centers_clipped)
    wall_dists = 0.5 - coords  # shape (N, 3)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centers_clipped, centers_clipped)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    a = min_pair / 2.0
    b = min_wall
    m = min(a, b)
    
    # Soft min value
    soft_min = m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta
    
    # Gradient: we need d(soft_min)/dx
    # This is complex. Let's use a simpler objective: 
    # minimize sum of max(0, r_target - d_ij)^2 + sum of max(0, r_target - wall_i)^2
    # with r_target being a target radius we try to achieve.
    
    # Alternative: use the "sum of inverse squared distances" approach
    # but that doesn't directly maximize the minimum.
    
    # Let's stick with the soft min but compute gradient numerically? No, too slow.
    
    # Better approach: use the fact that at optimum, the minimum distance is determined
    # by a few critical pairs. We can use a penalty method.
    
    # For now, let's just return the negative soft min and let L-BFGS-B use numerical gradients.
    # This is slower but correct.
    
    return -soft_min

def _hilbert_initial():
    """
    Initialize using a Hilbert curve space-filling curve.
    Map 1D indices to 3D points along a Hilbert curve in the unit cube.
    """
    # For simplicity, use a grid-based initialization but with some randomness
    # to break symmetry. A true Hilbert curve implementation is complex.
    # Let's use a 4x4x4 grid with slight random perturbations.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random perturbations to break symmetry
    centers += np.random.uniform(-0.01, 0.01, (N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _bcc_initial():
    """
    Initialize with a BCC-like lattice.
    BCC has 2 points per unit cell. For a 4x4x4 grid of BCC cells, we get 128 points.
    Take the first 50.
    """
    m = 4
    # BCC lattice: corners + center of each cube
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                # Corner
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                # Center
                points.append(((x + 1.0) / m - 0.5, (y + 1.0) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.01, 0.01, (N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _repulsive_dynamics(centers, time_limit=5.0, max_steps=2000):
    """
    Repulsive dynamics to expand the packing.
    """
    current = centers.copy()
    start_time = time.time()
    
    k_repel = 0.01
    k_wall = 0.01
    dt = 0.01
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    d = np.linalg.norm(diff)
                f_mag = k_repel / (d * d)
                f_vec = (diff / d) * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
                
        # Wall repulsion
        for i in range(N):
            for axis in range(3):
                d_pos = 0.5 - current[i, axis]
                if d_pos < 0.1:
                    f_mag = k_wall / (d_pos * d_pos + 1e-6)
                    forces[i, axis] -= f_mag
                
                d_neg = 0.5 + current[i, axis]
                if d_neg < 0.01:
                    f_mag = k_wall / (d_neg * d_neg + 1e-6)
                    forces[i, axis] += f_mag
        
        current += forces * dt
        current = np.clip(current, -0.5, 0.5)
        
    return current

def _kick_local_search(centers, time_limit=5.0, max_iter=10000):
    """
    Local search that resolves overlaps by kicking spheres apart.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find minimum distance pair
        from scipy.spatial.distance import cdist
        dists = cdist(current, current)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_idx_i, min_idx_j = min_idx
        
        # Check wall distance
        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        
        current_radius = min(min_d / 2.0, min_wall)
        
        if current_radius > best_score:
            best = current.copy()
            best_score = current_radius
        
        # If there is an overlap, kick them apart
        if min_d < 2.0 * best_score * 1.01:
            diff = current[min_idx_i] - current[min_idx_j]
            d = np.linalg.norm(diff)
            if d > 1e-6:
                push = (diff / d) * (2.0 * best_score * 1.01 - min_d) * 0.5
                current[min_idx_i] += push
                current[min_idx_j] -= push
        
        # Handle wall collisions
        for i in range(N):
            for axis in range(3):
                if current[i, axis] > 0.5 - best_score * 1.01:
                    current[i, axis] = 0.5 - best_score * 1.01
                elif current[i, axis] < -0.5 + best_score * 1.01:
                    current[i, axis] = -0.5 + best_score * 1.01
        
        # Small random perturbation
        if it % 100 == 0:
            idx = np.random.randint(N)
            current[idx] += np.random.uniform(-0.005, 0.005, 3)
            current = np.clip(current, -0.5, 0.5)
            
    return best, best_score

def _l_bfgs_refine(centers, time_limit=10.0):
    """Refine using L-BFGS-B."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Coarse pass
    if time.time() - start_time < time_limit * 0.5:
        result1 = minimize(
            _objective_and_grad, x0, args=(20.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x0 = result1.x
    
    # Fine pass
    if time.time() - start_time < time_limit * 0.9:
        result2 = minimize(
            _objective_and_grad, x0, args=(80.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x0 = result2.x
    
    return np.clip(x0.reshape(N, 3), -0.5, 0.5)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Try multiple initializations
    initializations = [
        _hilbert_initial(),
        _bcc_initial()
    ]
    
    best_centers = None
    best_score = 0.0
    
    for init in initializations:
        if time.time() > deadline - 60.0:
            break
        
        centers = init.copy()
        score = _exact_score(centers)
        
        # Repulsive dynamics
        if time.time() < deadline - 40.0:
            dyn_centers = _repulsive_dynamics(centers, time_limit=10.0)
            dyn_score = _exact_score(dyn_centers)
            if dyn_score > score:
                score = dyn_score
                centers = dyn_centers
        
        # Kick local search
        if time.time() < deadline - 20.0:
            ls_centers, ls_score = _kick_local_search(centers, time_limit=10.0)
            if ls_score > score:
                score = ls_score
                centers = ls_centers
        
        # L-BFGS-B refinement
        if time.time() < deadline - 10.0:
            refined = _l_bfgs_refine(centers, time_limit=15.0)
            ref_score = _exact_score(refined)
            if ref_score > score:
                score = ref_score
                centers = refined
        
        if best_centers is None or score > best_score:
            best_score = score
            best_centers = centers
    
    # Final polish
    if time.time() < deadline - 5.0 and best_centers is not None:
        final_centers, final_score = _kick_local_search(best_centers, time_limit=5.0)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
            
    if best_centers is None:
        # Fallback
        best_centers = _hilbert_initial()
        
    return best_centers
# EVOLVE-BLOCK-END
