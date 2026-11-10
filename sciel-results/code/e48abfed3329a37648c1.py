"""Improved packing of 50 equal spheres in a unit cube using bisection + projected gradient."""
import numpy as np
import time
import math

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

def _project_constraints(centers, r):
    """
    Project centers onto the feasible set:
    - All spheres must be inside [-1/2, 1/2]^3 with radius r
    - All pairs must be at least 2r apart
    
    This is a simplified projection that handles wall constraints exactly
    and pair constraints approximately via iterative relaxation.
    """
    current = centers.copy()
    
    # Wall constraints: each coordinate must be in [-0.5+r, 0.5-r]
    lo = -0.5 + r
    hi = 0.5 - r
    current = np.clip(current, lo, hi)
    
    # Pair constraints: iterative relaxation
    for iteration in range(50):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                min_d = 2.0 * r
                if d < min_d:
                    if d < 1e-10:
                        # Random small perturbation
                        diff = np.random.uniform(-0.01, 0.01, 3)
                        d = np.linalg.norm(diff)
                    # Push apart
                    overlap = (min_d - d) / 2.0
                    push = (diff / d) * overlap
                    current[i] += push
                    current[j] -= push
                    moved = True
        
        # Re-apply wall constraints
        current = np.clip(current, lo, hi)
        
        if not moved:
            break
    
    return current

def _feasibility_check(centers, r):
    """Check if centers are feasible for radius r."""
    # Wall check
    coords = np.abs(centers)
    if np.any(coords > 0.5 - r + 1e-12):
        return False
    
    # Pair check
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 2.0 * r - 1e-12:
        return False
    
    return True

def _bisection_search(centers, r_low, r_high, time_limit=30.0):
    """
    Bisection search on radius r.
    For each r, try to find a feasible configuration via projected gradient.
    """
    start_time = time.time()
    best_feasible = None
    best_r = r_low
    
    # First check if r_high is feasible
    projected = _project_constraints(centers, r_high)
    if _feasibility_check(projected, r_high):
        best_feasible = projected
        best_r = r_high
        return best_r, best_feasible
    
    # Bisection
    max_iters = 20
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        projected = _project_constraints(centers, r_mid)
        
        if _feasibility_check(projected, r_mid):
            best_feasible = projected
            best_r = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_r, best_feasible

def _gradient_ascent(centers, r, time_limit=10.0, lr=0.001):
    """
    Gradient ascent to maximize minimum distance for a fixed radius r.
    We want to increase the minimum pairwise distance and wall distance.
    """
    current = centers.copy()
    start_time = time.time()
    best = current.copy()
    best_min_dist = _compute_min_dist(current, r)
    
    for step in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient of the objective (min distance)
        # Use a smooth approximation: sum of exp(-d/epsilon) for close pairs
        epsilon = 0.05
        grad = np.zeros_like(current)
        
        # Pairwise gradients
        from scipy.spatial.distance import cdist
        dists = cdist(current, current)
        np.fill_diagonal(dists, np.inf)
        
        # Find close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 3.0 * r:
                    weight = math.exp(-d / epsilon)
                    diff = current[i] - current[j]
                    if d > 1e-10:
                        unit = diff / d
                        grad[i] += weight * unit
                        grad[j] -= weight * unit
        
        # Wall gradients
        for i in range(N):
            for axis in range(3):
                pos_dist = 0.5 - current[i, axis]
                neg_dist = 0.5 + current[i, axis]
                if pos_dist < r * 1.5:
                    grad[i, axis] -= math.exp(-pos_dist / epsilon)
                if neg_dist < r * 1.5:
                    grad[i, axis] += math.exp(-neg_dist / epsilon)
        
        # Update
        current += lr * grad
        current = np.clip(current, -0.5 + r, 0.5 - r)
        
        # Check improvement
        current_min = _compute_min_dist(current, r)
        if current_min > best_min_dist:
            best = current.copy()
            best_min_dist = current_min
        
        # Adaptive learning rate
        if step % 100 == 0:
            lr *= 0.95
    
    return best

def _compute_min_dist(centers, r):
    """Compute the minimum of pairwise distances/2 and wall distances."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _initial_grid(m=4):
    """Initialize with a regular grid."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.uniform(-0.005, 0.005, (N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _initial_bcc(m=4):
    """Initialize with BCC lattice."""
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                points.append(((x + 1.0) / m - 0.5, (y + 1.0) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.005, 0.005, (N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _repulsive_dynamics(centers, time_limit=5.0, max_steps=2000):
    """Repulsive dynamics to expand the packing."""
    current = centers.copy()
    start_time = time.time()
    
    k_repel = 0.005
    k_wall = 0.005
    dt = 0.005
    
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
                if d_pos < 0.15:
                    f_mag = k_wall / (d_pos * d_pos + 1e-6)
                    forces[i, axis] -= f_mag
                
                d_neg = 0.5 + current[i, axis]
                if d_neg < 0.15:
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

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Try multiple initializations
    initializations = [
        _initial_grid(),
        _initial_bcc()
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
            dyn_centers = _repulsive_dynamics(centers, time_limit=8.0)
            dyn_score = _exact_score(dyn_centers)
            if dyn_score > score:
                score = dyn_score
                centers = dyn_centers
        
        # Kick local search
        if time.time() < deadline - 25.0:
            ls_centers, ls_score = _kick_local_search(centers, time_limit=10.0)
            if ls_score > score:
                score = ls_score
                centers = ls_centers
        
        # Bisection search with projection
        if time.time() < deadline - 20.0:
            r_low = score
            r_high = 0.14  # Slightly above known best
            r_found, feasible_centers = _bisection_search(centers, r_low, r_high, time_limit=15.0)
            if feasible_centers is not None:
                feas_score = _exact_score(feasible_centers)
                if feas_score > score:
                    score = feas_score
                    centers = feasible_centers
        
        # Final kick local search
        if time.time() < deadline - 10.0:
            final_centers, final_score = _kick_local_search(centers, time_limit=8.0)
            if final_score > score:
                score = final_score
                centers = final_centers
        
        if best_centers is None or score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = _initial_grid()
        
    return best_centers
# EVOLVE-BLOCK-END
