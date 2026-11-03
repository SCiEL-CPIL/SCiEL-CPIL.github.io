"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(points):
    """Compute the maximum radius for the given set of centers."""
    if len(points) == 0:
        return 0.0
    points = np.asarray(points)
    # Distance to boundaries
    # Square is [-0.5, 0.5] in both x and y
    dist_x = np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0])
    dist_y = np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    
    # Pairwise distances
    # For N=200, O(N^2) is 40,000 operations, which is fast.
    # Use broadcasting for vectorization
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = 0.5 * min_dist
    r = min(r, np.min(dist_boundary))
    return r

def _hex_lattice_points(a, b, angle_rad, offset_x, offset_y, max_coord=0.6):
    """
    Generate points from a hexagonal lattice.
    a, b: lengths of basis vectors
    angle_rad: angle between them
    offset_x, offset_y: translation
    """
    # Basis vectors
    v1 = np.array([a, 0.0])
    v2 = np.array([b * np.cos(angle_rad), b * np.sin(angle_rad)])
    
    # Determine range of indices to cover the square [-0.5, 0.5]
    # To be safe, cover a slightly larger area
    max_idx = int(np.ceil(max_coord / min(a, b))) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2 + np.array([offset_x, offset_y])
            # Keep points inside or very close to the square
            # We will filter strictly later, but keeping a bit of margin helps
            # Actually, for the "super-lattice" approach, we want to select a subset
            # that fits best. But a standard hex lattice is periodic.
            # A better approach for "super-lattice" is to generate a dense set
            # and then use a greedy or optimization selection?
            # No, the "super-lattice" idea from the prompt implies that the
            # *parameters* of the lattice are optimized, and the points are
            # taken from this lattice.
            # However, a simple hex lattice with fixed N might not fill the square
            # optimally if the aspect ratio is wrong.
            # Let's stick to generating points in the square from the lattice.
            if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                points.append(p)
    
    if not points:
        return np.array([])
    return np.array(points)

def _select_best_subset(points, N_target):
    """
    Given a set of candidate points (possibly more than N_target),
    select N_target points that maximize the minimum distance.
    This is a hard problem, but for a lattice, we can just take the first N
    if the lattice is uniform? No, corners are important.
    
    Alternative: Since we are optimizing lattice parameters, we can just
    generate exactly N points by adjusting the lattice density?
    
    Let's refine the strategy:
    1. Generate a hexagonal lattice with parameters (a, angle, offset).
    2. Filter points inside the square.
    3. If count > N, we have a problem. We need exactly N.
    4. If count < N, we have a problem.
    
    To ensure exactly N points, we can use a "jittered" grid or a specific
    construction.
    
    Better Structural Change:
    Use a **Force-Directed Graph (Repulsion)** on a set of N particles
    initialized in a hexagonal pattern, but allow the *entire lattice*
    to deform via a global "strain" field?
    
    Let's go with **Global Annealing of Lattice Parameters** but handle
    the count mismatch by using a **Greedy Selection** from a dense super-lattice.
    If we generate a very dense hex lattice (e.g., 300 points), we can select
    the 200 that maximize the minimum distance.
    This is still O(M*N) or similar.
    
    Actually, a simpler and more effective structural change is:
    **Simulated Annealing on the coordinates** is what the seed basically does (or tries to).
    The prompt suggests "different design family".
    
    Let's try **Deterministic Relaxation with Momentum (L-BFGS style logic manually implemented)**
    on the objective function: Minimize -min_dist.
    But min_dist is non-differentiable.
    We can use a soft-min: -log(sum(exp(-d^2 / T)))
    
    Let's implement a **Gradient Descent with Momentum** on the soft-min objective.
    Initialization: Hexagonal grid.
    This is different from the "random start + SA" or "static grid".
    It uses the gradient of a smooth surrogate to escape local minima more effectively
    than random walks, especially in the early stages.
    """
    # This function is a placeholder for the actual strategy below
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal grid
    # Approximate optimal hex packing for N points in a square.
    # Area per point ~ 1/200. Side ~ 1/sqrt(200) ~ 0.07.
    # Hex lattice spacing a. Area per point = a^2 * sqrt(3)/2.
    # a = sqrt(2 / (sqrt(3) * 200)) ~ 0.075
    
    a_init = math.sqrt(2.0 / (math.sqrt(3.0) * N))
    
    # Generate initial hex grid
    points = []
    rows = int(np.ceil(2.0 / (a_init * math.sqrt(3) / 2.0))) + 2
    cols = int(np.ceil(2.0 / a_init)) + 2
    
    for i in range(-rows//2, rows//2 + 1):
        for j in range(-cols//2, cols//2 + 1):
            x = j * a_init + (i % 2) * (a_init / 2.0)
            y = i * (a_init * math.sqrt(3) / 2.0)
            # Center the grid
            # We want to center the cloud of points.
            # Rough centering:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many or too few, adjust
    # For simplicity, let's just take the first N if we have enough,
    # or pad if not (though hex grid usually gives enough).
    # To ensure exactly N, we can do a greedy selection if len(points) > N.
    if len(points) > N:
        # Greedy selection: Start with the point closest to center?
        # Or just take a random subset?
        # Better: Take the points that form the most regular subset.
        # For now, let's just slice, but that might break symmetry.
        # Let's use a simple "keep the ones with highest local density" or just random.
        # Actually, for a hex grid, the "corners" of the grid are often cut off.
        # Let's try to center the grid better.
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        points = points - centroid
        # Re-filter
        points = points[(points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)]
        
        if len(points) > N:
            # Random selection of N points
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Pad with random points in the square
            missing = N - len(points)
            new_points = np.random.uniform(-0.5, 0.5, (missing, 2))
            points = np.vstack([points, new_points])
    
    # Ensure exactly N
    if len(points) != N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    best_points = points.copy()
    best_score = _compute_radius(best_points)
    
    # 2. Optimization: Gradient Descent with Momentum on Soft-Min Objective
    # Objective: Maximize min_dist.
    # Surrogate: F(points) = -1/T * log( sum_{i<j} exp(-dist_ij^2 / T) ) - lambda * boundary_violation
    # Gradient of soft-min:
    # d/dx_k [ -1/T log(sum exp(-d^2/T)) ] = -1/T * (1/sum) * sum ( exp(-d^2/T) * d/dx_k (-d^2/T) )
    # = -1/T * (1/sum) * sum ( exp(-d^2/T) * (-2/T) * d * (x_i - x_j) )
    # = (2/T^2) * (1/sum) * sum ( exp(-d^2/T) * d * (x_i - x_j) )
    # This points in the direction of increasing the distances (repulsion).
    
    T = 0.01  # Temperature
    lr = 0.001
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    # Precompute indices for pairs
    n = len(points)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    while time.time() < deadline:
        # Calculate distances
        diff = points[i_idx] - points[j_idx]
        dists_sq = np.sum(diff ** 2, axis=1)
        dists = np.sqrt(dists_sq)
        
        # Soft-min weights
        # Avoid division by zero or underflow
        exp_terms = np.exp(-dists_sq / T)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            break
            
        # Gradient of the soft-min term (repulsion)
        # grad = (2/T^2) * (exp_terms / sum_exp) * dists * diff
        # Note: diff is x_i - x_j.
        # We want to push i away from j and j away from i.
        # Force on i: + grad_i
        # Force on j: - grad_i (since d/dx_j (-d^2/T) = ... )
        
        weights = (2.0 / (T * T)) * (exp_terms / sum_exp) * dists
        # weights shape (M,)
        # Force vector for each pair
        forces = weights[:, np.newaxis] * diff
        
        # Accumulate forces
        grad = np.zeros_like(points)
        np.add.at(grad, i_idx, forces)
        np.add.at(grad, j_idx, -forces)
        
        # Boundary Penalty
        # Push points away from boundaries
        # dist_x = min(x + 0.5, 0.5 - x)
        # If x < -0.5 + r, push right. If x > 0.5 - r, push left.
        # Since r is not fixed, we use a soft boundary.
        # Penalty: sum exp(-(dist_to_boundary)^2 / T_b)
        # Let's just add a force proportional to how close to the edge.
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Distance to left/right
        d_left = x + 0.5
        d_right = 0.5 - x
        d_bottom = y + 0.5
        d_top = 0.5 - y
        
        # Force in x:
        # If d_left < T_b, push right (+). If d_right < T_b, push left (-).
        T_b = 0.1
        force_x = np.zeros(n)
        mask_l = d_left < T_b
        force_x[mask_l] += (T_b - d_left[mask_l]) / T_b
        mask_r = d_right < T_b
        force_x[mask_r] -= (T_b - d_right[mask_r]) / T_b
        
        force_y = np.zeros(n)
        mask_b = d_bottom < T_b
        force_y[mask_b] += (T_b - d_bottom[mask_b]) / T_b
        mask_t = d_top < T_b
        force_y[mask_t] -= (T_b - d_top[mask_t]) / T_b
        
        # Add boundary forces to gradient
        grad[:, 0] += force_x
        grad[:, 1] += force_y
        
        # Update velocity and position
        velocity = momentum * velocity + lr * grad
        points = points + velocity
        
        # Clamp to square (hard constraint)
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Evaluate
        current_score = _compute_radius(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Cool down slightly?
        # T = max(0.001, T * 0.99)
        # lr = max(0.0001, lr * 0.99)
        
        # Check for convergence or stagnation
        # If we haven't improved in a while, break?
        # For now, just let it run until deadline.
        
    return best_points

# EVOLVE-BLOCK-END
