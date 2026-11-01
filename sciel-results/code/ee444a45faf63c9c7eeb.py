import numpy as np
import time

N = 50

def get_radius(points):
    """
    Computes the radius of the largest circle that can be placed at each point
    such that it doesn't overlap with others or the boundary.
    Returns the minimum such radius.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.min(radii)
    
    return min(min_pairwise / 2.0, min_boundary)

def soft_min_grad(points, alpha):
    """
    Computes the gradient direction to maximize the soft-min of distances.
    """
    n = len(points)
    
    # 1. Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1e-9)
    
    mask = ~np.eye(n, dtype=bool)
    
    # Weights w = exp(-alpha * d)
    w_pair = np.zeros_like(dists)
    w_pair[mask] = np.exp(-alpha * dists[mask])
    
    sum_w_pair = np.sum(w_pair)
    
    # Gradient of distance d_ij w.r.t point i is (p_i - p_j) / d_ij
    vecs = (points[:, np.newaxis, :] - points[np.newaxis, :, :]) / dists[:, :, np.newaxis]
    
    # grad_pair[i] = sum_j w_pair[i, j] * vecs[i, j]
    grad_pair = np.sum(w_pair[:, :, np.newaxis] * vecs, axis=1)
    
    # 2. Boundary distances
    radii = np.linalg.norm(points, axis=1)
    b_dists = 1.0 - radii
    
    w_bound = np.exp(-alpha * b_dists)
    sum_w_bound = np.sum(w_bound)
    
    # Gradient of b_i w.r.t p_i is -p_i / |p_i|
    norms = radii
    norms[norms < 1e-9] = 1e-9
    unit_vecs = points / norms[:, np.newaxis]
    
    # grad_bound[i] = w_bound[i] * (-unit_vecs[i])
    grad_bound = -w_bound[:, np.newaxis] * unit_vecs
    
    total_weight = sum_w_pair + sum_w_bound
    if total_weight < 1e-10:
        return np.zeros_like(points)
        
    grad_E = (grad_pair + grad_bound) / total_weight
    
    # We want to maximize distance, so minimize E.
    # Move in direction -grad_E.
    return -grad_E

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_points = None
    best_radius = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal Lattice
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        # Sort by radius to take central ones
        points = points[np.argsort(radius, kind="stable")[:N]]
        inits.append(points)
        
    # 2. Random starts
    np.random.seed(42)
    for _ in range(5):
        # Generate random points
        pts = np.random.uniform(-1, 1, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.8
        pts[mask] *= (0.8 / norms[mask])[:, np.newaxis]
        inits.append(pts)
        
    # 3. Ring structure
    # Center
    pts = np.zeros((1, 2))
    # Ring 1: 6
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts = np.vstack([pts, [0.3 * np.cos(angle), 0.3 * np.sin(angle)]])
    # Ring 2: 12
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts = np.vstack([pts, [0.6 * np.cos(angle), 0.6 * np.sin(angle)]])
    # Ring 3: remaining 31
    rem = N - len(pts)
    for i in range(rem):
        angle = 2 * np.pi * i / rem
        pts = np.vstack([pts, [0.85 * np.cos(angle), 0.85 * np.sin(angle)]])
    inits.append(pts)
    
    # Run optimization on each init
    for init in inits:
        if time.time() > deadline:
            break
            
        points = init.copy()
        current_radius = get_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        alpha = 20.0
        step_size = 0.05
        
        # Gradient Ascent Loop
        for i in range(50000): # Steps per run
            if time.time() > deadline:
                break
                
            grad = soft_min_grad(points, alpha)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-9:
                break
            grad = grad / grad_norm
                
            # Update
            new_points = points + step_size * grad
            
            # Project back to unit circle if needed
            new_radii = np.linalg.norm(new_points, axis=1)
            mask_out = new_radii > 1.0
            if np.any(mask_out):
                new_points[mask_out] *= (1.0 / new_radii[mask_out])[:, np.newaxis]
                
            new_radius = get_radius(new_points)
            
            if new_radius > current_radius:
                points = new_points
                current_radius = new_radius
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                    
                # Occasionally increase alpha to sharpen the min
                if i % 1000 == 0 and alpha < 100:
                    alpha *= 1.2
                # Decrease step size
                step_size *= 0.9995
            else:
                # If stuck, try a random kick
                if i % 500 == 0:
                    kick = np.random.normal(0, 0.01, (N, 2))
                    points = points + kick
                    # Ensure valid
                    norms = np.linalg.norm(points, axis=1)
                    mask = norms > 1.0
                    points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
                    current_radius = get_radius(points)
                    if current_radius > best_radius:
                        best_radius = current_radius
                        best_points = points.copy()

    if best_points is None:
        # Fallback
        return np.zeros((N, 2))
        
    return best_points
