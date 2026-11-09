# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial.distance as spd
import scipy.optimize as opt
import time

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective(points_flat):
    """
    Objective function to minimize: -log(d_min) + log(d_max)
    which is equivalent to minimizing -log(d_min/d_max)
    """
    points = points_flat.reshape(-1, 2)
    # Normalize to avoid scale drift, though ratio is scale invariant
    # It's often better to keep the scale roughly constant to help the optimizer
    # Let's normalize so that the mean squared distance from centroid is 1
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    norm = np.sqrt(np.sum(points_centered**2))
    if norm < 1e-6:
        return 1e6
    points_norm = points_centered / norm
    
    dists = spd.pdist(points_norm)
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_min <= 0:
        return 1e6
    if d_max <= 0:
        return 1e6
        
    # We want to maximize d_min/d_max, so minimize -log(d_min) + log(d_max)
    return -np.log(d_min) + np.log(d_max)

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    deadline = time.time() + 590  # Safety margin

    best_score = -np.inf
    best_points = None

    # Initializations
    # 1. Hexagonal grid
    # 2. Random points
    # 3. Symmetric polygon + inner points
    
    inits = []
    
    # Hexagonal grid 4x4
    rows, cols = 4, 4
    dx = 1.0
    dy = dx * np.sqrt(3) / 2
    hex_points = []
    for i in range(rows):
        for j in range(cols):
            x = j * dx + (i % 2) * (dx / 2)
            y = i * dy
            hex_points.append([x, y])
    inits.append(np.array(hex_points))
    
    # Random inits
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.randn(n, d))
        
    # Circle based init
    # 8 points on outer circle, 8 on inner circle
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    r_outer = 1.0
    r_inner = 0.5
    circle_points = np.vstack([
        np.c_[r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)],
        np.c_[r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)]
    ])
    inits.append(circle_points)
    
    # 4x4 grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_points = np.c_[grid_x.flatten(), grid_y.flatten()]
    inits.append(grid_points)

    for init in inits:
        if time.time() > deadline:
            break
            
        # Normalize init
        centroid = np.mean(init, axis=0)
        init_centered = init - centroid
        norm = np.sqrt(np.sum(init_centered**2))
        if norm > 1e-6:
            init_norm = init_centered / norm
        else:
            init_norm = init_centered + 0.01*np.random.randn(n, d)
            
        x0 = init_norm.flatten()
        
        try:
            # Use L-BFGS-B
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success or res.fun < 1e5:
                points_opt = res.x.reshape(n, d)
                # Restore scale (arbitrary, ratio is invariant)
                score = calculate_score(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt
        except Exception:
            pass

    # If no good result, fallback to best init score
    if best_points is None:
        for init in inits:
            score = calculate_score(init)
            if score > best_score:
                best_score = score
                best_points = init

    return best_points

# EVOLVE-BLOCK-END
