import numpy as np
from scipy.optimize import minimize
import time

N = 50
SIDE = 1.0
HALF_SIDE = SIDE / 2.0

def compute_min_dist(X):
    """
    Compute the minimum distance between any two circle centers or between a center and the boundary.
    X: (N, 2) array of centers.
    """
    # Distances between all pairs
    # Using broadcasting for speed
    # D[i, j] = ||X[i] - X[j]||
    D = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(D, np.inf)
    min_pair_dist = np.min(D)
    
    # Distances to walls
    # Left wall: 0.5 + x
    # Right wall: 0.5 - x
    # Bottom wall: 0.5 + y
    # Top wall: 0.5 - y
    dist_left = HALF_SIDE + X[:, 0]
    dist_right = HALF_SIDE - X[:, 0]
    dist_bottom = HALF_SIDE + X[:, 1]
    dist_top = HALF_SIDE - X[:, 1]
    
    min_wall_dist = np.min(np.concatenate([dist_left, dist_right, dist_bottom, dist_top]))
    
    return min(min_pair_dist, min_wall_dist)

def gradient_min_dist(X):
    """
    Compute the gradient of the minimum distance function.
    Returns gradient with respect to X.
    """
    grad = np.zeros_like(X)
    
    # Pair distances
    D = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(D, np.inf)
    
    min_pair_dist = np.min(D)
    
    # Wall distances
    dist_left = HALF_SIDE + X[:, 0]
    dist_right = HALF_SIDE - X[:, 0]
    dist_bottom = HALF_SIDE + X[:, 1]
    dist_top = HALF_SIDE - X[:, 1]
    
    # We need to find the global minimum
    global_min = min_pair_dist
    # Check walls
    # We can just find the min of all wall dists
    wall_dists = np.column_stack([dist_left, dist_right, dist_bottom, dist_top])
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < global_min:
        global_min = min_wall_dist
    
    # Accumulate gradients for all active constraints (those with distance == global_min)
    # Tolerance for floating point
    tol = 1e-8
    
    # 1. Pair constraints
    # Find all pairs (i, j) with D[i, j] <= global_min + tol
    # We need to be careful not to double count or handle i==j
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] <= global_min + tol:
                diff = X[i] - X[j]
                norm = D[i, j]
                if norm > 1e-12:
                    unit_vec = diff / norm
                    # Gradient of distance w.r.t X[i] is unit_vec
                    # Gradient of distance w.r.t X[j] is -unit_vec
                    grad[i] += unit_vec
                    grad[j] -= unit_vec
                else:
                    # If coincident, random direction? Or just 0. 
                    # This shouldn't happen if we start well.
                    pass

    # 2. Wall constraints
    # For each point i, check which wall is closest
    # We need to add gradient if the distance to that wall is global_min
    for i in range(N):
        # Left
        if dist_left[i] <= global_min + tol:
            grad[i, 0] += 1.0
        # Right
        if dist_right[i] <= global_min + tol:
            grad[i, 0] -= 1.0
        # Bottom
        if dist_bottom[i] <= global_min + tol:
            grad[i, 1] += 1.0
        # Top
        if dist_top[i] <= global_min + tol:
            grad[i, 1] -= 1.0
            
    return grad

def objective(X):
    # We want to maximize min_dist, so minimize -min_dist
    return -compute_min_dist(X)

def objective_grad(X):
    return -gradient_min_dist(X)

def hex_init(N):
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate number of rows
    # Area of hex grid vs square
    # For now, let's just create a grid and jitter it
    m = int(np.ceil(np.sqrt(N)))
    # Hex spacing
    # Let's try to fit m x m grid
    # Actually, let's just use a jittered grid for simplicity and robustness
    # Hex is better but harder to map to exactly N points.
    
    # Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    # Spacing
    dx = 1.0 / m
    dy = 1.0 / m
    
    centers = []
    for i in range(m):
        for j in range(m):
            if len(centers) >= N:
                break
            # Center of cell
            x = (i + 0.5) * dx - 0.5
            y = (j + 0.5) * dy - 0.5
            # Add small jitter
            x += np.random.uniform(-0.1, 0.1) * dx
            y += np.random.uniform(-0.1, 0.1) * dy
            centers.append([x, y])
            
    return np.array(centers[:N])

def random_init(N):
    """
    Random initialization within the square.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def construct():
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    best_X = None
    best_val = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # 1. Jittered grid
    inits.append(hex_init(N))
    
    # 2. Random
    inits.append(random_init(N))
    
    # 3. Another random
    inits.append(random_init(N))
    
    # 4. Hex-like: 7x7 grid subset?
    # Let's try a denser grid and pick best N? No, just use the jittered grid.
    
    for X0 in inits:
        if time.time() - start_time > max_time:
            break
            
        # Run optimization
        try:
            res = minimize(objective, X0, jac=objective_grad, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success or res.fun < best_val:
                # Check if valid
                min_d = compute_min_dist(res.x)
                if min_d > 0: # Should be
                    val = min_d
                    if val > best_val:
                        best_val = val
                        best_X = res.x
        except Exception as e:
            pass
            
    # If we didn't find anything good, return grid
    if best_X is None:
        return hex_init(N)
        
    # Final polish? 
    # Maybe run one more time from best_X with tighter tolerances?
    if time.time() - start_time < max_time - 10:
        try:
            res = minimize(objective, best_X, jac=objective_grad, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
            if res.fun < objective(best_X):
                best_X = res.x
        except:
            pass
            
    return best_X
