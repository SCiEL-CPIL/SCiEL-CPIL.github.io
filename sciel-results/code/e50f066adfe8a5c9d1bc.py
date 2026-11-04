"""Circle packing in perimeter-4 rectangle: L-BFGS-B solver with aspect ratio search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # safety margin under 120 s wall-clock

def _get_max_radius(centers, w, h):
    """Calculate the maximum radius that fits given centers and box dimensions."""
    # Distance to walls
    dist_x = np.minimum(centers[:, 0], w - centers[:, 0])
    dist_y = np.minimum(centers[:, 1], h - centers[:, 1])
    wall_dist = np.minimum(dist_x, dist_y).min()
    
    # Distance to other circles
    # Vectorized pairwise distance
    diff = centers[:, None, :] - centers[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = dist_mat.min()
    
    return min(wall_dist, min_pair_dist / 2.0)

def _objective(x, w, h):
    """Objective function to minimize: negative of max radius."""
    centers = x.reshape(N, 2)
    r = _get_max_radius(centers, w, h)
    return -r

def _solve_for_aspect_ratio(aspect_ratio, init_centers=None):
    """Solve for a fixed aspect ratio."""
    w = 2.0 * aspect_ratio
    h = 2.0 * (1.0 - aspect_ratio)
    if w <= 0 or h <= 0:
        return None, 0.0
    
    if init_centers is None:
        # Simple grid initialization
        # Approximate number of rows/cols
        cols = int(np.sqrt(N * w / h))
        rows = int(np.ceil(N / cols))
        xs = np.linspace(0.1, 0.9, cols) * w
        ys = np.linspace(0.1, 0.9, rows) * h
        centers = []
        for i in range(N):
            r_idx = i // cols
            c_idx = i % cols
            if r_idx < rows:
                centers.append([xs[c_idx], ys[r_idx]])
            else:
                centers.append([w/2, h/2])
        init_centers = np.array(centers)
        
    # Flatten
    x0 = init_centers.flatten()
    
    # Bounds: centers must be inside the rectangle
    # We don't strictly bound them to [0, w] in the optimizer because the objective
    # penalizes going outside via wall distance, but keeping them roughly inside helps.
    # Actually, if a center goes outside, wall_dist becomes negative, which is bad.
    # So the optimizer will naturally push them in.
    
    try:
        res = minimize(_objective, x0, args=(w, h), method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-6})
        if res.success:
            best_centers = res.x.reshape(N, 2)
            best_r = -res.fun
            return best_centers, best_r
    except Exception:
        pass
    return None, 0.0

def construct():
    global DEADLINE
    best_r = 0.0
    best_centers = None
    best_w = 1.0
    best_h = 1.0
    
    # Search over aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # aspect = w/h. w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    # Let's iterate aspect from 0.5 to 2.0
    aspects = np.linspace(0.5, 2.0, 15)
    
    for aspect in aspects:
        if time.time() > DEADLINE - 5.0:
            break
            
        # Try a few random initializations for robustness
        # 1. Grid
        # 2. Hexagonal-ish
        # For simplicity and speed, we'll just run the grid init once per aspect,
        # and maybe a jittered one if time permits.
        
        c, r = _solve_for_aspect_ratio(aspect)
        if c is not None and r > best_r:
            best_r = r
            best_centers = c
            w = 2.0 * aspect / (1.0 + aspect)
            h = 2.0 / (1.0 + aspect)
            best_w = w
            best_h = h
            
    # If no solution found, fallback
    if best_centers is None:
        # Fallback to a simple grid in a square (aspect=1)
        w, h = 1.0, 1.0
        cols = 5
        rows = 5
        xs = np.linspace(0.1, 0.9, cols)
        ys = np.linspace(0.1, 0.9, rows)
        centers = []
        for i in range(N):
            r_idx = i // cols
            c_idx = i % cols
            centers.append([xs[c_idx], ys[r_idx]])
        best_centers = np.array(centers)
        best_r = _get_max_radius(best_centers, w, h)
        best_w = w
        best_h = h

    # Final polish: scale centers to exact bounding box?
    # The optimizer maximizes r for the fixed box w,h.
    # The result is valid: N circles of radius best_r fit in w x h.
    # Perimeter is 2(w+h) = 4.
    # Sum of radii = N * best_r.
    
    # Construct output
    out = np.zeros((N, 3))
    out[:, 0] = best_centers[:, 0]
    out[:, 1] = best_centers[:, 1]
    out[:, 2] = best_r
    
    return out

if __name__ == "__main__":
    arr = construct()
    print(arr.shape, arr[:3])
