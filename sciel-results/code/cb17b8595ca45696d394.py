# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin of 20s
    
    n = 16
    d = 2
    
    def compute_ratio(points):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_ratio(points):
        return -compute_ratio(points)

    best_score = -np.inf
    best_points = None

    # Generate diverse initializations
    initializers = []
    
    # 1. Random Normal
    np.random.seed(42)
    for _ in range(50):
        initializers.append(np.random.randn(n, d))
        
    # 2. Random Uniform
    for _ in range(50):
        initializers.append(np.random.uniform(-2, 2, (n, d)))
        
    # 3. Grid 4x4
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    initializers.append(grid)
    
    # 4. Perturbed Grid
    for _ in range(20):
        perturbed = grid + np.random.randn(n, d) * 0.1
        initializers.append(perturbed)
        
    # 5. Symmetric Rings
    # 2 rings of 8
    for r1 in [0.5, 1.0, 1.5]:
        for r2 in [0.5, 1.0, 1.5]:
            if abs(r1 - r2) < 0.1: continue
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            ring1 = np.c_[r1*np.cos(angles1), r1*np.sin(angles1)]
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            ring2 = np.c_[r2*np.cos(angles2), r2*np.sin(angles2)]
            initializers.append(np.vstack([ring1, ring2]))
            
    # 6. Lattice (Hexagonal)
    # Generate a hexagonal lattice patch
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3)/2
            pts.append([x, y])
    # Take 16 points from this lattice, centered
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    # Sort by distance from center to get a compact set? Or just take first 16
    # Let's take the 16 closest to the origin
    dists_to_origin = np.sum(pts**2, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    initializers.append(pts[idx])
    
    # 7. Random rotations of the hex lattice
    for _ in range(10):
        theta = np.random.uniform(0, 2*np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        initializers.append(pts[idx] @ rot.T)

    # Optimization loop
    for init in initializers:
        if time.time() > deadline:
            break
            
        try:
            # Use Powell's method which is robust for non-smooth functions
            res = minimize(neg_ratio, init, method='Powell', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x
        except Exception:
            continue

    if best_points is None:
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
