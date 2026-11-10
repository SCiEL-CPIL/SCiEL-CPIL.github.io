# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0 # 600 seconds
    
    def compute_ratio(points):
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        P = x.reshape(n, d)
        # Center the points
        P = P - np.mean(P, axis=0)
        
        # Normalize to unit diameter to handle scale invariance
        dists = pdist(P)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0 # Invalid
        P = P / d_max
        
        # Recompute distances after normalization
        dists_new = pdist(P)
        d_min = np.min(dists_new)
        
        # We want to maximize d_min, so minimize -d_min
        return -d_min

    def refine(points_init):
        # Flatten
        x0 = points_init.flatten()
        
        # Use Powell's method which is derivative-free and robust for non-smooth objectives
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'ftol': 1e-10})
            return res.x.reshape(n, d)
        except Exception:
            return points_init

    best_points = None
    best_score = -1.0

    # 1. Hexagonal Lattice Initialization
    # Generate hexagonal grid points
    hex_points = []
    # Basis vectors for hexagonal lattice
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2.0])
    
    # Generate points in a grid large enough to contain 16 points
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            hex_points.append(p)
            
    hex_points = np.array(hex_points)
    # Distance to origin
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    # Sort by distance to origin
    sorted_indices = np.argsort(dists_to_origin)
    # Take the first 16 closest points
    hex_init = hex_points[sorted_indices[:n]]
    
    # Random Initialization
    np.random.seed(42)
    random_init = np.random.randn(n, d)
    
    # Ring Initialization
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring_init = np.column_stack([np.cos(angles), np.sin(angles)])
    
    initializations = [hex_init, random_init, ring_init]
    
    # Add a few perturbed hexagonal inits
    for i in range(3):
        perturbed = hex_init + np.random.randn(n, d) * 0.1
        initializations.append(perturbed)

    for init in initializations:
        if time.time() > deadline - 60: # Safety margin
            break
            
        refined = refine(init)
        score = compute_ratio(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined.copy()

    # If we have time, do a few more random restarts or finer tuning
    # But usually the hex init + Powell is very strong.
    # Let's do a final polish on the best result if time permits
    if best_points is not None and time.time() < deadline - 30:
        # Try a second pass of refinement on the best point
        # Maybe with a tighter tolerance
        def objective_tight(x):
            P = x.reshape(n, d)
            P = P - np.mean(P, axis=0)
            dists = pdist(P)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 0.0
            P = P / d_max
            dists_new = pdist(P)
            d_min = np.min(dists_new)
            return -d_min

        try:
            res = minimize(objective_tight, best_points.flatten(), method='Powell', options={'maxiter': 100, 'ftol': 1e-12})
            new_points = res.x.reshape(n, d)
            new_score = compute_ratio(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
