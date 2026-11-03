"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    N = 50
    
    # Ring structure: 1 center + rings of 6, 12, 12, 19
    ring_sizes = [1, 6, 12, 12, 19]
    
    def get_centers(params):
        """Generate centers from parameters: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]"""
        centers = [[0.0, 0.0]]
        for i in range(1, len(ring_sizes)):
            r = params[i-1]
            theta_offset = params[4 + i - 1]
            n = ring_sizes[i]
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta_offset
            for a in angles:
                centers.append([r * np.cos(a), r * np.sin(a)])
        return np.array(centers)
    
    def objective(params):
        """Negative of the minimum radius (to be minimized)"""
        centers = get_centers(params)
        # Check if all centers are inside unit circle
        dists = np.hypot(centers[:, 0], centers[:, 1])
        if np.any(dists > 1.0):
            return 1e6
        # Compute minimum distance between pairs
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < min_dist:
                    min_dist = d
        # The radius is min(min_pair_dist/2, min(1 - dist_to_origin))
        r_pair = min_dist / 2
        r_boundary = 1.0 - np.max(dists)
        return -min(r_pair, r_boundary)
    
    # Initial guess based on hexagonal packing estimates
    # For 50 circles, rough estimate: r ≈ 0.125
    # Ring radii approximately: 2r, 4r, 6r, 8r but need to fit in unit circle
    # Let's start with reasonable values
    initial_params = np.array([0.25, 0.5, 0.75, 0.95, 0, np.pi/6, 0, np.pi/12])
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple initializations
    for trial in range(10):
        if time.time() > deadline - 30:
            break
            
        if trial == 0:
            params0 = initial_params.copy()
        else:
            params0 = initial_params + np.random.randn(8) * 0.05
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to hexagonal
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    centers = get_centers(best_result)
    
    # Verify validity
    dists = np.hypot(centers[:, 0], centers[:, 1])
    if np.any(dists > 1.0):
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return centers
# EVOLVE-BLOCK-END
