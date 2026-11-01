import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _init_lattice():
    """Initialize points using a hexagonal lattice cut to the unit disk."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization
    positions = _init_lattice()
    
    # 2. Optimization Loop
    # We perform cyclic coordinate descent.
    # For each point, we optimize its position to maximize the minimum distance
    # to all other points and the boundary.
    
    # Pre-allocate a copy of positions to avoid modifying the original during a single point's optimization
    # Actually, we modify positions in-place, but we pass the current state to the objective function.
    
    for _ in range(1000): # Max passes, will break on time
        if time.time() > deadline - 5.0:
            break
            
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
                
            # Current position of point i
            p_i = positions[i].copy()
            
            # Objective function for point i
            def obj(x):
                # x is (2,)
                # Calculate distances to all other points
                # positions is (N, 2)
                # We need distance from x to positions[j] for all j != i
                # Vectorized:
                diffs = positions - x
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf # Exclude self
                
                min_pair = np.min(dists) / 2.0
                min_bound = 1.0 - np.linalg.norm(x)
                
                # We want to maximize min(min_pair, min_bound)
                # So we minimize -min(min_pair, min_bound)
                return -min(min_pair, min_bound)
            
            # Use Nelder-Mead to find local maximum
            # Initial guess is current position
            res = minimize(obj, p_i, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Update position
            positions[i] = res.x

    return positions
