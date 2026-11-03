import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _pack_for_aspect(w, h, n_circles=N, max_iter=2000, tol=1e-10):
    """Pack circles in w x h rectangle, return (positions, radii) maximizing sum of radii."""
    # Initialize in a grid
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    
    positions = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * w / cols
        positions[i, 1] = (row + 0.5) * h / rows
    
    # Iterative repulsion
    for iteration in range(max_iter):
        # Compute radii based on clearance
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            # Distance to walls
            min_dist = min(
                positions[i, 0],
                w - positions[i, 0],
                positions[i, 1],
                h - positions[i, 1]
            )
            # Distance to other circles
            for j in range(n_circles):
                if i != j:
                    d = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                               (positions[i, 1] - positions[j, 1])**2)
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(0, min_dist - 1e-9)
        
    # Use scipy optimization instead
    def objective(x):
        pos = x[:n_circles * 2].reshape(n_circles, 2)
        # Check bounds
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > w) or \
           np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > h):
            return 0.0
        radii = np.zeros(n_circles)
        for i in range(n_circles):
            min_dist = min(
                pos[i, 0],
                w - pos[i, 0],
                pos[i, 1],
                h - pos[i, 1]
            )
            for j in range(n_circles):
                if i != j:
                    d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                               (pos[i, 1] - pos[j, 1])**2)
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(0, min_dist)
        return -np.sum(radii)
    
    x0 = positions.flatten()
    bounds = [(0, w)] * n_circles + [(0, h)] * n_circles
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    pos = result.x[:n_circles * 2].reshape(n_circles, 2)
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        min_dist = min(
            pos[i, 0],
            w - pos[i, 0],
            pos[i, 1],
            h - pos[i, 1]
        )
        for j in range(n_circles):
            if i != j:
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + 
                           (pos[i, 1] - pos[j, 1])**2)
                min_dist = min(min_dist, d / 2.0)
        radii[i] = max(0, min_dist - 1e-9)
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_score = 0
    best_result = None
    
    # Try various aspect ratios
    aspect_ratios = np.linspace(0.3, 3.0, 20)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii = _pack_for_aspect(w, h)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
    
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
