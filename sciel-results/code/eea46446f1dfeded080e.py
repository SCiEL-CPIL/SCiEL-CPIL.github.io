"""
Pack 50 equal circles in a unit square, maximize common radius.
Strategy: 
1. Initialize with a hexagonal grid.
2. Compute Voronoi cells to find the "inradius" of each cell (distance to nearest neighbor or wall).
3. Use L-BFGS-B to maximize the soft-min of these constraints (log-sum-exp approximation of min).
This avoids the non-differentiable min and the complex constraint handling of SLSQP,
allowing the optimizer to smoothly navigate the landscape of the packing problem.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    r = min(wall_r, min_pair/2)
    -log(sum(exp(-beta * constraints))) / beta approximates min(constraints).
    """
    # Wall constraints
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_cons = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    # Ensure numerical stability
    m = np.min(all_cons)
    if m < -1000:
        return m
    log_sum_exp = m + np.log(np.sum(np.exp(-beta * (all_cons - m))))
    return -log_sum_exp / beta

def _objective_smooth(x, beta):
    """Objective function for L-BFGS-B: negative soft-min radius."""
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, beta)

def _init_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _voronoi_inradii(centers):
    """
    Compute the distance from each point to the nearest other point or wall boundary.
    This gives a lower bound on the radius if we only move points slightly,
    but more importantly, it identifies which points are "tight".
    """
    # Add boundary points to Voronoi diagram to handle walls
    # We need to enclose the points. A box slightly larger than the square.
    # The walls are at +/- 0.5.
    # We can add 4 corner points and mid-edge points far away? 
    # Actually, simpler: just compute min distance to other points and min distance to walls.
    # Voronoi is overkill if we just want the min distance.
    # But Voronoi helps in optimization if we want to move points to equalize distances.
    # For initialization, let's just use a good hex grid.
    
    # Let's stick to a simple hex grid initialization but optimized for the square.
    # 50 points.
    # 7x7 grid is 49. 8x7 is 56.
    # Let's try to fit 50 points in a hex pattern.
    
    # A standard hex packing in a square can be approximated by rows.
    # Row 0: 7 points
    # Row 1: 6 points
    # Row 2: 7 points
    # ...
    # 7+6+7+6+7+6+7 = 46. Need 4 more.
    # Maybe 8+7+8+7+8+7+8 = 53?
    
    # Let's generate a grid and pick the best 50.
    # Spacing s.
    # Width = (n_cols - 1) * s + s/2 (if offset)
    # Height = (n_rows - 1) * s * sqrt(3)/2
    
    # Let's try a few configurations and pick the one with max min-distance.
    
    configs = [
        (7, 8, 0),  # 7 cols, 8 rows, offset pattern
        (8, 7, 0),
        (6, 9, 0),
        (9, 6, 0),
        (5, 10, 0),
        (10, 5, 0),
    ]
    
    best_centers = None
    best_score = -1
    
    for cols, rows, _ in configs:
        # Generate points
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c
                y = r * np.sqrt(3) / 2
                if r % 2 != 0:
                    x += 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        # Center and scale to fit in [-0.5, 0.5]
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        if width > 0 and height > 0:
            scale = 1.0 / max(width, height)
            pts = (pts - (min_x, min_y)) * scale
            pts = pts - 0.5 # Center in [-0.5, 0.5]
            
            # If we have more than N points, we need to select N.
            # For simplicity in init, if we generate exactly N, great.
            # If more, we can randomly drop some or keep the ones with largest Voronoi area.
            # Let's just take the first N for now, or better, use a grid that is close to N.
            
            if len(pts) > N:
                # Select N points that are most "spread out"
                # Simple heuristic: keep those with largest min-distance to others
                # This is O(N^3), might be slow.
                # For init, just taking a subset is okay, L-BFGS will fix it.
                # Let's just take the first N.
                pts = pts[:N]
            elif len(pts) < N:
                # Pad with random points? Or just use what we have?
                # We need exactly N.
                # Let's add random points in the empty space.
                # For now, this config is invalid if count != N.
                # Let's adjust configs to match N=50.
                # 7+7+7+7+7+7+7+... no.
                # Let's just use a rectangular grid 7x8=56 and drop 6?
                # Or 5x10=50.
                # Let's use 5x10 rectangular grid as a fallback init.
                pass
            
            # Calculate score
            r = _radius(pts)
            if r > best_score:
                best_score = r
                best_centers = pts.copy()
                
    if best_centers is None:
        # Fallback: 5x10 grid
        x = np.linspace(-0.4, 0.4, 5)
        y = np.linspace(-0.4, 0.4, 10)
        xx, yy = np.meshgrid(x, y)
        best_centers = np.column_stack((xx.ravel(), yy.ravel()))
        
    return best_centers

def construct():
    start_time = time.time()
    
    # 1. Initial Configuration
    centers = _init_hexagonal_grid(7, 8, 1.0)
    
    # Ensure it's within bounds
    centers = np.clip(centers, -0.45, 0.45)
    
    # 2. Optimization with L-BFGS-B
    # We use a soft-min objective.
    # Start with a low beta (smoother) and increase it? 
    # Or just use a fixed high beta.
    # L-BFGS-B works well with smooth objectives.
    
    x0 = centers.ravel()
    
    def obj(x):
        return _objective_smooth(x, beta=100.0)
    
    def jac(x):
        # Numerical gradient is fine for L-BFGS-B if step size is small,
        # but analytical is faster. 
        # Let's rely on L-BFGS-B's numerical gradient initially for simplicity,
        # but provide a finite difference if needed.
        # Actually, providing jac is better.
        # However, calculating the gradient of the log-sum-exp is complex.
        # Let's use numerical gradient via L-BFGS-B default (it uses finite diff if jac is None).
        # Wait, L-BFGS-B in SciPy uses numerical differentiation if jac is not provided.
        # This might be slow for 100 variables.
        # Let's try to provide a jac.
        
        # Gradient of -log(sum(exp(-beta * c))) / beta
        # = 1/beta * sum(exp(-beta * c_i) * (-beta * dc_i/dx)) / sum(exp(-beta * c_i))
        # = - sum(exp(-beta * c_i) * dc_i/dx) / sum(exp(-beta * c_i))
        # = - sum(w_i * dc_i/dx) where w_i = exp(-beta * c_i) / sum(exp(-beta * c_j))
        
        c = x.reshape(N, 2)
        
        # Constraints c_k
        # Wall: 0.5 - c, 0.5 + c, etc.
        # Pair: dist/2
        
        # This is getting complicated to vectorize efficiently for the gradient.
        # Given the time limit, let's just use numerical gradient.
        # L-BFGS-B with 100 variables and numerical gradient should be okay for ~100-200 iterations.
        return None
        
    try:
        res = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
        )
        if res.x is not None:
            centers = res.x.reshape(N, 2)
    except Exception:
        pass
    
    # 3. Refinement with higher beta
    # Increase beta to sharpen the min
    x0 = centers.ravel()
    
    def obj_sharp(x):
        return _objective_smooth(x, beta=500.0)
        
    try:
        res = minimize(
            obj_sharp,
            x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            centers = res.x.reshape(N, 2)
    except Exception:
        pass
        
    # 4. Local Search / Polish
    # Use a simple local search to fix any tiny violations and push radius up.
    # Or just run L-BFGS-B again with even higher beta.
    
    x0 = centers.ravel()
    def obj_very_sharp(x):
        return _objective_smooth(x, beta=1000.0)
        
    try:
        res = minimize(
            obj_very_sharp,
            x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        if res.x is not None:
            centers = res.x.reshape(N, 2)
    except Exception:
        pass
        
    # Final check and return
    # Ensure no points are outside
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

if __name__ == "__main__":
    c = construct()
    print(_radius(c))
