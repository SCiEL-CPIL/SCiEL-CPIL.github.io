"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles using vectorization."""
    n = len(points)
    if n < 3:
        return 0.0
        
    # Vectorized area calculation
    # points shape (N, 2)
    p1 = points[:, np.newaxis, :] # (N, 1, 2)
    p2 = points[np.newaxis, :, :] # (1, N, 2)
    diff = p2 - p1 # (N, N, 2) -> diff[i, j] = points[j] - points[i]
    
    d1 = diff[:, :, np.newaxis, :] # (N, N, 1, 2)
    d2 = diff[:, np.newaxis, :, :] # (N, 1, N, 2)
    
    # Cross product z-component: (x1*y2 - y1*x2)
    # d1[..., 0] is x of (j-i), d1[..., 1] is y of (j-i)
    # d2[..., 0] is x of (k-i), d2[..., 1] is y of (k-i)
    cross_z = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0] # (N, N, N)
    
    # Mask out cases where i==j, i==k, or j==k
    idx = np.arange(n)
    mask = np.ones((n, n, n), dtype=bool)
    mask[idx, idx, :] = False
    mask[idx, :, idx] = False
    mask[:, idx, idx] = False
    
    areas = 0.5 * np.abs(cross_z)
    areas[~mask] = np.inf
    
    return np.min(areas)

def _penalty(points):
    """Calculate penalty for points outside the triangle."""
    h = np.sqrt(3) / 2
    x = points[:, 0]
    y = points[:, 1]
    
    # Constraints:
    # 1. y >= 0
    # 2. y <= 2*h*x  (line from (0,0) to (0.5, h) is y = 2*sqrt(3)*x? No.
    #    Slope of (0,0) to (0.5, h): h / 0.5 = 2h = sqrt(3).
    #    So y = sqrt(3) x.
    #    Wait, h = sqrt(3)/2.
    #    Slope = (sqrt(3)/2) / 0.5 = sqrt(3).
    #    So y <= sqrt(3) x.
    # 3. y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    
    # Let's re-verify the lines.
    # Vertex A(0,0), B(1,0), C(0.5, h).
    # Edge AB: y=0.
    # Edge AC: Passes through (0,0) and (0.5, h). Slope = h/0.5 = 2h = sqrt(3). Equation: y = sqrt(3)x.
    # Edge BC: Passes through (1,0) and (0.5, h). Slope = (h-0)/(0.5-1) = h/-0.5 = -2h = -sqrt(3).
    # Equation: y - 0 = -sqrt(3)(x-1) => y = sqrt(3)(1-x).
    
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - np.sqrt(3)*x)
    v3 = np.maximum(0, y - np.sqrt(3)*(1-x))
    
    return np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)

def construct():
    deadline = time.time() + 140
    h = np.sqrt(3) / 2
    
    def objective_flat(flat_points):
        points = flat_points.reshape(N, 2)
        min_area = _min_triangle_area(points)
        pen = _penalty(points)
        return -min_area + pen * 0.1 # Penalty weight
        
    # Initial bounds
    # x in [0, 1], y in [0, h]
    bounds = [(0, 1), (0, h)] * N
    
    # Use Differential Evolution for global search
    # popsize=15, maxiter=100 is a good start.
    # tol=0.001
    
    best_result = None
    best_score = -np.inf
    
    # Run DE
    try:
        de_result = differential_evolution(
            objective_flat, 
            bounds=bounds, 
            maxiter=150, 
            popsize=15, 
            tol=0.001,
            seed=42,
            polish=False # We will polish manually
        )
        if time.time() < deadline - 10:
            best_flat = de_result.x
            best_score = -de_result.fun
            # Check if valid (penalty should be 0 or small)
            pts = best_flat.reshape(N, 2)
            if _penalty(pts) < 1e-6:
                best_result = pts
            else:
                # If DE didn't find a valid point, try to project or use a fallback
                # For now, just keep it if it's the best we have, but maybe reject if penalty is high
                pass
    except Exception as e:
        pass

    # If DE failed or didn't find a good valid point, try Nelder-Mead from a good heuristic start
    if best_result is None or best_score < 0.001:
        # Heuristic start: Grid
        # Create a grid of points inside the triangle
        # 3x3 grid roughly
        xs = [0.2, 0.5, 0.8]
        ys = [0.1, 0.3, 0.5]
        points = []
        for x in xs:
            for y in ys:
                # Check if inside
                if y < np.sqrt(3)*x and y < np.sqrt(3)*(1-x) and y > 0:
                    points.append([x, y])
        # Add center if not present
        if len(points) < N:
             points.append([0.5, h/3])
        # Add random points if needed
        while len(points) < N:
             points.append([np.random.rand(), np.random.rand()*h*0.5])
        start_points = np.array(points[:N])
        
        # Polish with Nelder-Mead
        try:
            nm_result = minimize(
                objective_flat, 
                start_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() < deadline - 5:
                nm_score = -nm_result.fun
                if nm_score > best_score:
                    best_score = nm_score
                    pts = nm_result.x.reshape(N, 2)
                    if _penalty(pts) < 1e-6:
                        best_result = pts
        except Exception:
            pass

    # If we still don't have a result, use the heuristic start
    if best_result is None:
        # Fallback to a simple grid
        best_result = np.array([[0.5, h/3]] * N) # Degenerate, but valid
        # Better fallback:
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.5, 3)
        pts = []
        for x in xs:
            for y in ys:
                if y < np.sqrt(3)*x and y < np.sqrt(3)*(1-x):
                    pts.append([x, y])
        while len(pts) < N:
            pts.append([0.5, h/3])
        best_result = np.array(pts[:N])

    # Final Polish with Nelder-Mead on the best result found
    if time.time() < deadline - 5:
        try:
            nm_result = minimize(
                objective_flat, 
                best_result.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                pts = nm_result.x.reshape(N, 2)
                if _penalty(pts) < 1e-6:
                    best_result = pts
                    best_score = nm_score
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
