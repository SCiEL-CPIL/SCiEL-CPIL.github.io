"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _validity_check(arr):
    """Check if the arrangement is valid: circles inside [0,1]^2 with disjoint interiors."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
        return False
    if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9) or np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
        return False
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _sum_radii(arr):
    return float(np.sum(arr[:, 2]))

def _phyllotaxis(n, scale=0.95):
    """Generate phyllotaxis points in a unit square."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = scale * np.sqrt(i / n)
        theta = i * golden_angle
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        points.append([x, y])
    return np.array(points)

def _initial_grid(n):
    """Generate a grid-based initial arrangement."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])

def _coordinate_descent(points, radii, max_iter=5000, tol=1e-10):
    """
    Coordinate descent to maximize sum of radii with non-overlap constraints.
    points: (n, 2) array of centers
    radii: (n,) array of radii
    Returns: (points, radii)
    """
    n = len(points)
    points = points.copy()
    radii = radii.copy()
    
    # Compute initial valid radii based on nearest neighbors
    for i in range(n):
        min_dist = min(1.0 - points[i][0], points[i][0], 1.0 - points[i][1], points[i][1])
        for j in range(n):
            if i != j:
                d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                min_dist = min(min_dist, d / 2)
        radii[i] = max(0.0, min_dist - 1e-9)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Try to increase radius[i]
            # Find the limiting constraint
            min_dist_to_boundary = min(
                points[i][0],
                1.0 - points[i][0],
                points[i][1],
                1.0 - points[i][1]
            )
            max_r = min_dist_to_boundary
            for j in range(n):
                if i != j:
                    d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    max_r = min(max_r, d - radii[j])
            max_r = max(0.0, max_r)
            if max_r > radii[i] + tol:
                radii[i] = max_r
                improved = True
            elif max_r < radii[i] - tol:
                radii[i] = max(0.0, max_r)
                improved = True
        
        # Try to move points to allow larger radii
        for i in range(n):
            # Gradient of sum of radii w.r.t. position
            # For each j, the constraint is dist(i,j) >= r[i] + r[j]
            # We want to maximize r[i], which is limited by min over j of (dist(i,j) - r[j])
            # and boundary constraints
            
            # Compute the limiting neighbor
            limit_val = min(
                points[i][0],
                1.0 - points[i][0],
                points[i][1],
                1.0 - points[i][1]
            )
            limit_dir = np.array([0.0, 0.0])
            if points[i][0] <= limit_val:
                limit_val = points[i][0]
                limit_dir = np.array([1.0, 0.0])
            if 1.0 - points[i][0] < limit_val:
                limit_val = 1.0 - points[i][0]
                limit_dir = np.array([-1.0, 0.0])
            if points[i][1] < limit_val:
                limit_val = points[i][1]
                limit_dir = np.array([0.0, 1.0])
            if 1.0 - points[i][1] < limit_val:
                limit_val = 1.0 - points[i][1]
                limit_dir = np.array([0.0, -1.0])
            
            for j in range(n):
                if i != j:
                    d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    val = d - radii[j]
                    if val < limit_val:
                        limit_val = val
                        if d > 1e-12:
                            limit_dir = (points[i] - points[j]) / d
                        else:
                            limit_dir = np.array([1.0, 0.0])
            
            # Move point i in the direction that increases the limiting constraint
            step = 0.01
            new_pos = points[i] + step * limit_dir
            # Check if this move is valid
            valid = True
            if new_pos[0] < 0 or new_pos[0] > 1 or new_pos[1] < 0 or new_pos[1] > 1:
                valid = False
            if valid:
                for j in range(n):
                    if i != j:
                        d = np.sqrt((new_pos[0] - points[j][0])**2 + (new_pos[1] - points[j][1])**2)
                        if d < radii[i] + radii[j] - 1e-9:
                            valid = False
                            break
            if valid:
                points[i] = new_pos
                improved = True
        
        if not improved:
            break
    
    return points, radii

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with phyllotaxis initial positions
    best_arr = None
    best_score = 0.0
    
    # Generate initial points using phyllotaxis
    points = _phyllotaxis(N)
    radii = np.zeros(N)
    
    # Initial radii: small value
    for i in range(N):
        min_dist = min(points[i][0], 1.0 - points[i][0], points[i][1], 1.0 - points[i][1])
        for j in range(N):
            if i != j:
                d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                min_dist = min(min_dist, d / 2)
        radii[i] = max(0.0, min_dist - 1e-6)
    
    arr = np.column_stack([points, radii])
    if _validity_check(arr):
        best_arr = arr.copy()
        best_score = _sum_radii(arr)
    
    # Run coordinate descent
    if time.time() < deadline:
        points, radii = _coordinate_descent(points, radii, max_iter=2000, tol=1e-10)
        arr = np.column_stack([points, radii])
        if _validity_check(arr):
            score = _sum_radii(arr)
            if score > best_score:
                best_arr = arr.copy()
                best_score = score
    
    # Try a few more iterations of refinement
    for restart in range(5):
        if time.time() >= deadline:
            break
        
        # Perturb the best solution slightly
        if best_arr is not None:
            points = best_arr[:, :2].copy()
            radii = best_arr[:, 2].copy()
            # Add small noise
            noise_scale = 0.01 / (restart + 1)
            points += noise_scale * np.random.randn(N, 2)
            points = np.clip(points, 0, 1)
            
            # Re-validate and adjust radii
            for i in range(N):
                min_dist = min(points[i][0], 1.0 - points[i][0], points[i][1], 1.0 - points[i][1])
                for j in range(N):
                    if i != j:
                        d = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                        min_dist = min(min_dist, d - radii[j])
                radii[i] = max(0.0, min_dist)
            
            arr = np.column_stack([points, radii])
            if _validity_check(arr):
                score = _sum_radii(arr)
                if score > best_score:
                    best_score = score
                    best_arr = arr.copy()
            
            # Coordinate descent from perturbed start
            if time.time() < deadline:
                points, radii = _coordinate_descent(points, radii, max_iter=1000, tol=1e-10)
                arr = np.column_stack([points, radii])
                if _validity_check(arr):
                    score = _sum_radii(arr)
                    if score > best_score:
                        best_arr = arr.copy()
                        best_score = score
    
    # Final high-precision polish
    if best_arr is not None and time.time() < deadline:
        points = best_arr[:, :2].copy()
        radii = best_arr[:, 2].copy()
        points, radii = _coordinate_descent(points, radii, max_iter=5000, tol=1e-12)
        arr = np.column_stack([points, radii])
        if _validity_check(arr):
            score = _sum_radii(arr)
            if score > best_score:
                best_arr = arr.copy()
                best_score = score
    
    if best_arr is None:
        # Fallback to grid
        return _initial_grid(N)
    
    return best_arr
# EVOLVE-BLOCK-END
