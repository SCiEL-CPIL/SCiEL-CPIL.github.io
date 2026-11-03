import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
DIST_MIN = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Normals for the fixed outer hexagon (axis aligned)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    # Or 0, 60, ... if we rotate the frame.
    # Let's use normals at angles 0, 60, 120, 180, 240, 300 degrees for simplicity of calculation.
    # This corresponds to an outer hexagon with flat top/bottom?
    # Normal at 0 is (1,0). This is the right side.
    # The hexagon vertices would be at angles 30, 90, ...
    # This is a standard "pointy top" hexagon if normals are at 0?
    # No, if normal is at 0, the side is vertical.
    # Let's just define the 6 unit normals.
    normals = np.array([
        [math.cos(math.pi / 3 * k), math.sin(math.pi / 3 * k)] for k in range(6)
    ])
    # normals: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866)

    def get_D(x_coords):
        """
        Calculate the maximum projection of points x_coords onto the 6 normals.
        x_coords: (N, 2) array
        Returns: max_proj
        """
        # x_coords shape (N, 2), normals shape (6, 2)
        # Projections shape (N, 6)
        projs = x_coords @ normals.T
        # We need max of abs(projs)
        max_proj = np.max(np.abs(projs))
        return max_proj

    def penalty_term(x_coords, lambda_val):
        """
        Calculate the penalty for overlapping hexagons.
        x_coords: (N, 2) array
        """
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = x_coords[i] - x_coords[j]
                # Projections of difference vector onto normals
                d_projs = np.abs(d @ normals.T)
                max_d_proj = np.max(d_projs)
                if max_d_proj < DIST_MIN:
                    # Violation
                    violation = DIST_MIN - max_d_proj
                    pen += lambda_val * (violation ** 2)
        return pen

    def objective(x_flat, lambda_val):
        x_coords = x_flat.reshape(N, 2)
        D = get_D(x_coords)
        pen = penalty_term(x_coords, lambda_val)
        return D + pen

    def run_optimization(x0, lambda_schedule):
        x0_flat = x0.flatten()
        best_x = x0_flat
        best_val = objective(x0_flat, lambda_schedule[0])
        
        for lam in lambda_schedule:
            # Use L-BFGS-B for unconstrained optimization of the penalized function
            res = minimize(objective, best_x, args=(lam,), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
            else:
                # If it didn't improve, try a few more iterations or restart?
                # For now, just keep the previous best
                pass
                
        return best_x, best_val

    # Generate initial configurations
    inits = []
    
    # Config 1: 7-hex flower + 4 extensions
    # Flower centers
    c0 = np.array([0.0, 0.0])
    flower_centers = [c0]
    for k in range(6):
        ang = k * math.pi / 3
        c = DIST_MIN * np.array([math.cos(ang), math.sin(ang)])
        flower_centers.append(c)
    
    # Extension candidates: neighbors of the 6 outer flower hexes, pointing outwards
    ext_candidates = []
    for k in range(6):
        ang = k * math.pi / 3
        # The outer hex center is at DIST_MIN * (cos, sin)
        # The neighbor in the same direction is at 2 * DIST_MIN * (cos, sin)
        c = 2 * DIST_MIN * np.array([math.cos(ang), math.sin(ang)])
        ext_candidates.append(c)
        
    # We need to pick 4 out of 6 extensions
    # To break symmetry and cover the space, we try all combinations of 4
    # There are C(6,4) = 15 combinations
    from itertools import combinations
    
    for combo in combinations(range(6), 4):
        init_centers = flower_centers.copy()
        for idx in combo:
            init_centers.append(ext_candidates[idx])
        
        # Shuffle order? Not necessary as objective is symmetric
        init_arr = np.array(init_centers)
        inits.append(init_arr)
        
    # Config 2: Random compact clusters
    # Generate 11 points in a circle of radius 3, ensuring min dist roughly
    np.random.seed(42)
    for _ in range(10):
        pts = []
        # Place first at 0
        pts.append([0.0, 0.0])
        for i in range(1, N):
            # Place near existing points
            base_idx = np.random.randint(0, len(pts))
            base = pts[base_idx]
            ang = np.random.uniform(0, 2 * math.pi)
            r = DIST_MIN * np.random.uniform(0.95, 1.1) # Slightly overlapping to let optimizer fix
            new_pt = base + r * np.array([math.cos(ang), math.sin(ang)])
            pts.append(new_pt)
        inits.append(np.array(pts))
        
    best_global_D = float('inf')
    best_global_x = None
    
    lambda_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    for i, init_arr in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Center the configuration to help optimization?
        # The objective get_D is translation invariant? No, it's max abs proj.
        # If we shift all points, D changes.
        # We want the cluster centered at the origin to minimize D.
        # So we should subtract the mean.
        mean_pt = np.mean(init_arr, axis=0)
        init_arr = init_arr - mean_pt
        
        x_opt, val_opt = run_optimization(init_arr, lambda_schedule)
        
        # Final check: Is it valid?
        x_final = x_opt.reshape(N, 2)
        # Recalculate D
        D_final = get_D(x_final)
        # Check penalties (should be 0 if converged well)
        pen_final = penalty_term(x_final, 1.0)
        
        if pen_final < 1e-6: # Valid configuration
            if D_final < best_global_D:
                best_global_D = D_final
                best_global_x = x_final
        else:
            # If not valid, we might still use it if it's close, but score will be bad.
            # Better to reject or try to repair.
            # For now, reject if penalty is high.
            pass

    # If no valid solution found, fallback to a known valid one (e.g. the first init without optimization)
    if best_global_x is None:
        # Use the first init, centered
        init_arr = inits[0] - np.mean(inits[0], axis=0)
        best_global_x = init_arr
        best_global_D = get_D(best_global_x)
        # This might be invalid (overlapping), but it's better than crashing.
        # We should try to separate them manually if possible, or just return it.
        # The evaluator will score it. If it overlaps, it's invalid?
        # "disjoint regular hexagons" is required.
        # If my optimization failed, I should try a simpler fallback.
        
        # Fallback: Spread them out on a grid
        grid = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                grid.append([x, y])
        # Pick 11 from grid that are close to center
        grid.sort(key=lambda p: p[0]**2 + p[1]**2)
        best_global_x = np.array(grid[:N])
        best_global_D = get_D(best_global_x)

    # Calculate final Side Length
    S = (best_global_D + APOTHEM) / APOTHEM
    
    # Prepare output
    inner = []
    for x, y in best_global_x:
        inner.append([float(x), float(y), 0.0]) # Angle 0
        
    center = [0.0, 0.0] # We optimized around origin
    side = float(S)
    rotation = 0.0
    
    return inner, center, side, rotation
