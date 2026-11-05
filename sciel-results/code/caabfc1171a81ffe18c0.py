import time
import random
import itertools
import numpy as np
from collections import Counter
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def _calculate_max_multiplicity(points):
        n = len(points)
        if n < 3:
            return 0, 0
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, max_area

    def _run_milp_search(grid_points, target_k, time_limit, n_points=30):
        """
        Uses MILP to find n_points from grid_points maximizing the number of triangles with area K.
        Variables:
        x_i: 1 if point i is selected
        y_{i,j,k}: 1 if i<j<k are selected and form triangle of area K
        """
        n_grid = len(grid_points)
        if n_grid < n_points:
            return None, 0

        # Precompute all triples (i, j, k) with i<j<k
        triples = []
        for i in range(n_grid):
            for j in range(i + 1, n_grid):
                for k in range(j + 1, n_grid):
                    x1, y1 = grid_points[i]
                    x2, y2 = grid_points[j]
                    x3, y3 = grid_points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det == target_k:
                        triples.append((i, j, k))
        
        if not triples:
            return None, 0

        # MILP formulation
        # Variables:
        # 0 to n_grid-1: x_i (binary)
        # n_grid to n_grid + len(triples) - 1: y_t (binary)
        
        n_vars = n_grid + len(triples)
        
        # Objective: Maximize sum(y_t)
        # milp minimizes, so we minimize -sum(y_t)
        c = np.zeros(n_vars)
        c[n_grid:] = -1.0
        
        # Constraints:
        # 1. Sum(x_i) == n_points
        # 2. y_t <= x_i, y_t <= x_j, y_t <= x_k for each triple t=(i,j,k)
        #    Equivalently: y_t - x_i <= 0, etc.
        
        # Build constraint matrix A
        # Row 0: Sum(x_i) == n_points
        # Rows 1..: y_t - x_i <= 0
        
        n_constraints = 1 + 3 * len(triples)
        A = np.zeros((n_constraints, n_vars))
        
        # Constraint 0: Sum(x_i) == n_points
        A[0, :n_grid] = 1.0
        
        lb = np.zeros(n_constraints)
        ub = np.zeros(n_constraints)
        lb[0] = n_points
        ub[0] = n_points
        
        # Constraints for each triple
        for idx, (i, j, k) in enumerate(triples):
            row = 1 + idx
            # y_{i,j,k} - x_i <= 0
            A[row, n_grid + idx] = 1.0
            A[row, i] = -1.0
            lb[row] = 0
            ub[row] = 0
            
            row = 1 + idx + len(triples)
            # y_{i,j,k} - x_j <= 0
            A[row, n_grid + idx] = 1.0
            A[row, j] = -1.0
            lb[row] = 0
            ub[row] = 0
            
            row = 1 + idx + 2 * len(triples)
            # y_{i,j,k} - x_k <= 0
            A[row, n_grid + idx] = 1.0
            A[row, k] = -1.0
            lb[row] = 0
            ub[row] = 0

        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_vars)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={"time_limit": time_limit, "presolve": True})
            
            if res.status == 0 or res.status == 2: # Optimal or Time Limit (with feasible solution)
                if res.x is not None:
                    selected_indices = np.where(res.x[:n_grid] > 0.5)[0]
                    if len(selected_indices) == n_points:
                        points = [grid_points[i] for i in selected_indices]
                        # Verify distinctness and bounds
                        if len(set(points)) == n_points and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in points):
                            score, _ = _calculate_max_multiplicity(points)
                            return points, score
        except Exception:
            pass
            
        return None, 0

    # Generate candidate grids
    # We want grids that are "dense" enough to have many triangles of small area K
    # But not so large that the MILP explodes.
    # Grid size approx 10x10 to 15x15 is reasonable for 30 points.
    
    grids = []
    
    # Standard grids centered at 0
    for size in [10, 11, 12, 13, 14, 15]:
        half = size // 2
        pts = []
        for x in range(-half, size - half):
            for y in range(-half, size - half):
                pts.append((x, y))
        # Randomly shuffle to help MILP? No, fixed order is fine.
        grids.append(pts)
        
    # Sheared grids (parallelograms)
    for size in [10, 11, 12]:
        for shear in [-2, -1, 0, 1, 2]:
            pts = []
            for x in range(-size//2, size//2 + 1):
                for y in range(-size//2, size//2 + 1):
                    px = x + shear * y
                    py = y
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
            if len(pts) >= 30:
                grids.append(pts)

    best_points = None
    best_score = 0
    
    # Try a few promising area values K
    # Small areas like 1, 2, 3, 4, 5 often have high multiplicity in dense grids
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10]
    
    # We will run MILP on the largest grids first with the most common areas
    # To manage time, we limit the number of MILP runs
    
    milp_runs = 0
    max_milp_runs = 15 # Rough estimate, each run might take 1-5s
    
    for grid in sorted(grids, key=len, reverse=True)[:4]: # Top 4 largest grids
        if time.time() > deadline - 5:
            break
            
        for K in target_areas:
            if time.time() > deadline - 2:
                break
            if milp_runs >= max_milp_runs:
                break
                
            remaining_time = deadline - time.time() - 2
            if remaining_time < 1:
                break
                
            # Use a fraction of remaining time for this run
            # If we have 50s left, give 5s to first run, 4s to next, etc?
            # Better: fixed small time limit per run to explore multiple Ks
            time_limit = min(remaining_time, 5.0)
            
            points, score = _run_milp_search(grid, K, time_limit)
            milp_runs += 1
            
            if points is not None:
                if score > best_score:
                    best_score = score
                    best_points = points

    # Fallback if MILP didn't find anything good or timed out early
    if best_points is None or best_score < 100:
        # Simple grid construction
        # 6x5 grid
        pts = []
        for x in range(-3, 3):
            for y in range(-2, 3):
                pts.append((x, y))
        if len(pts) == 30:
            best_points = pts
            best_score, _ = _calculate_max_multiplicity(best_points)
        else:
            # Random fallback
            random.seed(42)
            pts = set()
            while len(pts) < 30:
                pts.add((random.randint(-15, 15), random.randint(-15, 15)))
            best_points = list(pts)
            best_score, _ = _calculate_max_multiplicity(best_points)

    return best_points
