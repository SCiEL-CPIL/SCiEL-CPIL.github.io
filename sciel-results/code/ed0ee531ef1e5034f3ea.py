# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

TOL = 1e-7
DEADLINE_MARGIN = 30.0

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _objective(x, n=10):
    """
    Objective function for optimization.
    x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    Returns negative sum of sides (to minimize) plus penalty for violations.
    """
    centers = x[0::3].reshape(-1, 2)[:, :2] if x.size % 3 == 0 else x.reshape(-1, 3)[:, :2]
    # Actually x is flat: [cx0, cy0, s0, cx1, cy1, s1, ...]
    centers = np.array([[x[i*3], x[i*3+1]] for i in range(n)])
    sides = np.array([x[i*3+2] for i in range(n)])
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < 0:
            penalty += (100 * abs(x0)) ** 2
        if x1 > 1:
            penalty += (100 * abs(x1 - 1)) ** 2
        if y0 < 0:
            penalty += (100 * abs(y0)) ** 2
        if y1 > 1:
            penalty += (100 * abs(y1 - 1)) ** 2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                # Overlap penalty: how much they overlap
                overlap_x = max(0, t - dx)
                overlap_y = max(0, t - dy)
                penalty += (100 * overlap_x * overlap_y) ** 2
    
    return -np.sum(sides) + penalty

def _refine_slsqp(x0, n=10, deadline=None):
    """
    Refine a solution using SLSQP with proper constraints.
    """
    # Bounds: cx in [0,1], cy in [0,1], s in [0.01, 0.6]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 0.6)) # s
    
    # Inequality constraints: for each pair, either |cx_i - cx_j| >= t_ij OR |cy_i - cy_j| >= t_ij
    # This is non-convex, so we use penalty method or reformulation
    # For SLSQP, we'll use a penalty-based approach within the objective
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x0 >= 0: cx_i - s_i/2 >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i*3] - x[i*3+2]/2
        })
        # x1 <= 1: cx_i + s_i/2 <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[i*3] - x[i*3+2]/2
        })
        # y0 >= 0: cy_i - s_i/2 >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]/2
        })
        # y1 <= 1: cy_i + s_i/2 <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[i*3+1] - x[i*3+2]/2
        })
    
    def _slsqp_objective(x):
        """Objective for SLSQP: minimize -sum(sides) + penalty for overlaps"""
        centers = np.array([[x[i*3], x[i*3+1]] for i in range(n)])
        sides = np.array([x[i*3+2] for i in range(n)])
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    overlap_x = max(0, t - dx)
                    overlap_y = max(0, t - dy)
                    penalty += 1000 * overlap_x * overlap_y
        
        return -np.sum(sides) + penalty
    
    try:
        result = minimize(
            _slsqp_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        return result.x
    except:
        return x0

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN
    
    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)
    
    # Differential evolution for global search
    # Bounds: cx in [0,1], cy in [0,1], s in [0.05, 0.5]
    bounds_de = []
    for i in range(n):
        bounds_de.append((0.0, 1.0))   # cx
        bounds_de.append((0.0, 1.0))   # cy
        bounds_de.append((0.05, 0.5))  # s
    
    # Use multiple random restarts with differential evolution
    np.random.seed(42)
    
    best_de_val = best_val
    best_de_centers = best_centers.copy()
    best_de_sides = best_sides.copy()
    
    # Run differential evolution with time budget
    de_budget = min(300, deadline - start_time - 60)  # Leave 60s for refinement
    if de_budget > 30:
        try:
            # Differential evolution
            de_result = differential_evolution(
                _objective,
                bounds_de,
                args=(n,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False,  # We'll do our own refinement
                updating='deferred',
                workers=-1
            )
            
            if de_result.success and de_result.fun < -best_de_val:
                x_de = de_result.x
                centers_de = np.array([[x_de[i*3], x_de[i*3+1]] for i in range(n)])
                sides_de = np.array([x_de[i*3+2] for i in range(n)])
                
                if _is_valid(centers_de, sides_de, n):
                    val_de = np.sum(sides_de)
                    if val_de > best_de_val:
                        best_de_val = val_de
                        best_de_centers = centers_de.copy()
                        best_de_sides = sides_de.copy()
        except Exception as e:
            pass
    
    # Update best if DE found something better
    if best_de_val > best_val:
        best_val = best_de_val
        best_centers = best_de_centers.copy()
        best_sides = best_de_sides.copy()
    
    # Refine with SLSQP from multiple starting points
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Try refinement from the best DE solution and grid baseline
        start_points = [
            np.concatenate([best_centers.ravel(), best_sides]),
            np.concatenate([best_de_centers.ravel(), best_de_sides]) if best_de_val > best_val else None,
        ]
        
        # Add some random perturbations of the best solution
        for _ in range(3):
            perturbed = best_centers.copy()
            perturbed_sides = best_sides.copy()
            perturbed[:, 0] += np.random.uniform(-0.05, 0.05, n)
            perturbed[:, 1] += np.random.uniform(-0.05, 0.05, n)
            perturbed_sides += np.random.uniform(-0.02, 0.02, n)
            # Clamp to valid range
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
            perturbed_sides = np.clip(perturbed_sides, 0.05, 0.5)
            start_points.append(np.concatenate([perturbed.ravel(), perturbed_sides]))
        
        for sp in start_points:
            if sp is None:
                continue
            if time.time() > deadline:
                break
            
            refined = _refine_slsqp(sp, n, deadline)
            if refined is not None:
                centers_r = np.array([[refined[i*3], refined[i*3+1]] for i in range(n)])
                sides_r = np.array([refined[i*3+2] for i in range(n)])
                
                if _is_valid(centers_r, sides_r, n):
                    val_r = np.sum(sides_r)
                    if val_r > best_val:
                        best_val = val_r
                        best_centers = centers_r.copy()
                        best_sides = sides_r.copy()
    
    # Final validation
    if not _is_valid(best_centers, best_sides, n):
        # Fallback to baseline
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
