import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    def check_validity(cx, cy, s):
        """Check if the configuration is valid."""
        # Boundary check
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        if np.any(s <= 0):
            return False
        if not np.all(np.isfinite(cx)) or not np.all(np.isfinite(cy)) or not np.all(np.isfinite(s)):
            return False
        
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True
    
    # Objective function: minimize -sum(sides)
    def objective(params):
        p = params.reshape(n, 3)
        s = p[:, 2]
        return -np.sum(s)
    
    # Boundary constraints: cx - s/2 >= -tol, cx + s/2 <= 1+tol, etc.
    # SLSQP requires g(x) >= 0 for inequality constraints
    def boundary_constraints(params):
        p = params.reshape(n, 3)
        cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
        cons = []
        for i in range(n):
            # Left: cx - s/2 + tol >= 0
            cons.append(cx[i] - s[i]/2 + tol)
            # Right: 1 + tol - cx - s/2 >= 0
            cons.append(1 + tol - cx[i] - s[i]/2)
            # Bottom: cy - s/2 + tol >= 0
            cons.append(cy[i] - s[i]/2 + tol)
            # Top: 1 + tol - cy - s/2 >= 0
            cons.append(1 + tol - cy[i] - s[i]/2)
        return np.array(cons)
    
    # Non-overlap constraints for each pair
    # For pair (i,j), we need: |cx_i - cx_j| >= (s_i+s_j)/2 - tol OR |cy_i - cy_j| >= (s_i+s_j)/2 - tol
    # This is a disjunction, which is non-smooth.
    # We'll use a penalty approach within SLSQP by adding smooth constraints.
    # Alternative: use multiple runs with different "separation directions"
    
    # For SLSQP, we can't directly express OR constraints.
    # Instead, we'll use a penalty method with SLSQP, or use multiple starting points
    # where we encourage separation in specific directions.
    
    def make_overlap_constraints(directions):
        """
        Create overlap constraints assuming separation in given directions.
        directions: list of (i, j, axis) where axis is 0 (x) or 1 (y)
        For each such pair, we enforce separation in that axis.
        """
        def constraints(params):
            p = params.reshape(n, 3)
            cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
            cons = []
            for (i, j, axis) in directions:
                if axis == 0:
                    # Enforce |cx_i - cx_j| >= (s_i+s_j)/2 - tol
                    # We need to handle the absolute value.
                    # We'll use two constraints: cx_i - cx_j >= t OR cx_j - cx_i >= t
                    # Since we don't know which, we'll use a smooth approximation.
                    # Actually, let's just penalize violations.
                    t = (s[i] + s[j]) / 2 - tol
                    dx = cx[i] - cx[j]
                    # We want dx >= t or -dx >= t, i.e., |dx| >= t
                    # Penalty: max(0, t - |dx|) should be 0
                    # Smooth: we can use -min(dx - t, -(dx + t)) but that's not smooth at 0
                    # Let's just add both as soft constraints
                    cons.append(dx - t)  # cx_i - cx_j >= t
                    cons.append(-dx - t)  # cx_j - cx_i >= t
                else:
                    t = (s[i] + s[j]) / 2 - tol
                    dy = cy[i] - cy[j]
                    cons.append(dy - t)
                    cons.append(-dy - t)
            return np.array(cons)
        return constraints
    
    # Better approach: use SLSQP with a penalty for overlaps
    def penalized_objective(params, penalty_weight=1000.0):
        p = params.reshape(n, 3)
        cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
        
        base = -np.sum(s)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # If both dx < t and dy < t, they overlap
                # Penalty: product of violations
                viol_x = max(0, t - dx)
                viol_y = max(0, t - dy)
                penalty += viol_x * viol_y
                # Also penalize if only one dimension violates (to encourage separation)
                # Actually, the product is zero if either is 0, so it's correct
        return base + penalty_weight * penalty
    
    # Generate initial configurations
    def generate_initial_configs():
        configs = []
        
        # Grid-like configurations
        # 2x5 grid
        for cols in [2, 3, 4, 5]:
            rows = (n + cols - 1) // cols
            if rows * cols < n:
                continue
            side = min(1.0/cols, 1.0/rows)
            idx = np.arange(n)
            c_idx = idx % cols
            r_idx = idx // cols
            cx = (c_idx + 0.5) / cols
            cy = (r_idx + 0.5) / rows
            s = np.full(n, side)
            params = np.column_stack([cx, cy, s]).flatten()
            configs.append(params)
        
        # Random configurations
        rng = np.random.RandomState(42)
        for _ in range(20):
            cx = rng.uniform(0.1, 0.9, n)
            cy = rng.uniform(0.1, 0.9, n)
            s = rng.uniform(0.05, 0.3, n)
            params = np.column_stack([cx, cy, s]).flatten()
            configs.append(params)
        
        return configs
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 1.0)] * n
    
    start_time = time.time()
    time_limit = 600
    safety_margin = 30
    
    best_result = None
    best_score = -np.inf
    
    # First, try SLSQP from multiple starting points
    initial_configs = generate_initial_configs()
    
    for init_params in initial_configs:
        if time.time() - start_time > time_limit - safety_margin:
            break
        
        try:
            result = minimize(
                penalized_objective,
                init_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.fun < best_score:
                p = result.x.reshape(n, 3)
                cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
                
                if check_validity(cx, cy, s):
                    score = np.sum(s)
                    if score > best_score:
                        best_score = score
                        best_result = (cx.copy(), cy.copy(), s.copy())
                        
        except Exception as e:
            pass
    
    # If SLSQP didn't find a good solution, try differential evolution
    if time.time() - start_time < time_limit - safety_margin:
        def de_objective(params):
            p = params.reshape(n, 3)
            cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
            
            penalty = 0.0
            # Boundary
            left = cx - s / 2.0
            right = cx + s / 2.0
            bottom = cy - s / 2.0
            top = cy + s / 2.0
            viol_left = np.maximum(0, -tol - left)
            viol_right = np.maximum(0, right - (1.0 + tol))
            viol_bottom = np.maximum(0, -tol - bottom)
            viol_top = np.maximum(0, top - (1.0 + tol))
            bound_penalty = np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)
            penalty += 100.0 * bound_penalty
            
            # Overlap
            for i in range(n):
                for j in range(i + 1, n):
                    t = (s[i] + s[j]) / 2 - tol
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    viol_x = max(0, t - dx)
                    viol_y = max(0, t - dy)
                    penalty += 1000.0 * viol_x * viol_y
            
            return -np.sum(s) + penalty
        
        for seed in [1, 2, 3]:
            if time.time() - start_time > time_limit - safety_margin:
                break
            
            try:
                result = differential_evolution(
                    de_objective,
                    bounds,
                    seed=seed,
                    maxiter=500,
                    popsize=15,
                    tol=1e-6,
                    disp=False,
                    polish=True
                )
                
                p = result.x.reshape(n, 3)
                cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
                
                if check_validity(cx, cy, s):
                    score = np.sum(s)
                    if score > best_score:
                        best_score = score
                        best_result = (cx.copy(), cy.copy(), s.copy())
                        
            except Exception as e:
                pass
    
    if best_result is None:
        # Fallback to grid
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    cx, cy, s = best_result
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    return centers, sides
