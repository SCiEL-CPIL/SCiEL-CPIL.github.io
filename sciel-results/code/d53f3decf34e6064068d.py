# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    start_time = time.time()
    time_limit = 620.0
    deadline = start_time + time_limit
    
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = c * wx - s * wy
        v = s * wx + c * wy
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def get_placements(path):
        """Generate all placements from a path including endpoints and interpolations."""
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m + 1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        # Remove duplicate consecutive placements
        unique_placements = []
        prev = None
        for p in placements:
            if prev is None or abs(p[0]-prev[0]) > 1e-10 or abs(p[1]-prev[1]) > 1e-10 or abs(p[2]-prev[2]) > 1e-10:
                unique_placements.append(p)
            prev = p
        
        return unique_placements
    
    def evaluate_path(path):
        """Compute the hard score: number of sofa points inside all placements."""
        placements = get_placements(path)
        
        mask = np.ones(n_points, dtype=bool)
        
        for x, y, a in placements:
            inside = check_inside(x, y, a)
            mask &= inside
            
            if not mask.any():
                return 0.0
        
        return np.sum(mask) * 0.0001
    
    def evaluate_path_soft(path, beta=20.0):
        """Compute a smooth surrogate score using soft min of margins."""
        placements = get_placements(path)
        n_placements = len(placements)
        
        if n_placements == 0:
            return 0.0
        
        # For each point, compute the minimum margin across all placements
        # margin > 0 means inside, margin < 0 means outside
        # soft_min = -beta * log(sum(exp(-margin_j / beta)))
        
        # We'll accumulate log-sum-exp for numerical stability
        log_sum_exp = np.full(n_points, -np.inf, dtype=np.float64)
        
        for x, y, a in placements:
            theta = -np.deg2rad(a)
            c = np.cos(theta)
            s = np.sin(theta)
            
            wx = X_flat - x
            wy = Y_flat - y
            
            u = c * wx - s * wy
            v = s * wx + c * wy
            
            # Distance to boundary of each rectangle (positive = inside)
            # For r1: [0,1] x [-3,1]
            d1_u = np.minimum(u, 1.0 - u)
            d1_v = np.minimum(v + 3.0, 1.0 - v)
            dist_r1 = np.minimum(d1_u, d1_v)
            
            # For r2: [-3,1] x [0,1]
            d2_u = np.minimum(u + 3.0, 1.0 - u)
            d2_v = np.minimum(v, 1.0 - v)
            dist_r2 = np.minimum(d2_u, d2_v)
            
            # Distance to the L-shaped corridor boundary
            # A point is inside if it's in either rectangle
            # The margin is the distance to the nearest boundary point
            # If in r1: margin = dist_r1
            # If in r2: margin = dist_r2
            # If in both: margin = max(dist_r1, dist_r2)
            # If in neither: margin = -max(dist_to_r1_boundary, dist_to_r2_boundary)
            
            in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
            in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            
            # Compute margin
            # If in both rectangles, margin is max of the two distances
            # If in only one, margin is that distance
            # If in neither, margin is negative (distance to nearest boundary)
            
            margin = np.zeros(n_points, dtype=np.float64)
            
            # For points in both rectangles
            both = in_r1 & in_r2
            margin[both] = np.maximum(dist_r1[both], dist_r2[both])
            
            # For points in only r1
            only_r1 = in_r1 & ~in_r2
            margin[only_r1] = dist_r1[only_r1]
            
            # For points in only r2
            only_r2 = in_r2 & ~in_r1
            margin[only_r2] = dist_r2[only_r2]
            
            # For points in neither
            neither = ~in_r1 & ~in_r2
            # Distance to r1 boundary
            d_to_r1 = np.maximum(
                np.maximum(0.0 - u[neither], u[neither] - 1.0),
                np.maximum(-3.0 - v[neither], v[neither] - 1.0)
            )
            # Distance to r2 boundary
            d_to_r2 = np.maximum(
                np.maximum(-3.0 - u[neither], u[neither] - 1.0),
                np.maximum(0.0 - v[neither], v[neither] - 1.0)
            )
            # The point is outside the L-shape, so margin is negative
            # The distance to the nearest boundary point
            margin[neither] = -np.minimum(d_to_r1, d_to_r2)
            
            # Update log-sum-exp
            # log_sum_exp = log(sum(exp(-margin_j / beta)))
            # Using the trick: log(a + b) = max(a,b) + log(1 + exp(-|a-b|))
            current = -margin / beta
            
            # For numerical stability, use the max trick
            max_val = np.maximum(log_sum_exp, current)
            log_sum_exp = max_val + np.log(np.exp(log_sum_exp - max_val) + np.exp(current - max_val))
        
        # soft_min = -beta * log_sum_exp
        soft_min = -beta * log_sum_exp
        
        # Points with soft_min > 0 are "safely inside"
        # Convert to a count-like score
        # Use a smooth step function
        safe = 1.0 / (1.0 + np.exp(-soft_min * 5.0))
        
        return np.sum(safe) * 0.0001
    
    # Initial seed path
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_path = seed_path.copy()
    best_score = evaluate_path(best_path)
    
    def make_path_from_params(params):
        """Convert parameter vector to path.
        Parameters: for each of 8 waypoints, (x, y). Angles are fixed."""
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 86.0]
        path = []
        for i, a in enumerate(angles):
            path.append([params[2*i], params[2*i+1], a])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters
    current_params = np.array([
        -0.05, 0.05,
        -0.15, 0.15,
        -0.25, 0.25,
        -0.3, 0.3,
        -0.45, 0.25,
        -0.55, 0.2,
        -0.65, 0.22,
        -0.72, 0.28
    ])
    
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit_check=None, step_size=0.1):
        """Nelder-Mead optimization with time limit."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += step_size
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_params = x0.copy()
        best_val = min(fvals)
        
        for iteration in range(max_iter):
            if time_limit_check is not None and time.time() > time_limit_check:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            # Track best
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Phase 1: Optimize with soft objective (smoother landscape)
    def objective_soft(params):
        path = make_path_from_params(params)
        return -evaluate_path_soft(path, beta=20.0)
    
    if time.time() < deadline - 60:
        best_params_soft, best_val_soft = nelder_mead(
            objective_soft, current_params,
            max_iter=150, tol=1e-5,
            time_limit_check=deadline - 60,
            step_size=0.1
        )
        
        path_soft = make_path_from_params(best_params_soft)
        score_soft = evaluate_path(path_soft)
        
        if score_soft > best_score:
            best_score = score_soft
            best_path = path_soft
            current_params = best_params_soft
        else:
            current_params = best_params_soft  # Still use the soft-optimized params as starting point
    
    # Phase 2: Refine with hard objective
    def objective_hard(params):
        path = make_path_from_params(params)
        return -evaluate_path(path)
    
    if time.time() < deadline - 30:
        best_params_hard, best_val_hard = nelder_mead(
            objective_hard, current_params,
            max_iter=200, tol=1e-7,
            time_limit_check=deadline - 30,
            step_size=0.05
        )
        
        path_hard = make_path_from_params(best_params_hard)
        score_hard = evaluate_path(path_hard)
        
        if score_hard > best_score:
            best_score = score_hard
            best_path = path_hard
            current_params = best_params_hard
    
    # Phase 3: Random restarts with hard objective
    np.random.seed(42)
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Perturb around current best
        perturbation_scale = 0.15
        perturbed = current_params + np.random.normal(0, perturbation_scale, size=len(current_params))
        
        bp, bv = nelder_mead(
            objective_hard, perturbed,
            max_iter=80, tol=1e-6,
            time_limit_check=deadline - 15,
            step_size=0.08
        )
        
        path_r = make_path_from_params(bp)
        score_r = evaluate_path(path_r)
        
        if score_r > best_score:
            best_score = score_r
            best_path = path_r
            current_params = bp
    
    # Phase 4: Fine-tune the best solution
    if time.time() < deadline - 10:
        # Try very small perturbations
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            small_perturbation = 0.02
            perturbed = current_params + np.random.normal(0, small_perturbation, size=len(current_params))
            
            bp, bv = nelder_mead(
                objective_hard, perturbed,
                max_iter=50, tol=1e-8,
                time_limit_check=deadline - 5,
                step_size=0.02
            )
            
            path_f = make_path_from_params(bp)
            score_f = evaluate_path(path_f)
            
            if score_f > best_score:
                best_score = score_f
                best_path = path_f
                current_params = bp
    
    return best_path

# EVOLVE-BLOCK-END
