import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    n = 8
    
    # Precompute indices for unordered pairs (i < j)
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    num_pairs = len(pair_indices)
    
    # Constants for the objective function
    # Score = (Product of all 56 ordered pair distances) / 8^8
    # After scaling to diameter D=2:
    # d_scaled = d * (2/D)
    # Product_scaled = Product(d) * (2/D)^56
    # log(Score) = log(Product(d)) + 56*log(2) - 56*log(D) - 8*log(8)
    # log(Product(d)) = 2 * sum_{i<j} log(d_ij)
    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    CONST_OFFSET = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective_full(x):
        """
        x: 16-vector representing 8 points in 2D.
        Returns -log(Score) for minimization.
        """
        if time.time() > deadline:
            return 1e10
        
        points = x.reshape(n, 2)
        
        # Calculate pairwise distances for i < j
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Check for zero distances (distinct points required)
        if np.any(dists < 1e-12):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-12:
            return 1e10
            
        # Sum of logs of distances
        sum_log_d = np.sum(np.log(dists))
        
        # log(Score) = CONST_OFFSET + 2 * sum_log_d - 56 * log(D)
        log_score = CONST_OFFSET + 2.0 * sum_log_d - 56.0 * np.log(D)
        
        return -log_score

    def two_squares_points(r1, r2, theta):
        """
        Generate 8 points as two concentric squares.
        Square 1: radius r1, angle offset 0
        Square 2: radius r2, angle offset theta
        """
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        # Interleave or just stack? The order doesn't matter for the set of points.
        # Let's just stack them.
        pts = np.vstack([p1, p2])
        return pts.flatten()

    def objective_manifold(params):
        """
        params: [r1, r2, theta]
        Returns -log(Score) for the two-squares configuration.
        """
        if time.time() > deadline:
            return 1e10
            
        r1, r2, theta = params
        
        # Ensure radii are positive
        if r1 < 1e-9 or r2 < 1e-9:
            return 1e10
            
        pts_flat = two_squares_points(r1, r2, theta)
        
        # Use the full objective function logic
        return objective_full(pts_flat)

    best_val = float('inf')
    best_x = np.zeros(2*n)
    
    # --- Phase 1: Optimize the Two-Squares Manifold ---
    # This is highly likely to contain the global optimum for n=8.
    # Bounds: r1, r2 in [0.1, 2.0], theta in [0, pi/4] (symmetry of square)
    bounds_manifold = [(0.1, 2.0), (0.1, 2.0), (0.0, np.pi/4)]
    
    try:
        de_res = differential_evolution(
            objective_manifold,
            bounds_manifold,
            maxiter=300,
            popsize=30,
            tol=1e-10,
            seed=42,
            init='latinhypercube',
            polish=True,
            polish_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200}}
        )
        
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_x = two_squares_points(*de_res.x)
            
    except Exception:
        pass

    # --- Phase 2: Local Refinement in Full 16D Space ---
    # Start from the best symmetric solution and allow symmetry breaking.
    if time.time() < deadline - 20.0:
        try:
            # Use Powell for robustness
            res_powell = minimize(
                objective_full,
                best_x,
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res_powell.fun < best_val:
                best_val = res_powell.fun
                best_x = res_powell.x
        except Exception:
            pass

        # Use L-BFGS-B for high precision
        if time.time() < deadline - 10.0:
            try:
                res_lbfgs = minimize(
                    objective_full,
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res_lbfgs.fun < best_val:
                    best_val = res_lbfgs.fun
                    best_x = res_lbfgs.x
            except Exception:
                pass

    # --- Phase 3: Diverse Full-Space Search (if time permits) ---
    # In case the two-squares structure is not optimal, try some random starts.
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        # Regular octagon start
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x0_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
        starts = [x0_oct]
        # Add some random perturbations
        for _ in range(10):
            noise = np.random.normal(0, 0.5, 2*n)
            starts.append(x0_oct + noise)
        
        # Add some random uniform starts
        for _ in range(10):
            starts.append(np.random.uniform(-2, 2, 2*n))
            
        for s in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    objective_full,
                    s,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    # --- Final Validation and Return ---
    best_points = best_x.reshape(n, 2)
    
    # Ensure distinctness
    min_d = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-8:
        # Nudge closest pair
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-8:
                    best_points[j] += np.array([1e-6, 0])
                    break
            if min_d >= 1e-8:
                break

    # Scale to ensure coordinates are within bounds (|coord| < 1e6)
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
