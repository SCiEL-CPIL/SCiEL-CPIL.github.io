# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Direct optimization of discrete waypoints using Differential Evolution.
    The path is defined by N intermediate points. The evaluator interpolates linearly
    between these points and the fixed start/end. We optimize the (x, y) of these 
    intermediate points to maximize the area of the sofa that remains inside the 
    corridor throughout the move.
    """
    
    # 1. Setup Grid
    # Sofa points: x in [-3, 1.5), y in [-1.5, 1.5), step 0.01
    # Note: np.arange(-3, 1.5, 0.01) creates 450 points.
    # np.arange(-1.5, 1.5, 0.01) creates 300 points.
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    # Start and End states
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    # Define the number of intermediate waypoints to optimize.
    # The evaluator adds linear interpolations between consecutive rows.
    # If we return K rows, the total sequence is:
    # Start, W1, W2, ..., WK, End
    # The evaluator interpolates between each pair.
    # We want a dense enough path to allow complex shapes, but not so many
    # that the optimization dimension is too high or evaluation is too slow.
    # Let's try K=10 intermediate points.
    # Total path points for evaluation: Start + K + End = K+2 points.
    # However, the evaluator *also* adds interpolations.
    # "Between consecutive rows it adds linear interpolations ... in m = max(1, ceil(|a2 - a1|)) equal steps"
    # This means if we provide points with angle steps of, say, 10 degrees, 
    # the evaluator will add 9 intermediate interpolated points between them.
    # So the actual path sampled by the evaluator is much denser than our waypoints.
    # This is good: we define the "control points" and the evaluator densifies.
    
    K = 10  # Number of intermediate waypoints
    
    # We optimize the (x, y) for these K waypoints.
    # The angle a for these waypoints should be evenly spaced between 0 and 90
    # to ensure the interpolation steps are reasonable and cover the turn smoothly.
    # If we choose angles evenly, the interpolation steps will be uniform.
    # a_i = 90 * (i+1) / (K+1) for i in 0..K-1
    
    a_waypoints = np.linspace(90.0 / (K + 1), 90.0 * K / (K + 1), K)
    
    # Bounds for x and y.
    # The inner corner (x,y) must move such that the corridor covers the sofa.
    # The corridor is roughly 4x4. The sofa is 4.5x3.
    # x should probably stay in [-3, 1] and y in [-1, 1] roughly.
    # Let's use generous bounds: x in [-4, 2], y in [-2, 2].
    x_bounds = (-4.0, 2.0)
    y_bounds = (-2.0, 2.0)
    
    # Bounds array for scipy.optimize.differential_evolution
    # The vector to optimize is [x1, y1, x2, y2, ..., xK, yK]
    bounds = []
    for _ in range(K):
        bounds.append(x_bounds)
        bounds.append(y_bounds)
        
    def evaluate_path(params):
        """
        params: array of length 2*K, [x1, y1, ..., xK, yK]
        Returns the area (score) of the sofa inside the corridor.
        """
        # Reconstruct the full path including start and end
        # The evaluator sequence is: Start, W1, ..., WK, End
        # We need to evaluate the intersection of the sofa with the corridor
        # for ALL placements in the sequence.
        # The evaluator inserts interpolated points.
        # To be safe and accurate, we should simulate the evaluator's interpolation.
        
        # Build the list of (x, y, a) waypoints
        waypoints = []
        # Start
        waypoints.append((0.0, 0.0, 0.0))
        
        # Intermediate
        for i in range(K):
            xi = params[2*i]
            yi = params[2*i+1]
            ai = a_waypoints[i]
            waypoints.append((xi, yi, ai))
            
        # End
        waypoints.append((-2.0, 0.0, 90.0))
        
        # Now, generate the full dense path as the evaluator would.
        # "Between consecutive rows it adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
        # This implies that for each segment (p1, p2), we create m+1 points (including p1 and p2? 
        # Usually "interpolations in m steps" means m intermediate points, so m+1 total points per segment, 
        # but the endpoints are shared. Let's assume we generate the points strictly between, 
        # or just generate the full sequence including endpoints and dedupe.
        
        # Let's generate the sequence of placements to check.
        # We need to check if a sofa point is inside *all* placements.
        # So we compute the mask for each placement and AND them.
        
        mask = np.ones(N_points, dtype=bool)
        
        # Helper to check one placement
        def check_placement(x, y, a_deg):
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = wx - x
            dy = wy - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Base corridor:
            # R1: 0 <= u <= 1, -3 <= v <= 1
            # R2: -3 <= u <= 1, 0 <= v <= 1
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            return in_r1 | in_r2

        # Iterate through segments
        for i in range(len(waypoints) - 1):
            x1, y1, a1 = waypoints[i]
            x2, y2, a2 = waypoints[i + 1]
            
            # m = max(1, ceil(|a2 - a1|))
            # a is in degrees.
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))
            
            # Generate m+1 points? Or m intermediate?
            # "adds linear interpolations ... in m ... steps"
            # Usually this means the segment is divided into m sub-segments,
            # resulting in m+1 points (including endpoints).
            # Since the endpoints are handled by the loop iteration boundaries,
            # we need to be careful not to double-check or miss.
            # Let's generate t from 0 to 1 in m+1 steps (0, 1/m, ..., 1).
            # This ensures we check the endpoints of the segment.
            
            ts = np.linspace(0, 1, m + 1)
            
            for t in ts:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                a = a1 + t * (a2 - a1)
                
                in_corridor = check_placement(x, y, a)
                mask &= in_corridor
                
                # Early exit if no points are left
                if not np.any(mask):
                    return 0.0
        
        return float(np.sum(mask)) * 0.0001

    # Optimization
    # Differential Evolution is a global optimizer.
    # It is slow but robust.
    # We have 20 parameters (K=10).
    # We have 600s.
    # Let's set a deadline.
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial population size
    # popsize=15 * dim = 300 initial points.
    # maxiter=100
    # This might be too slow. Let's monitor time.
    
    # To speed up, we can reduce K or use a faster optimizer.
    # Let's try K=5 first (10 params) to see if it's fast enough.
    # If K=5 is too coarse, we can refine.
    # Actually, let's stick with K=8 (16 params) as a balance.
    
    # Re-initialize K and bounds for the actual run
    K = 8
    a_waypoints = np.linspace(90.0 / (K + 1), 90.0 * K / (K + 1), K)
    bounds = []
    for _ in range(K):
        bounds.append(x_bounds)
        bounds.append(y_bounds)
        
    # Reset function scope variables if necessary? 
    # The function `evaluate_path` captures K and a_waypoints from the outer scope.
    # We must ensure K and a_waypoints are updated before calling.
    # In Python, closures capture variables by reference. 
    # So updating K and a_waypoints here will affect evaluate_path.
    
    # Let's use a wrapper or just ensure the variables are defined.
    # The `evaluate_path` defined above uses `K` and `a_waypoints` from the local scope of `moving_sofa_2d`.
    # Since we are redefining them, it's fine.
    
    best_score = 0.0
    best_params = np.random.uniform(x_bounds[0], x_bounds[1], 2*K)
    
    # Run Differential Evolution
    try:
        # tol=1e-4, maxiter=20
        # We want to finish quickly to allow for potential restarts or refinement.
        # DE is stochastic.
        
        # To make it faster, we can use a smaller population.
        # popsize=10
        result = scipy.optimize.differential_evolution(
            lambda p: -evaluate_path(p),
            bounds,
            maxiter=30,
            popsize=15,
            tol=1e-4,
            seed=42,
            # Callback to check time?
            # scipy.optimize.differential_evolution doesn't have a built-in time callback in all versions
            # but we can check after.
        )
        
        score = -result.fun
        if score > best_score:
            best_score = score
            best_params = result.x
            
    except Exception as e:
        # Fallback to a simple random search if DE fails or is too slow
        pass

    # If we have time left, do a local refinement with Nelder-Mead on the best found
    current_time = time.time()
    if current_time < deadline - 60: # 60s buffer
        try:
            # Nelder-Mead on the best params found
            res_nelder = scipy.optimize.minimize(
                lambda p: -evaluate_path(p),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            score_nelder = -res_nelder.fun
            if score_nelder > best_score:
                best_score = score_nelder
                best_params = res_nelder.x
        except Exception:
            pass

    # Construct the final output array
    # We return the intermediate waypoints.
    # Shape (K, 3)
    
    result_path = np.zeros((K, 3))
    for i in range(K):
        result_path[i, 0] = best_params[2*i]
        result_path[i, 1] = best_params[2*i+1]
        result_path[i, 2] = a_waypoints[i]
        
    return result_path

# EVOLVE-BLOCK-END
