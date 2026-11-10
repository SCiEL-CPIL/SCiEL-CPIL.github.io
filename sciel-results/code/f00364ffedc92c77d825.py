import numpy as np
import time
import math
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    """
    Strategy: Direct discrete waypoint optimization.
    
    Instead of using a smooth parametric representation (like polynomials) which might 
    miss the exact discrete "pinch points" that limit the intersection area, we 
    parameterize the path by K explicit waypoints. The objective function computes the 
    EXACT score as defined by the evaluator (densifying the path and checking all grid 
    points). We use Differential Evolution (DE) to globally optimize these waypoints.
    
    This approach is robust to the non-convexity and discontinuities inherent in the 
    "intersection of moving sets" problem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit

    # Precompute the sofa grid
    # The task says: (-3 + 0.01i, -1.5 + 0.01j), i = 0..449, j = 0..299
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    gx = X.ravel()
    gy = Y.ravel()
    n_pts = len(gx)

    def _densify_path(path):
        """
        Densify path exactly as evaluator: prepend (0,0,0), append (-2,0,90),
        then interpolate between consecutive points in m = max(1, ceil(|da|)) steps.
        Returns array of shape (N, 3).
        """
        # Ensure path is float array
        path = np.asarray(path, dtype=float)
        if len(path) == 0:
            full = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 90.0]])
        else:
            full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
        
        result = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(math.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                result.append(interp)
        return np.array(result)

    def _evaluate_path_vectorized(path):
        """
        Vectorized evaluation: compute for all placements at once using broadcasting.
        Returns the raw value (area) as a float.
        """
        dense = _densify_path(path)
        n_placements = len(dense)
        
        if n_placements == 0:
            return 0.0

        # Extract placement parameters
        cx = dense[:, 0]
        cy = dense[:, 1]
        a_deg = dense[:, 2]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Compute rotated coordinates for all grid points and all placements
        # We do this in chunks to avoid memory explosion if n_placements is large, 
        # though for K=10 waypoints, n_placements is roughly 90 + 20 = 110, which is fine.
        # n_pts is ~ 135,000. 110 * 135,000 * 8 bytes (float64) * 2 (u,v) ~ 240 MB. 
        # This is borderline. Let's chunk the placements to be safe.
        
        chunk_size = 20
        total_mask = np.ones(n_pts, dtype=bool)
        
        for start in range(0, n_placements, chunk_size):
            end = min(start + chunk_size, n_placements)
            c_chunk = cx[start:end]
            cy_chunk = cy[start:end]
            a_rad_chunk = a_rad[start:end]
            cos_chunk = np.cos(a_rad_chunk)
            sin_chunk = np.sin(a_rad_chunk)
            
            dx = gx[None, :] - c_chunk[:, None]
            dy = gy[None, :] - cy_chunk[:, None]
            u = cos_chunk[:, None] * dx + sin_chunk[:, None] * dy
            v = -sin_chunk[:, None] * dx + cos_chunk[:, None] * dy
            
            in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
            in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            in_base = in_arm1 | in_arm2
            
            # Point must be in ALL placements in this chunk
            mask_chunk = in_base.all(axis=0)
            total_mask &= mask_chunk
            
            # Early exit if no points remain
            if not np.any(total_mask):
                break

        return float(np.sum(total_mask)) * 0.0001

    # Optimization setup
    # We define K interior waypoints.
    # The path is: (0,0,0) -> wp_1 -> ... -> wp_K -> (-2,0,90)
    # We optimize the coordinates of wp_1 ... wp_K.
    
    K = 12  # Number of interior waypoints
    
    # Bounds for each waypoint:
    # x: The sofa moves from x=0 to x=-2. It might swing out. 
    #    The corridor is width 1. The sofa is 3x3. 
    #    Safe bounds: x in [-4, 2], y in [-3, 3], theta in [-10, 100] (monotonic-ish)
    
    # To enforce monotonicity of angle roughly, we can just let DE find it, 
    # but the densification handles non-monotonic angles by just interpolating.
    # However, large angle swings create many interpolation steps, which is good for resolution 
    # but bad for "smoothness" if it creates sharp turns. 
    # The evaluator adds steps based on |da|. So if we jump 90 deg in one step, we get 90 steps.
    # This is actually beneficial for capturing the intersection area accurately.
    
    # Let's define bounds for the K waypoints.
    # x: [-3, 1]
    # y: [-2, 2]
    # theta: [0, 90] (Keep it monotonic to avoid weird folding back which likely reduces intersection)
    
    bounds = []
    for i in range(K):
        bounds.append((-3.0, 1.0))  # x
        bounds.append((-2.0, 2.0))  # y
        bounds.append((0.0, 90.0))  # theta

    def _params_to_path(params):
        """
        params: flat array of length 3*K
        Returns path array of shape (K, 3)
        """
        path = params.reshape(K, 3)
        return path

    def _objective(params):
        if time.time() > deadline:
            return 0.0
        
        path = _params_to_path(params)
        
        # Optional: Enforce monotonicity of theta to prevent "back-and-forth" which is suboptimal
        # If theta goes up and down, we waste "steps" and the intersection might shrink due to redundant constraints.
        # Simple penalty for non-monotonicity:
        thetas = path[:, 2]
        diffs = np.diff(thetas)
        if np.any(diffs < -1e-6):
            # Penalize non-monotonic angle
            penalty = np.sum(diffs[diffs < 0]) * 10.0
            # We return -score + penalty. Since we minimize, large penalty is bad.
            # But DE minimizes. So we want to return a large number for bad paths.
            # Let's just clamp thetas to be monotonic in the path generation? 
            # No, let's just let the evaluator handle it. The "intersection of all" logic 
            # naturally penalizes redundant or conflicting placements.
            pass
        
        score = _evaluate_path_vectorized(path)
        return -score

    # Run Differential Evolution
    # DE is good for global optimization in moderate dimensions.
    # popsize: 15 * K? K=12, so 180 individuals. 
    # Each evaluation is ~100ms. 180 * 1000 iterations = 180,000 evals = 18000s. Too slow.
    # We need to be smarter.
    # Let's reduce K to 8 and popsize to 10.
    # Or use a coarser grid for DE and then refine?
    
    # Let's try K=8, popsize=10, maxiter=200.
    # 10 * 200 = 2000 evals. 2000 * 0.1s = 200s. This fits in the budget.
    
    K_opt = 8
    bounds_opt = []
    for i in range(K_opt):
        bounds_opt.append((-3.0, 1.0))
        bounds_opt.append((-2.0, 2.0))
        bounds_opt.append((0.0, 90.0))
        
    def _objective_opt(params):
        if time.time() > deadline:
            return 0.0
        path = params.reshape(K_opt, 3)
        score = _evaluate_path_vectorized(path)
        return -score

    best_path = None
    best_score = -1.0

    # Initial guess: linear interpolation
    t_init = np.linspace(0, 1, K_opt + 2)[1:-1]
    x_init = 0 + t_init * (-2 - 0)
    y_init = 0 + t_init * (0 - 0)
    a_init = 0 + t_init * (90 - 0)
    init_params = np.column_stack((x_init, y_init, a_init)).ravel()

    # Run DE
    try:
        res = differential_evolution(
            _objective_opt, 
            bounds=bounds_opt, 
            seed=42,
            popsize=10,
            maxiter=150,
            tol=1e-4,
            init='sobol'
        )
        if res.fun < -best_score:
            best_score = -res.fun
            best_path = res.x.reshape(K_opt, 3)
    except Exception as e:
        pass

    # Refine with a local optimizer (Nelder-Mead) on the best path found
    if best_path is not None:
        best_params = best_path.ravel()
        
        # Try a few random restarts around the best solution
        np.random.seed(42)
        for r in range(5):
            if time.time() > deadline - 30:
                break
            
            noise = np.random.normal(0, 0.5, size=len(best_params))
            x_try = best_params + noise
            
            def _local_obj(p):
                if time.time() > deadline:
                    return 0.0
                path = p.reshape(K_opt, 3)
                # Ensure angles are within bounds roughly to avoid weird stuff
                score = _evaluate_path_vectorized(path)
                return -score

            try:
                from scipy.optimize import minimize
                res_local = minimize(_local_obj, x_try, method='Nelder-Mead', 
                                     options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
                if res_local.fun < -best_score:
                    best_score = -res_local.fun
                    best_params = res_local.x
            except Exception:
                pass

        best_path = best_params.reshape(K_opt, 3)
    else:
        # Fallback to initial guess
        best_path = np.column_stack((x_init, y_init, a_init))

    # Final verification and return
    # Ensure the path is valid
    if len(best_path) == 0:
        return np.array([[0.0, 0.0, 0.0]])
    
    return best_path

# EVOLVE-BLOCK-END
