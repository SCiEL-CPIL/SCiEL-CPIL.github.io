import numpy as np
import time
import sys

def moving_sofa_2d() -> np.ndarray:
    # Time management
    start_time = time.time()
    deadline = start_time + 580  # 580s limit (20s margin before 600s, 50s before 630s kill)

    # Grid definition
    # i = 0..449 -> -3 + 0.01i
    # j = 0..299 -> -1.5 + 0.01j
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    # Flatten for easier vectorization
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    # Start and End states added by evaluator
    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Takes user path (n, 3), prepends start, appends end.
        Interpolates linearly.
        Returns list of (x, y, a) arrays.
        """
        if len(path) == 0:
            return np.array([P_start, P_end])
        
        full_path = np.vstack([P_start, path, P_end])
        # full_path is (n+2, 3)
        
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            # m = max(1, ceil(|a2 - a1|))
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            if m == 1:
                # Just add p2? No, the loop adds p2 at the end of the next iteration or now?
                # "adds linear interpolations ... in m equal steps"
                # If m=1, the step is from p1 to p2.
                # The points are p1 + t*(p2-p1) for t in [1/m, ..., 1]
                # So we add p2.
                dense_points.append(p2)
            else:
                for k in range(1, m + 1):
                    t = k / m
                    p_int = p1 + t * (p2 - p1)
                    dense_points.append(p_int)
        
        return np.array(dense_points)

    def check_inside_fast(placements, valid_mask=None):
        """
        placements: array of shape (M, 3)
        valid_mask: optional boolean mask of points to check. If None, starts with all True.
        Returns final valid_mask.
        """
        if valid_mask is None:
            valid_mask = np.ones(n_points, dtype=bool)
        
        # Pre-filter points? No, just use the mask in the operations to save time?
        # Actually, numpy operations on full array are faster if mask is dense.
        # If mask becomes sparse, extracting indices is faster.
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Local coordinates
            dx = X_flat - x
            dy = Y_flat - y
            
            # R(-a) * (dx, dy)
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Base corridor:
            # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            # If valid_mask is sparse, we can just update the subset
            if np.count_nonzero(valid_mask) < 0.2 * n_points:
                indices = np.nonzero(valid_mask)[0]
                if len(indices) > 0:
                    u_sub = u[indices]
                    v_sub = v[indices]
                    in_r1 = (u_sub >= 0) & (u_sub <= 1) & (v_sub >= -3) & (v_sub <= 1)
                    in_r2 = (u_sub >= -3) & (u_sub <= 1) & (v_sub >= 0) & (v_sub <= 1)
                    inside_sub = in_r1 | in_r2
                    valid_mask[indices] = inside_sub
            else:
                valid_mask = valid_mask & inside
            
            # Early exit
            if not np.any(valid_mask):
                return valid_mask

        return valid_mask

    def score_path(path):
        dense = densify_path(path)
        mask = check_inside_fast(dense)
        count = np.count_nonzero(mask)
        return count * 0.0001

    # Initial Candidates
    # The Seed
    seed_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = seed_path
    best_score = score_path(best_path)
    
    # Strategy: 
    # 1. Random restarts to find good basins.
    # 2. Local refinement (Hill Climbing / Coordinate Ascent) on the best found so far.
    # 3. Use Nelder-Mead or Powell if available? Let's stick to robust manual coordinate ascent to avoid scipy dependency issues or overhead, 
    #    but scipy is allowed. Let's use scipy.optimize.minimize with Nelder-Mead on the negative score.
    #    However, the function is non-smooth (area of intersection changes discretely). 
    #    Gradient-based methods won't work well. 
    #    Direct search methods are better.
    #    Nelder-Mead is a direct search method.
    
    from scipy.optimize import minimize, differential_evolution
    
    # Objective function for minimization (minimize negative score)
    # Parameters: flatten the path. 
    # Problem: Path length is variable. 
    # Let's fix the number of waypoints to K. 
    # If we fix K, we can optimize the positions and angles.
    
    K = 6 # Number of intermediate waypoints
    
    def path_to_params(path):
        """
        If path has K points, return flat array of 3*K.
        If not, pad or truncate? 
        For the optimizer, we need a fixed dimension.
        """
        if len(path) == K:
            return path.flatten()
        else:
            # Interpolate path to K points? 
            # This is complex. 
            # Instead, let's just run the optimizer on a fixed-size path, and separately run the variable-size random search.
            # We will convert the best variable-size path to a fixed-size K path by resampling.
            pass
            
    def params_to_path(params):
        """
        Convert 3*K params to a (K, 3) path.
        Constraints:
        x in [-1.5, 0.5]
        y in [-0.5, 1.5]
        a in [5, 85]
        """
        path = params.reshape(K, 3)
        # Clip to bounds
        path[:, 0] = np.clip(path[:, 0], -1.5, 0.5)
        path[:, 1] = np.clip(path[:, 1], -0.5, 1.5)
        path[:, 2] = np.clip(path[:, 2], 5, 85)
        # Sort by angle? 
        # The evaluator assumes the path is given in order. 
        # If we optimize parameters, we should ensure the angles are increasing to avoid backtracking, which is likely suboptimal.
        # Enforce monotonicity in the objective or transform?
        # Easier: Just clip and sort? No, sorting changes the meaning of the parameters.
        # Let's penalize non-monotonic angles in the objective? 
        # Or just force them to be sorted in params_to_path.
        idx = np.argsort(path[:, 2])
        path = path[idx]
        return path

    def objective(params):
        path = params_to_path(params)
        s = score_path(path)
        # Add penalty if angles are not strictly increasing? 
        # params_to_path sorts them, so the path is always monotonic.
        # But the mapping from params to path is discontinuous if sorting swaps elements.
        # This makes gradient-free methods like Nelder-Mead struggle with the discontinuities.
        
        # Alternative: Enforce order in the parameter space.
        # a_1 <= a_2 <= ... <= a_K
        # This is a constrained optimization.
        
        # For now, let's just minimize -score.
        return -s

    # Phase 1: Random Search to find a good initial point for the fixed-K optimizer
    rng = np.random.default_rng(42)
    
    # Generate a random path of size K
    def generate_random_path_k(K):
        angles = np.sort(rng.uniform(10, 80, K))
        xs = rng.uniform(-1.2, 0.2, K)
        ys = rng.uniform(-0.2, 1.2, K)
        # Ensure smoothness? No, random is fine for global search.
        path = np.column_stack([xs, ys, angles])
        return path

    # Initialize best for fixed-K optimization
    best_params_k = None
    best_score_k = -np.inf
    
    # Try a few random initializations for the fixed-K problem
    for i in range(5):
        p = generate_random_path_k(K)
        params = p.flatten()
        s = score_path(p)
        if s > best_score_k:
            best_score_k = s
            best_params_k = params
    
    # Update global best
    if best_score_k > best_score:
        best_score = best_score_k
        best_path = params_to_path(best_params_k)
        
    # Phase 2: Optimize fixed-K path using Nelder-Mead
    # Nelder-Mead can handle the noise/discontinuities somewhat if we use appropriate options.
    # However, it's slow. 
    # Let's try differential_evolution first for a global search over the fixed-K space.
    # DE is robust to discontinuities.
    
    bounds = [(-1.5, 0.5), (-0.5, 1.5), (5, 85)] * K
    
    # DE is expensive. Let's use a small population.
    # maxiter=10, tol=1e-3
    
    # Check time
    if time.time() < deadline - 100:
        try:
            # Use DE for a few iterations
            # Note: DE minimizes. We want to maximize score, so minimize -score.
            # objective returns -score.
            
            # To speed up DE, we can wrap objective to limit evals? 
            # Or just run it with maxiter=5.
            
            result = differential_evolution(
                objective, bounds, 
                maxiter=5, 
                popsize=10, 
                seed=42, 
                tol=1e-2, 
                init='latinhypercube'
            )
            
            if result.fun < -best_score_k: # result.fun is -score
                new_score = -result.fun
                new_params = result.x
                if new_score > best_score:
                    best_score = new_score
                    best_path = params_to_path(new_params)
                    best_params_k = new_params
                    best_score_k = new_score
        except Exception:
            pass

    # Phase 3: Local Refinement on the best path (variable length or fixed K?)
    # The current best_path might be variable length (from random search) or fixed K (from optimizer).
    # Let's assume it's fixed K if we just updated it.
    # Let's do a coordinate ascent on the fixed-K parameters.
    
    if time.time() < deadline - 50:
        # We can try to refine best_params_k
        if best_params_k is not None:
            current_params = best_params_k.copy()
            current_score = -objective(current_params) # score
            
            # Coordinate ascent: update one parameter at a time.
            # Or small random steps (hill climbing).
            step_size = 0.1
            for i in range(200):
                if time.time() > deadline - 20: break
                
                # Random perturbation
                # Update one dimension at a time for stability?
                # Or all at once?
                # All at once is faster per step.
                perturbation = rng.normal(0, step_size, size=current_params.shape)
                new_params = current_params + perturbation
                
                # Clip to bounds
                new_params[0::3] = np.clip(new_params[0::3], -1.5, 0.5)
                new_params[1::3] = np.clip(new_params[1::3], -0.5, 1.5)
                new_params[2::3] = np.clip(new_params[2::3], 5, 85)
                
                new_score = -objective(new_params)
                
                if new_score > current_score:
                    current_params = new_params
                    current_score = new_score
                    step_size *= 0.9 # Shrink step
                else:
                    step_size *= 0.95 # Shrink step more aggressively if no improvement
                
                if current_score > best_score:
                    best_score = current_score
                    best_path = params_to_path(current_params)
                    best_params_k = current_params
                    best_score_k = current_score

    # Phase 4: Try to see if adding/removing points improves the score.
    # This is complex to implement generically. 
    # Instead, let's run a few more random paths with different K values.
    
    for K_test in [4, 5, 7, 8]:
        if time.time() > deadline - 30: break
        # Generate 10 random paths of size K_test
        for i in range(10):
            p = generate_random_path_k(K_test)
            s = score_path(p)
            if s > best_score:
                best_score = s
                best_path = p
    
    # Final polish: If best_path is not fixed K, try to convert to K=6 and optimize?
    # Or just return best_path.
    
    return best_path
