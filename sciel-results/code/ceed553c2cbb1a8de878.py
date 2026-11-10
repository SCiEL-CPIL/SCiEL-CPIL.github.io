import numpy as np
import time
import scipy.spatial
import scipy.optimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize the ratio 
    of the smallest triangle area to the convex hull area.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Seconds
    
    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]
    n_tris = len(tri_indices)
    
    # Map each point index to the list of triangle indices it belongs to
    # point_to_tris[p] is a list of indices in tri_indices where p is involved
    point_to_tris = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(tri_indices):
        point_to_tris[i].append(t_idx)
        point_to_tris[j].append(t_idx)
        point_to_tris[k].append(t_idx)

    def calculate_exact_score(points):
        """Calculate the true score: A_min / A_hull"""
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        # Signed area * 2
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0
        return A_min / A_hull

    def soft_min_objective_and_grad(flat_points, k):
        """
        Objective: -SoftMin(areas) / A_hull
        Gradient: d/dp (-SoftMin(areas) / A_hull)
        
        Note: We maximize A_min/A_hull.
        Let f = A_min / A_hull. We minimize -f.
        SoftMin approximates A_min.
        """
        points = flat_points.reshape(n, 2)
        
        # 1. Calculate Convex Hull Area
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 1e10, np.zeros_like(flat_points) # Penalize invalid
        except Exception:
            return 1e10, np.zeros_like(flat_points)
            
        # 2. Calculate Triangle Areas
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        # Signed area * 2
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        signed_areas = 0.5 * cross
        abs_areas = np.abs(signed_areas)
        
        # 3. Soft Min Calculation
        # To avoid overflow in exp, shift by max
        # We want -1/k * log(sum(exp(-k * A_i)))
        # Let A_i be the areas.
        # If k is large, exp(-k * A_i) underflows for large A_i.
        # We use the log-sum-exp trick.
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        # Here x_i = -k * A_i.
        
        # If any area is 0, SoftMin is 0.
        if np.any(abs_areas < 1e-10):
            # Degenerate
            return 1e10, np.zeros_like(flat_points)

        # Weights for SoftMin
        # w_i = exp(-k * A_i) / sum(exp(-k * A_m))
        # To compute stably:
        max_neg_k_A = np.max(-k * abs_areas)
        exp_terms = np.exp(-k * abs_areas - max_neg_k_A)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp < 1e-100:
             # Numerical issue, fallback
             return 1e10, np.zeros_like(flat_points)

        weights = exp_terms / sum_exp
        
        # SoftMin Value
        # -1/k * log(sum(exp(-k A))) = -1/k * (log(sum_exp) + max_neg_k_A)
        # = -1/k * log(sum_exp) - max_neg_k_A/k
        # = -1/k * log(sum_exp) + min(A) approx
        # Let's just use the formula:
        soft_min_val = -1.0/k * (np.log(sum_exp) + max_neg_k_A)
        
        # Objective to Minimize: - (soft_min_val / A_hull)
        val = - (soft_min_val / A_hull)
        
        # 4. Gradient Calculation
        # grad_val = - (1/A_hull) * grad(soft_min_val)
        # grad(soft_min_val) = sum_i (d soft_min/d A_i) * (d A_i / d p)
        # d soft_min/d A_i = - weights[i]
        # So grad(soft_min_val) = - sum_i weights[i] * grad(A_i)
        
        grad_soft_min = np.zeros((n, 2))
        
        # For each triangle, add contribution to its 3 vertices
        # d A_i / d p_j for vertex j of triangle (i, j, k)
        # A = 0.5 * | (p_i - p_j) x (p_k - p_j) |
        # Let S = signed area. A = |S|. dA/dp = sign(S) * dS/dp.
        
        # dS/dp_i = 0.5 * (p_k - p_j) rotated 90 deg?
        # S = 0.5 * ( (x_i-x_j)(y_k-y_j) - (y_i-y_j)(x_k-x_j) )
        # dS/dx_i = 0.5 * (y_k - y_j)
        # dS/dy_i = -0.5 * (x_k - x_j)
        # So vector dS/dp_i = 0.5 * (y_k - y_j, -(x_k - x_j)) = 0.5 * (p_k - p_j)_rot90
        
        # Let's compute gradients for all triangles vectorized
        # For triangle t with points i, j, k:
        # grad_p_i = 0.5 * sign(S_t) * (p_k - p_j)_rot90
        # grad_p_j = 0.5 * sign(S_t) * (p_i - p_k)_rot90
        # grad_p_k = 0.5 * sign(S_t) * (p_j - p_i)_rot90
        
        signs = np.sign(signed_areas)
        # If signed area is 0, gradient is undefined, but we checked abs_areas > 1e-10
        
        # Weights for this triangle
        w = weights * signs * 0.5
        
        # Accumulate gradients
        # For vertex i_idx:
        # diff_kj = pk - pj
        # rot90(diff) = (dy, -dx)
        diff_kj = pk - pj
        grad_i = w[:, None] * np.column_stack([diff_kj[:, 1], -diff_kj[:, 0]])
        
        diff_ik = pi - pk
        grad_j = w[:, None] * np.column_stack([diff_ik[:, 1], -diff_ik[:, 0]])
        
        diff_ji = pj - pi
        grad_k = w[:, None] * np.column_stack([diff_ji[:, 1], -diff_ji[:, 0]])
        
        # Scatter add to grad_soft_min
        # np.add.at is efficient for this
        np.add.at(grad_soft_min, i_idx, grad_i)
        np.add.at(grad_soft_min, j_idx, grad_j)
        np.add.at(grad_soft_min, k_idx, grad_k)
        
        # Total gradient of objective
        # obj = - soft_min / A_hull
        # grad_obj = - grad_soft_min / A_hull
        grad_obj = - grad_soft_min / A_hull
        
        return val, grad_obj.flatten()

    def run_optimization(initial_guess, time_remaining):
        """
        Run multi-stage optimization starting from initial_guess.
        Returns best points and score.
        """
        # Normalize initial guess
        p = initial_guess - np.mean(initial_guess, axis=0)
        max_dist = np.max(np.linalg.norm(p, axis=1))
        if max_dist > 1e-10:
            p = p / max_dist
        
        best_p = p.copy()
        best_score = calculate_exact_score(p)
        
        # Stages of k (sharpness)
        k_stages = [10, 50, 100, 200, 500, 1000]
        
        current_p = p.copy()
        
        for k in k_stages:
            if time.time() - start_time > time_limit - 10:
                break
                
            # Define function for L-BFGS-B
            def func_and_grad(x):
                val, grad = soft_min_objective_and_grad(x, k)
                return val, grad
            
            try:
                result = scipy.optimize.minimize(
                    func_and_grad,
                    current_p.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if result.success or result.fun < -best_score: # -best_score is the target for negative objective
                    new_p = result.x.reshape(n, 2)
                    new_score = calculate_exact_score(new_p)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_p = new_p
                        current_p = new_p
                    else:
                        # Even if exact score didn't improve, the soft-min might have improved the landscape
                        # We should probably keep current_p if it's better in soft-min sense, 
                        # but for tracking best, we use exact score.
                        # However, for the next stage, we should start from the result of the optimization
                        # even if exact score dipped slightly due to approximation, as long as it's valid.
                        # Let's use the result if it's valid.
                        if result.fun < 1e9: # Valid
                            current_p = new_p
            except Exception:
                pass

        return best_p, best_score

    # Generate Initial Guesses
    initial_guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)
    
    # 2. Jittered Regular Polygon
    rng = np.random.default_rng(42)
    for _ in range(20):
        jitter = rng.normal(0, 0.1, (n, 2))
        initial_guesses.append(reg_poly + jitter)
        
    # 3. Random points in unit circle
    for _ in range(20):
        r = np.sqrt(rng.random(n))
        theta = 2 * np.pi * rng.random(n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_guesses.append(pts)
        
    # 4. Grid-based
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    for _ in range(10):
        idx = rng.choice(16, 13, replace=False)
        initial_guesses.append(grid_points[idx])

    best_overall_score = 0.0
    best_overall_points = initial_guesses[0]
    
    # Optimization Loop
    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 20:
            break
            
        p, score = run_optimization(guess, time_limit - (time.time() - start_time))
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = p

    # Final Polish on the best point found
    # Use a very high k to approximate the true min with high precision
    if time.time() - start_time < time_limit - 5:
        def final_func_and_grad(x):
            return soft_min_objective_and_grad(x, 5000)
            
        try:
            result = scipy.optimize.minimize(
                final_func_and_grad,
                best_overall_points.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            final_p = result.x.reshape(n, 2)
            final_score = calculate_exact_score(final_p)
            
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_points = final_p
        except Exception:
            pass

    return best_overall_points
