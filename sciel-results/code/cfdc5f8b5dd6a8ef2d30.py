# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def band_objective(x, gamma=0.01, beta=0.05):
        """
        Smooth objective: normalize distances by soft_max, then penalize
        deviation from the ideal band [1-gamma, 1+gamma].
        This is smoother than the ratio objective near the optimum.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        if np.all(dists == 0):
            return 1e10

        # Soft max for normalization
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_max <= 1e-12:
            return 1e10

        # Normalized distances
        norm_dists = dists / soft_max
        
        # Penalize distances outside [1-gamma, 1+gamma]
        # Using smooth hinge: max(0, |x - 1| - gamma)^2
        deviations = np.abs(norm_dists - 1.0)
        excess = np.maximum(deviations - gamma, 0.0)
        penalty = np.sum(excess ** 2)
        
        # Also add a small regularization to keep points well-spread
        center = np.mean(p, axis=0)
        spread = np.mean(np.sum((p - center) ** 2, axis=1))
        
        return penalty + 1e-4 * spread

    def t_objective(x):
        """
        Epigraph-style: minimize t such that all distances are in [1/t, t]
        after scaling. We minimize t^2 with smooth constraints.
        x contains 42 point coords + 1 value for t (index 42).
        """
        p = x[:n*d].reshape((n, d))
        t = x[n*d]
        
        if not np.all(np.isfinite(p)) or not np.isfinite(t):
            return 1e10
        if t < 1e-6:
            return 1e10

        dists = pdist(p, metric='euclidean')
        d_max = np.max(dists)
        d_min = np.min(dists)
        
        if d_max <= 1e-12:
            return 1e10

        # Scale so that d_max = 1
        scale = d_max
        norm_dists = dists / scale
        
        # We want all norm_dists >= 1/t (i.e., d_min >= d_max/t)
        # Penalty for violations: smooth max(0, 1/t - norm_dists)^2
        lower_bound = 1.0 / t
        lower_violations = np.maximum(lower_bound - norm_dists, 0.0)
        lower_penalty = np.sum(lower_violations ** 2)
        
        # Also penalize if t is too large (we want to minimize t)
        t_penalty = t ** 2
        
        return t_penalty + 100.0 * lower_penalty

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max == 0:
            return 1e10
        return (d_max / d_min) ** 2

    # Initial guess: Start with icosahedron + 2 poles, then do a quick pre-optimization
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Add two points to make 14. Poles are a natural choice for symmetry.
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    points_init = points_init - np.mean(points_init, axis=0)
    
    # Quick pre-optimization to get a better starting point
    try:
        res_pre = minimize(
            lambda x: smooth_objective(x, beta=0.1),
            points_init.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res_pre.fun < 1e9:
            points_init = res_pre.x.reshape((n, d))
    except Exception:
        pass
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # Also try Fibonacci sphere as alternative initial
    def fibonacci_sphere(n_points, seed=0):
        rng = np.random.RandomState(seed)
        points = []
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / n_points
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        pts = pts - np.mean(pts, axis=0)
        return pts
    
    fib_init = fibonacci_sphere(n, seed=42)
    try:
        res_fib = minimize(
            lambda x: smooth_objective(x, beta=0.1),
            fib_init.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res_fib.fun < 1e9:
            fib_opt = res_fib.x.reshape((n, d))
            fib_score = get_score(fib_opt)
            if fib_score > best_score:
                best_points = fib_opt
                best_score = fib_score
    except Exception:
        pass
    
    # 1. Global Optimization using Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            de_deadline = start_time + (time_limit * 0.30)
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                return smooth_objective(x, beta=0.1)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=[(-5, 5)] * (n * d), 
                maxiter=50, 
                popsize=15, 
                tol=1e-6, 
                seed=42,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Local Refinement using L-BFGS-B with adaptive beta
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(8):
        perturb = best_points + np.random.randn(n, d) * 0.12
        candidates_to_polish.append(perturb)
    
    # Fibonacci sphere initial guesses
    for s in [1, 2, 3, 4, 5]:
        candidates_to_polish.append(fibonacci_sphere(n, seed=s))
    
    # Add scaled versions
    for scale in [0.8, 1.2]:
        candidates_to_polish.append(best_points * scale)
    
    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # First polish with moderate beta
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Second polish with smaller beta
                    if time.time() - start_time < time_limit - safety_margin:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # 3. Band objective polish - smoother near optimum
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_band = minimize(
                lambda x: band_objective(x, gamma=0.02, beta=0.05),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_band.fun < 1e9:
                band_points = res_band.x.reshape((n, d))
                band_score = get_score(band_points)
                if band_score > best_score:
                    best_points = band_points
                    best_score = band_score
        except Exception:
            pass

    # 4. T-objective (epigraph) polish
    if time.time() - start_time < time_limit - safety_margin:
        try:
            # Initial t estimate from current best
            dists_best = pdist(best_points, metric='euclidean')
            t_init = np.max(dists_best) / np.min(dists_best)
            
            x_t_init = np.concatenate([best_points.ravel(), [t_init]])
            
            res_t = minimize(
                t_objective,
                x_t_init,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * (n * d) + [(1.0, 100.0)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_t.fun < 1e9:
                t_points = res_t.x[:n*d].reshape((n, d))
                t_score = get_score(t_points)
                if t_score > best_score:
                    best_points = t_points
                    best_score = t_score
        except Exception:
            pass

    # 5. Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass
        
        # Additional polish with Powell method
        if time.time() - start_time < time_limit - 20:
            try:
                res_powell = minimize(
                    exact_objective, 
                    best_points.ravel(), 
                    method='Powell', 
                    options={'maxiter': 3000, 'ftol': 1e-16, 'xtol': 1e-14}
                )
                if res_powell.fun < 1e9:
                    powell_points = res_powell.x.reshape((n, d))
                    powell_score = get_score(powell_points)
                    if powell_score > best_score:
                        best_points = powell_points
                        best_score = powell_score
            except Exception:
                pass

    # 6. Final band polish with very tight gamma
    if time.time() - start_time < time_limit - 15:
        try:
            res_band2 = minimize(
                lambda x: band_objective(x, gamma=0.005, beta=0.02),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_band2.fun < 1e9:
                band2_points = res_band2.x.reshape((n, d))
                band2_score = get_score(band2_points)
                if band2_score > best_score:
                    best_points = band2_points
                    best_score = band2_score
        except Exception:
            pass

    # 7. Pairwise gradient refinement: directly adjust points to equalize critical distances
    def pairwise_gradient_refine(pts, n_iter=200, lr_init=0.01):
        """
        Directly adjust points to increase min distances and decrease max distances.
        Uses gradient of the log-ratio objective computed from pairwise distances.
        """
        p = pts.copy()
        lr = lr_init
        
        for it in range(n_iter):
            if time.time() > time_limit - 10:
                break
                
            dists = pdist(p, metric='euclidean')
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_min <= 0 or d_max <= 0:
                break
                
            # Normalize: work in scaled space where d_max = 1
            scale = d_max
            norm_dists = dists / scale
            
            # Compute gradient of log(d_min) - log(d_max) w.r.t. point positions
            # For each pair (i,j), distance d_ij contributes to:
            #   log(d_min) if it's one of the min distances
            #   log(d_max) if it's one of the max distances
            # Gradient of log(d_ij) w.r.t. p_i = (p_i - p_j) / d_ij
            # Gradient of log(d_ij) w.r.t. p_j = (p_j - p_i) / d_ij
            
            # Find which pairs are critical (near min or max)
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Also consider pairs within a small band of min and max for robustness
            min_threshold = d_min * 1.05
            max_threshold = d_max * 0.95
            
            # Build gradient
            grad = np.zeros_like(p)
            
            # Convert pdist index to pair indices
            def pdist_to_pair(idx):
                i = int((np.sqrt(1 + 8 * idx) - 1) / 2)
                j = i + 1 + (idx - i * (i + 1) // 2)
                return i, j
            
            # Process min-distance pairs (we want to INCREASE these)
            min_mask = dists <= min_threshold
            min_indices = np.where(min_mask)[0]
            for idx in min_indices:
                i, j = pdist_to_pair(idx)
                d_ij = dists[idx]
                if d_ij > 1e-12:
                    diff = (p[i] - p[j]) / d_ij
                    # To increase d_ij, move i away from j and j away from i
                    grad[i] += diff
                    grad[j] -= diff
            
            # Process max-distance pairs (we want to DECREASE these)
            max_mask = dists >= max_threshold
            max_indices = np.where(max_mask)[0]
            for idx in max_indices:
                i, j = pdist_to_pair(idx)
                d_ij = dists[idx]
                if d_ij > 1e-12:
                    diff = (p[i] - p[j]) / d_ij
                    # To decrease d_ij, move i toward j and j toward i
                    grad[i] -= diff
                    grad[j] += diff
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            grad /= grad_norm
            
            # Update with adaptive learning rate
            p_new = p + lr * grad
            
            # Check if improvement
            new_score = get_score(p_new)
            if new_score > get_score(p):
                p = p_new
                # Increase learning rate slightly if improving
                lr = min(lr * 1.02, 0.05)
            else:
                # Decrease learning rate
                lr *= 0.7
                if lr < 1e-6:
                    break
        
        return p

    # Apply pairwise gradient refinement
    if time.time() - start_time < time_limit - 15:
        try:
            refined = pairwise_gradient_refine(best_points, n_iter=300, lr_init=0.005)
            refined_score = get_score(refined)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
        except Exception:
            pass

    # 8. Coordinated pairwise adjustment: simultaneously adjust all critical pairs
    def coordinated_pairwise_adjust(pts, n_iter=100):
        """
        For each critical pair, compute the optimal adjustment direction
        and apply a weighted combination.
        """
        p = pts.copy()
        
        for it in range(n_iter):
            if time.time() > time_limit - 10:
                break
                
            dists = pdist(p, metric='euclidean')
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_min <= 0 or d_max <= 0:
                break
            
            # Target: make all distances as close as possible
            # Use the median as a reference
            d_target = np.median(dists)
            
            # Compute adjustment for each pair
            adjustments = np.zeros_like(p)
            
            def pdist_to_pair(idx):
                i = int((np.sqrt(1 + 8 * idx) - 1) / 2)
                j = i + 1 + (idx - i * (i + 1) // 2)
                return i, j
            
            for idx in range(len(dists)):
                d_ij = dists[idx]
                if d_ij < 1e-12:
                    continue
                
                # Weight based on how far from target
                error = d_ij - d_target
                
                i, j = pdist_to_pair(idx)
                diff = p[i] - p[j]
                unit_diff = diff / d_ij
                
                # Move points to reduce the error
                # If d_ij > d_target, move them closer
                # If d_ij < d_target, move them apart
                move = error * unit_diff * 0.01
                adjustments[i] -= move
                adjustments[j] += move
            
            # Apply adjustment
            adj_norm = np.linalg.norm(adjustments)
            if adj_norm < 1e-15:
                break
            adjustments /= adj_norm
            
            p_new = p + adjustments * 0.1
            
            new_score = get_score(p_new)
            old_score = get_score(p)
            
            if new_score > old_score:
                p = p_new
            else:
                # Try smaller step
                p_try = p + adjustments * 0.01
                if get_score(p_try) > old_score:
                    p = p_try
                else:
                    break
        
        return p

    if time.time() - start_time < time_limit - 15:
        try:
            coordinated = coordinated_pairwise_adjust(best_points, n_iter=150)
            coord_score = get_score(coordinated)
            if coord_score > best_score:
                best_points = coordinated
                best_score = coord_score
        except Exception:
            pass

    # 9. Final super-polish with multiple optimization methods
    if time.time() - start_time < time_limit - 10:
        for method_name, method, opts in [
            ('L-BFGS-B', 'L-BFGS-B', {'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}),
            ('Powell', 'Powell', {'maxiter': 3000, 'ftol': 1e-16, 'xtol': 1e-14}),
            ('Nelder-Mead', 'Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14}),
        ]:
            if time.time() > time_limit - 8:
                break
            try:
                res_mp = minimize(
                    exact_objective, 
                    best_points.ravel(), 
                    method=method,
                    options=opts
                )
                if res_mp.fun < 1e9:
                    mp_points = res_mp.x.reshape((n, d))
                    mp_score = get_score(mp_points)
                    if mp_score > best_score:
                        best_points = mp_points
                        best_score = mp_score
            except Exception:
                pass

    # 10. Iterative re-centering and normalization to remove drift
    if time.time() - start_time < time_limit - 8:
        for _ in range(50):
            if time.time() > time_limit - 6:
                break
            try:
                # Center
                center = np.mean(best_points, axis=0)
                centered = best_points - center
                
                # Normalize to unit sphere
                norms = np.linalg.norm(centered, axis=1)
                max_norm = np.max(norms)
                if max_norm > 1e-12:
                    normalized = centered / max_norm
                else:
                    normalized = centered
                
                # Check if this improved the score
                norm_score = get_score(normalized)
                if norm_score > best_score:
                    best_points = normalized
                    best_score = norm_score
                else:
                    # Try just centering
                    center_score = get_score(centered)
                    if center_score > best_score:
                        best_points = centered
                        best_score = center_score
                    else:
                        break
            except Exception:
                break

    # Try a few more random restarts if time allows
    if time.time() - start_time < time_limit - 10:
        for seed in [777, 888, 999, 1000, 1111, 2222, 3333, 4444, 5555]:
            if time.time() - start_time > time_limit - 8:
                break
            try:
                rp = fibonacci_sphere(n, seed=seed)
                res_q = minimize(
                    lambda x: smooth_objective(x, beta=0.03), 
                    rp.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if res_q.fun < 1e9:
                    qp = res_q.x.reshape((n, d))
                    qs = get_score(qp)
                    if qs > best_score:
                        best_points = qp
                        best_score = qs
                        
                        # Quick band polish
                        if time.time() - start_time < time_limit - 8:
                            res_q2 = minimize(
                                lambda x: band_objective(x, gamma=0.01, beta=0.03),
                                qp.ravel(),
                                method='L-BFGS-B',
                                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-11}
                            )
                            if res_q2.fun < 1e9:
                                qp2 = res_q2.x.reshape((n, d))
                                qs2 = get_score(qp2)
                                if qs2 > best_score:
                                    best_points = qp2
                                    best_score = qs2
                                    
                                    # Final exact polish on this candidate
                                    if time.time() - start_time < time_limit - 8:
                                        res_q3 = minimize(
                                            exact_objective,
                                            qp2.ravel(),
                                            method='L-BFGS-B',
                                            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                                        )
                                        if res_q3.fun < 1e9:
                                            qp3 = res_q3.x.reshape((n, d))
                                            qs3 = get_score(qp3)
                                            if qs3 > best_score:
                                                best_points = qp3
                                                best_score = qs3
            except Exception:
                pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
