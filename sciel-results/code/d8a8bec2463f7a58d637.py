import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    n_triples = len(triples)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
            
        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        min_area = float('inf')
        for i, j, k in triples:
            v1x = x[j] - x[i]
            v1y = y[j] - y[i]
            v2x = x[k] - x[i]
            v2y = y[k] - y[i]
            cross = abs(v1x * v2y - v1y * v2x)
            area = 0.5 * cross
            if area < min_area:
                min_area = area

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
            
        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        
        # Compute all triangle areas
        areas = np.empty(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1x = x[j] - x[i]
            v1y = y[j] - y[i]
            v2x = x[k] - x[i]
            v2y = y[k] - y[i]
            cross = abs(v1x * v2y - v1y * v2x)
            areas[idx] = 0.5 * cross

        # Soft-min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_config(coeffs, n_harmonics=3):
        """
        Generate 13 points using a low-frequency Fourier series on the unit circle.
        coeffs: array of shape (2*(n_harmonics+1),) representing real/imag parts
        of Fourier coefficients a_0, b_0, a_1, b_1, ..., a_K, b_K
        The radius at angle theta is: r(theta) = a_0 + sum_{k=1}^{K} (a_k*cos(k*theta) + b_k*sin(k*theta))
        Points are placed at equally spaced angles, with radius modulated by the Fourier series.
        """
        K = n_harmonics
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # Parse coefficients
        a0 = coeffs[0]
        b0 = coeffs[1]
        
        radii = np.full(N, a0)
        for k in range(1, K + 1):
            a_k = coeffs[2*k]
            b_k = coeffs[2*k + 1]
            radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
        
        # Ensure positive radii
        radii = np.maximum(radii, 0.05)
        
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_objective(coeffs, beta=50.0):
        """Objective for Fourier coefficient optimization."""
        points = _fourier_config(coeffs)
        return _soft_min_score(points, beta)

    def _fourier_hard_objective(coeffs):
        """Hard objective for Fourier coefficient optimization."""
        points = _fourier_config(coeffs)
        score, _ = compute_score(points)
        return -score

    def _make_initial_configurations():
        """Create several diverse initial configurations."""
        configs = []
        
        configs.append(_regular_polygon(13, radius=1.0))
        
        hull_pts = _regular_polygon(12, radius=1.0)
        center = np.array([[0.0, 0.0]])
        configs.append(np.vstack([hull_pts, center]))
        
        hull_pts = _regular_polygon(11, radius=1.0)
        interior1 = np.array([[0.3, 0.0]])
        interior2 = np.array([[-0.15, 0.26]])
        configs.append(np.vstack([hull_pts, interior1, interior2]))
        
        hull_pts = _regular_polygon(10, radius=1.0)
        interior = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([hull_pts, interior]))
        
        hull_pts = _regular_polygon(9, radius=1.0)
        interior = _regular_polygon(4, radius=0.35)
        configs.append(np.vstack([hull_pts, interior]))
        
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=0.5)
        configs.append(np.vstack([outer, inner]))
        
        outer = _regular_polygon(8, radius=1.0)
        inner = _regular_polygon(5, radius=0.45)
        configs.append(np.vstack([outer, inner]))
        
        outer = _regular_polygon(9, radius=1.0)
        inner = _regular_polygon(4, radius=0.4)
        configs.append(np.vstack([outer, inner]))
        
        outer = _regular_polygon(10, radius=1.0)
        inner = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([outer, inner]))
        
        outer = _regular_polygon(11, radius=1.0)
        inner = _regular_polygon(2, radius=0.25)
        configs.append(np.vstack([outer, inner]))
        
        return configs

    def _optimize_config(config, start_time, time_limit):
        """Optimize a single configuration using multiple methods."""
        best_local_points = config.copy()
        best_local_score, _ = compute_score(config)
        
        # Phase 1: L-BFGS-B with soft-min surrogate (beta annealing)
        current = config.copy()
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x: _soft_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_local_score:
                    best_local_score = opt_score
                    best_local_points = optimized.copy()
                current = optimized
            except Exception:
                pass
        
        # Phase 2: Powell refinement
        if time.time() > start_time + time_limit - 15:
            return best_local_points, best_local_score
        
        try:
            res = opt.minimize(
                _hard_score_neg,
                best_local_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            pw_points = res.x.reshape(-1, 2)
            pw_score, _ = compute_score(pw_points)
            if pw_score > best_local_score:
                best_local_score = pw_score
                best_local_points = pw_points.copy()
        except Exception:
            pass
        
        # Phase 3: Nelder-Mead refinement with hard score
        if time.time() > start_time + time_limit - 15:
            return best_local_points, best_local_score
        
        try:
            res = opt.minimize(
                _hard_score_neg,
                best_local_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            nm_points = res.x.reshape(-1, 2)
            nm_score, _ = compute_score(nm_points)
            if nm_score > best_local_score:
                best_local_score = nm_score
                best_local_points = nm_points.copy()
        except Exception:
            pass
        
        return best_local_points, best_local_score

    def _fourier_search(start_time, time_limit):
        """
        Use differential evolution to search Fourier coefficient space.
        This enforces global smoothness and avoids local minima of unconstrained optimization.
        """
        n_harmonics = 3
        n_coeffs = 2 * (n_harmonics + 1)  # a0, b0, a1, b1, a2, b2, a3, b3
        
        # Bounds for Fourier coefficients
        bounds = [(-0.5, 0.5)] * n_coeffs
        # a0 should be positive (mean radius)
        bounds[0] = (0.5, 1.5)
        bounds[1] = (-0.5, 0.5)  # b0
        
        # Initial population: mix of regular polygon-like and perturbed configs
        np.random.seed(42)
        
        def _init_fourier(x0):
            """Initialize Fourier coefficients from a point configuration."""
            # Fit a simple Fourier series to the radii of the points
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            xs, ys = x0[:, 0], x0[:, 1]
            radii = np.sqrt(xs**2 + ys**2)
            
            coeffs = np.zeros(n_coeffs)
            coeffs[0] = np.mean(radii)  # a0
            coeffs[1] = 0.0  # b0
            
            for k in range(1, n_harmonics + 1):
                coeffs[2*k] = np.mean(radii * np.cos(k * angles)) / (N/2) if N > 0 else 0
                coeffs[2*k+1] = np.mean(radii * np.sin(k * angles)) / (N/2) if N > 0 else 0
            
            return coeffs
        
        # Create initial configs and convert to Fourier coefficients
        configs = _make_initial_configurations()
        initial_coeffs = []
        for config in configs[:8]:
            try:
                c = _init_fourier(config)
                # Clip to bounds
                c = np.clip(c, [b[0] for b in bounds], [b[1] for b in bounds])
                initial_coeffs.append(c)
            except Exception:
                pass
        
        # Add some random initializations
        for _ in range(4):
            c = np.array([np.random.uniform(0.8, 1.2)] + 
                        [np.random.uniform(-0.3, 0.3) for _ in range(n_coeffs - 1)])
            c = np.clip(c, [b[0] for b in bounds], [b[1] for b in bounds])
            initial_coeffs.append(c)
        
        # Use differential evolution
        best_fourier_points = None
        best_fourier_score = -float('inf')
        
        try:
            # Use a smaller population for speed
            popsize = min(20, len(initial_coeffs) * 2)
            
            result = opt.differential_evolution(
                _fourier_objective,
                bounds,
                args=(50.0,),
                seed=42,
                maxiter=30,
                tol=1e-6,
                popsize=popsize,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            
            if result.success or result.fun < 1e5:
                best_coeffs = result.x
                best_points = _fourier_config(best_coeffs, n_harmonics)
                best_score, _ = compute_score(best_points)
                if best_score > best_fourier_score:
                    best_fourier_score = best_score
                    best_fourier_points = best_points.copy()
                    
        except Exception:
            pass
        
        return best_fourier_points, best_fourier_score

    best_points = None
    best_score = -float('inf')
    
    configs = _make_initial_configurations()
    
    # Optimize each initial configuration
    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 20:
            break
        
        local_points, local_score = _optimize_config(config, start_time, time_limit)
        if local_score > best_score:
            best_score = local_score
            best_points = local_points.copy()
    
    # Fourier-based global search
    if time.time() < start_time + time_limit - 30:
        fourier_points, fourier_score = _fourier_search(start_time, time_limit)
        if fourier_points is not None and fourier_score > best_score:
            best_score = fourier_score
            best_points = fourier_points.copy()
        
        # Refine the Fourier-found solution with standard optimization
        if best_points is not None and time.time() < start_time + time_limit - 20:
            refined_points, refined_score = _optimize_config(best_points, start_time, time_limit)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
    
    # Random restarts with soft-min optimization
    if time.time() < start_time + time_limit - 20:
        np.random.seed(123)
        n_restarts = 6
        for _ in range(n_restarts):
            if time.time() > start_time + time_limit - 20:
                break
            n_outer = np.random.randint(6, 12)
            n_inner = N - n_outer
            outer = _regular_polygon(n_outer, radius=np.random.uniform(0.8, 1.2))
            if n_inner > 0:
                inner = _regular_polygon(n_inner, radius=np.random.uniform(0.2, 0.5))
                config = np.vstack([outer, inner])
            else:
                config = outer
            config = config + np.random.normal(0, 0.1, config.shape)
            
            local_points, local_score = _optimize_config(config, start_time, time_limit)
            if local_score > best_score:
                best_score = local_score
                best_points = local_points.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 300.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
