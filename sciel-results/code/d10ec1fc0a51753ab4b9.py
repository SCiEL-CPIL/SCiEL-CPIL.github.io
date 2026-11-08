import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min: -log(sum(exp(-beta * areas))) / beta
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _exact_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_to_points(coeffs):
        """Convert Fourier coefficients to 13 point positions.
        coeffs = [a1, b1, a2, b2, a3, b3, a4, b4] (8 params)
        r_k = 1 + sum_{m=1}^{4} (a_m * cos(m*θ_k) + b_m * sin(m*θ_k))
        """
        n = len(coeffs) // 2  # number of harmonics
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for m in range(1, n + 1):
            a_m = coeffs[2 * (m - 1)]
            b_m = coeffs[2 * (m - 1) + 1]
            radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        # Ensure positive radii (convexity)
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])
    
    def _fourier_objective(coeffs, beta=100.0):
        """Objective for Fourier parameterization (minimize negative soft-min area)."""
        pts = _fourier_to_points(coeffs)
        return _soft_objective(pts.flatten(), beta)
    
    def _fourier_exact_objective(coeffs):
        """Exact objective for Fourier parameterization."""
        pts = _fourier_to_points(coeffs)
        return _exact_objective(pts.flatten())
    
    def _newton_refine_fourier(coeffs, beta=200.0, max_iter=50, tol=1e-10):
        """
        Newton-Raphson refinement in Fourier space.
        Computes Hessian numerically and applies Newton's method.
        """
        c = coeffs.copy()
        f_val = _fourier_objective(c, beta)
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
                
            # Compute gradient
            grad = np.zeros_like(c)
            eps = 1e-5
            for i in range(len(c)):
                c_plus = c.copy()
                c_plus[i] += eps
                f_plus = _fourier_objective(c_plus, beta)
                c_minus = c.copy()
                c_minus[i] -= eps
                f_minus = _fourier_objective(c_minus, beta)
                c[i] = coeffs[i]  # restore
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            # Check convergence
            if np.max(np.abs(grad)) < tol:
                break
            
            # Compute Hessian
            H = np.zeros((len(c), len(c)))
            eps2 = 1e-4
            for i in range(len(c)):
                for j in range(i, len(c)):
                    c_pp = c.copy()
                    c_pp[i] += eps2
                    c_pp[j] += eps2
                    f_pp = _fourier_objective(c_pp, beta)
                    
                    c_pm = c.copy()
                    c_pm[i] += eps2
                    c_pm[j] -= eps2
                    f_pm = _fourier_objective(c_pm, beta)
                    
                    c_mp = c.copy()
                    c_mp[i] -= eps2
                    c_mp[j] += eps2
                    f_mp = _fourier_objective(c_mp, beta)
                    
                    c_mm = c.copy()
                    c_mm[i] -= eps2
                    c_mm[j] -= eps2
                    f_mm = _fourier_objective(c_mm, beta)
                    
                    h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps2**2)
                    H[i, j] = h_ij
                    H[j, i] = h_ij
                    c[i] = coeffs[i]
                    c[j] = coeffs[j]
            
            # Add small regularization for numerical stability
            H += 1e-6 * np.eye(len(c))
            
            # Solve H * delta = -grad
            try:
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step
                delta = -grad * 0.01
            
            # Line search to ensure decrease
            step = 1.0
            for _ in range(20):
                c_new = c + step * delta
                f_new = _fourier_objective(c_new, beta)
                if f_new < f_val:
                    c = c_new
                    f_val = f_new
                    break
                step *= 0.5
            
            if step < 1e-6:
                break
        
        return c

    def _coordinate_descent(points, max_iter=500, tol=1e-10):
        """
        Coordinate descent to maximize min triangle area / hull area.
        Uses adaptive step sizes and more thorough local search.
        Optimized to reduce ConvexHull calls.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        
        # Adaptive step sizes - start smaller for refinement
        base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score
                
                # Try perturbations with adaptive step sizes
                for step in base_steps:
                    # 8 directions
                    directions = [
                        (step, 0), (-step, 0), (0, step), (0, -step),
                        (step*0.707, step*0.707), (-step*0.707, step*0.707),
                        (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                    ]
                    for dx, dy in directions:
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + tol:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                
                if improved:
                    pts[i] = best_local
                    best_score = best_local_score
            
            if not improved:
                break
        
        return pts, best_score

    # Generate initial configurations
    initializations = []
    
    # 1. Regular 13-gon (all on boundary)
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 on boundary + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Various split configurations: k on outer ring, (13-k) on inner ring
    for k_outer in range(5, 13):
        k_inner = N - k_outer
        outer = _regular_polygon(k_outer, 1.0)
        for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if k_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(k_inner, r_inner)
            config = np.vstack([outer, inner])
            initializations.append(config)
    
    # 4. Concentric polygons with different symmetries
    for r1 in [0.3, 0.4, 0.5]:
        for r2 in [0.6, 0.7, 0.8]:
            center = np.array([[0.0, 0.0]])
            inner_hex = _regular_polygon(6, r1)
            outer_hex = _regular_polygon(6, r2)
            config = np.vstack([center, inner_hex, outer_hex])
            initializations.append(config)
    
    # 5. 13 = 1 + 3 + 9 (center + triangle + nonagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        triangle = _regular_polygon(3, r1)
        nonagon = _regular_polygon(9, 1.0)
        config = np.vstack([center, triangle, nonagon])
        initializations.append(config)
    
    # 6. 13 = 1 + 4 + 8 (center + square + octagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        square = _regular_polygon(4, r1)
        octagon = _regular_polygon(8, 1.0)
        config = np.vstack([center, square, octagon])
        initializations.append(config)
    
    # 7. 13 = 1 + 5 + 7 (center + pentagon + heptagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        pentagon = _regular_polygon(5, r1)
        heptagon = _regular_polygon(7, 1.0)
        config = np.vstack([center, pentagon, heptagon])
        initializations.append(config)
    
    # 8. 13 = 1 + 2 + 10 (center + dipole + decagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        dipole = np.array([[-r1, 0.0], [r1, 0.0]])
        decagon = _regular_polygon(10, 1.0)
        config = np.vstack([center, dipole, decagon])
    
    # Add some random perturbations to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 5):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        initializations.append(perturbed)
    
    # Fourier-based initializations: start from regular polygon and add small deformations
    # Use 4 harmonics (8 parameters)
    fourier_bounds = [(-0.15, 0.15)] * 8
    rng_fourier = np.random.RandomState(123)
    for _ in range(10):
        coeffs = rng_fourier.uniform(-0.1, 0.1, size=8)
        pts = _fourier_to_points(coeffs)
        initializations.append(pts)
    
    # Also try pure harmonic deformations
    for m in range(1, 5):
        for amp in [0.05, 0.1, 0.15]:
            for phase in [0, np.pi/4, np.pi/2]:
                coeffs = np.zeros(8)
                coeffs[2*(m-1)] = amp * np.cos(phase)
                coeffs[2*(m-1)+1] = amp * np.sin(phase)
                pts = _fourier_to_points(coeffs)
                initializations.append(pts)
    
    # Deduplicate and limit initializations
    best_score = -float('inf')
    best_points = None
    seen_scores = set()
    
    # Differential evolution on Fourier parameterization (reduced search space)
    # This helps escape local minima that full 26-D search might get stuck in
    if time.time() < start_time + time_limit - 60:
        try:
            de_result = opt.differential_evolution(
                _fourier_objective,
                bounds=fourier_bounds,
                seed=42,
                maxiter=80,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < 1e5:
                best_coeffs = de_result.x
                de_points = _fourier_to_points(best_coeffs)
                de_score = compute_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                # Add DE result to initializations for further refinement
                initializations.append(de_points)
                
                # Newton refinement on Fourier space
                if time.time() < start_time + time_limit - 30:
                    try:
                        refined_coeffs = _newton_refine_fourier(best_coeffs, beta=200.0, max_iter=30)
                        refined_points = _fourier_to_points(refined_coeffs)
                        refined_score = compute_score(refined_points)
                        if refined_score > best_score:
                            best_score = refined_score
                            best_points = refined_points.copy()
                        initializations.append(refined_points)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Optimize each initialization
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 25:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Quick check: skip if this is a known bad config
        score_key = round(current_score, 6)
        if score_key in seen_scores:
            continue
        seen_scores.add(score_key)
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 1b: Fourier-space optimization with Newton refinement
        if time.time() < start_time + time_limit - 20:
            try:
                # Estimate Fourier coefficients from current points
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                radii = np.sqrt(current[:, 0]**2 + current[:, 1]**2)
                # Fit Fourier series to radii
                A = np.zeros((N, 8))
                for m in range(1, 5):
                    A[:, 2*(m-1)] = np.cos(m * angles)
                    A[:, 2*(m-1)+1] = np.sin(m * angles)
                # Least squares fit
                coeffs_fit, _, _, _ = np.linalg.lstsq(A, radii - 1.0, rcond=None)
                
                # Optimize in Fourier space with L-BFGS-B
                for beta_f in [50.0, 100.0, 200.0]:
                    if time.time() > start_time + time_limit - 18:
                        break
                    try:
                        res_f = opt.minimize(
                            lambda c, b=beta_f: _fourier_objective(c, b),
                            coeffs_fit,
                            method='L-BFGS-B',
                            bounds=fourier_bounds,
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        if res_f.fun < 1e5:
                            f_pts = _fourier_to_points(res_f.x)
                            f_score = compute_score(f_pts)
                            if f_score > compute_score(current):
                                current = f_pts
                    except Exception:
                        pass
                
                # Newton refinement in Fourier space
                if time.time() < start_time + time_limit - 16:
                    try:
                        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                        radii = np.sqrt(current[:, 0]**2 + current[:, 1]**2)
                        A = np.zeros((N, 8))
                        for m in range(1, 5):
                            A[:, 2*(m-1)] = np.cos(m * angles)
                            A[:, 2*(m-1)+1] = np.sin(m * angles)
                        coeffs_fit, _, _, _ = np.linalg.lstsq(A, radii - 1.0, rcond=None)
                        
                        newton_coeffs = _newton_refine_fourier(coeffs_fit, beta=200.0, max_iter=40)
                        newton_pts = _fourier_to_points(newton_coeffs)
                        newton_score = compute_score(newton_pts)
                        if newton_score > compute_score(current):
                            current = newton_pts
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Step 2: Coordinate descent refinement
        if time.time() > start_time + time_limit - 10:
            break
        try:
            current, score_cd = _coordinate_descent(current, max_iter=300)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass
        
        # Step 3: Nelder-Mead with exact objective for final polish
        if time.time() > start_time + time_limit - 8:
            break
        try:
            res = opt.minimize(
                _exact_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 4: Additional coordinate descent pass after Nelder-Mead
        if time.time() > start_time + time_limit - 5:
            break
        try:
            candidate, score_cd2 = _coordinate_descent(candidate, max_iter=150)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 5: High-precision final polish with very small steps
        if time.time() < start_time + time_limit - 2:
            try:
                # Ultra-fine coordinate descent
                fine_steps = [0.0001, 0.00005, 0.00002, 0.00001]
                fine_pts = best_points.copy() if best_points is not None else current.copy()
                fine_score = compute_score(fine_pts)
                for step in fine_steps:
                    for i in range(N):
                        directions = [
                            (step, 0), (-step, 0), (0, step), (0, -step),
                            (step*0.707, step*0.707), (-step*0.707, step*0.707),
                            (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                        ]
                        for dx, dy in directions:
                            new_pt = fine_pts.copy()
                            new_pt[i] = fine_pts[i] + np.array([dx, dy])
                            score = compute_score(new_pt)
                            if score > fine_score + 1e-12:
                                fine_score = score
                                fine_pts[i] = new_pt[i]
                if fine_score > best_score:
                    best_score = fine_score
                    best_points = fine_pts.copy()
            except Exception:
                pass
    
    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]
    
    return best_points
