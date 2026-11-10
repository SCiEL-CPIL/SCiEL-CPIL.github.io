# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-stage optimization with adaptive beta annealing and
    high-precision final polish.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_objective(points, beta=50.0):
        """
        Smooth surrogate: maximize soft_min(dists) / max(dists)
        """
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        min_d = dists.min()
        shifted = -beta * (dists - min_d)
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = min_d - log_sum / beta
        
        if not np.isfinite(soft_min) or soft_min <= 0:
            return 0.0
        
        ratio = soft_min / d_max
        if not np.isfinite(ratio):
            return 0.0
        return ratio * ratio

    def neg_smooth(x, beta=50.0):
        """Negative smooth objective for minimization."""
        points = x.reshape(n, d)
        return -smooth_objective(points, beta)

    def neg_objective(x):
        """Negative exact objective for minimization."""
        points = x.reshape(n, d)
        return -objective(points)

    # Initialization strategies
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def static_solution():
        pts = np.array([
            [ 0.000000,  0.000000,  1.000000],
            [ 0.000000, -1.000000,  0.000000],
            [ 0.000000,  1.000000,  0.000000],
            [ 1.000000,  0.000000,  0.000000],
            [-1.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -1.000000],
            [ 0.577350,  0.577350,  0.577350],
            [ 0.577350, -0.577350, -0.577350],
            [-0.577350,  0.577350, -0.577350],
            [-0.577350, -0.577350,  0.577350],
            [ 0.816497,  0.408248, -0.408248],
            [-0.816497,  0.408248,  0.408248],
            [ 0.408248, -0.816497,  0.408248],
            [-0.408248, -0.816497, -0.408248],
        ])
        return pts
    
    def parametric_circle(n_points, seed=0):
        """Parametric approach: points on a circle with radial perturbation."""
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        # Add small angular jitter
        angles += rng.uniform(-0.1, 0.1, n_points)
        # Radial perturbation
        radii = 1.0 + rng.uniform(-0.1, 0.1, n_points)
        # Z-coordinate perturbation
        z_vals = rng.uniform(-0.2, 0.2, n_points)
        
        points = np.zeros((n_points, 3))
        points[:, 0] = radii * np.cos(angles)
        points[:, 1] = radii * np.sin(angles)
        points[:, 2] = z_vals
        return points

    # Phase 1: Differential evolution for global optimization
    if not check_deadline():
        try:
            bounds = [(-1.5, 1.5)] * (n * d)
            de_result = differential_evolution(
                neg_smooth,
                bounds,
                args=(20.0,),
                maxiter=150,
                popsize=20,
                seed=42,
                tol=1e-7,
                polish=False
            )
            
            if de_result.success or not check_deadline():
                de_points = de_result.x.reshape(n, d)
                de_score = objective(de_points)
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Multi-start basin hopping + refinement
    init_strategies = []
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    init_strategies.append(("static", static_solution()))
    
    for seed in range(15):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    for seed in range(5):
        init_strategies.append((f"param_circle_{seed}", parametric_circle(n, seed)))
    
    # Add best DE result if found
    if best_points is not None:
        init_strategies.append(("de_best", best_points.copy()))
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Basin hopping with smooth surrogate
            result = basinhopping(
                lambda x: neg_smooth(x, beta=30.0),
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 500,
                        'xatol': 1e-9,
                        'fatol': 1e-11
                    }
                },
                niter=25,
                stepsize=0.8,
                T=100.0,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Adaptive beta annealing with increasing precision
            x_current = result.x
            
            # Extended beta schedule for better convergence
            beta_schedule = [50.0, 100.0, 200.0, 300.0, 500.0, 800.0, 1200.0]
            
            for beta in beta_schedule:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_current,
                    method='Powell',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                x_current = res.x
            
            if check_deadline():
                break
            
            # High-precision SLSQP polish with tighter tolerances
            res_slsqp = minimize(
                lambda x: neg_smooth(x, beta=1200.0),
                x_current,
                method='SLSQP',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if check_deadline():
                break
            
            # Final exact-objective Nelder-Mead polish
            res_final = minimize(
                neg_objective,
                res_slsqp.x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            
            if check_deadline():
                break
            
            optimized_points = res_final.x.reshape(n, d)
            score = objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish on the best result
    if best_points is not None and not check_deadline():
        try:
            x_best = best_points.flatten()
            
            # One more round of high-beta optimization
            for beta in [1500.0, 2000.0]:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_best,
                    method='Powell',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                x_best = res.x
            
            if not check_deadline():
                res_final = minimize(
                    neg_objective,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                
                polished_points = res_final.x.reshape(n, d)
                polished_score = objective(polished_points)
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_points.copy()
                    
        except Exception:
            pass
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    # Final validation
    if not np.isfinite(best_score) or best_score < 0:
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
