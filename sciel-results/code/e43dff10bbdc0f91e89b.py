import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _jittered_polygon(n, radius=1.0, jitter=0.1, seed=42):
    """Generate jittered regular n-gon."""
    rng = np.random.default_rng(seed)
    base = _regular_polygon(n, radius)
    noise = rng.normal(0, jitter, size=(n, 2))
    return base + noise

def _random_convex_position(n, seed=42):
    """Generate n points in convex position using random angles."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 1.0, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _cmaes_optimize(objective, x0, sigma0=0.1, maxiter=500, seed=42, deadline=None):
    """
    Simple CMA-ES implementation for optimization.
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    best_x = x0.copy()
    best_f = objective(x0)
    
    # CMA-ES parameters
    pop_size = int(4 + 3 * np.log(n))
    pop_size = max(pop_size, 10)
    
    # Initial parameters
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    
    # Adaptation rates
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    c_sigma = (mueff + 2) / (n + mueff + 5)
    c_c = (4 - mueff / n) / (n + 4 - mueff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    
    for it in range(maxiter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate population
        ys = []
        xs = []
        fs = []
        
        for _ in range(pop_size):
            z = rng.standard_normal(n)
            # Transform z using B and D
            y = B @ (D * z)
            x = mean + sigma * y
            f = objective(x)
            ys.append(y)
            xs.append(x)
            fs.append(f)
        
        ys = np.array(ys)
        xs = np.array(xs)
        fs = np.array(fs)
        
        # Sort by fitness
        order = np.argsort(fs)
        ys = ys[order]
        xs = xs[order]
        fs = fs[order]
        
        # Update best
        if fs[0] < best_f:
            best_f = fs[0]
            best_x = xs[0].copy()
        
        # Update mean
        mean_new = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
        
        # Update sigma
        y_mean = B.T @ (D * (mean_new - mean) / sigma)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
        h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) / chi_n < 1.4 + 2 / (n + 1))
        if h_sigma:
            p_c = (1 - c_c) * p_c + c_c * (mean_new - mean) / sigma
        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update covariance
        if h_sigma:
            delta_h = (1 - c_c) - c_c
        else:
            delta_h = 0
        rank_1 = np.outer(p_c, p_c)
        rank_mu = np.sum(weights[:, np.newaxis] * np.outer(ys[:mu], ys[:mu]), axis=0)
        C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
        
        # Decompose C
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        D = 1.0 / np.sqrt(eigvals)
        
        mean = mean_new
        
        # Convergence check
        if sigma < 1e-10 or np.max(eigvals) / np.min(eigvals) > 1e14:
            break
    
    return best_x, -best_f

def _nelder_mead_optimize(objective, x0, maxiter=300, xatol=1e-8, fatol=1e-10, deadline=None):
    """
    Nelder-Mead optimization with time guard.
    """
    from scipy.optimize import minimize
    
    def callback(xk):
        if deadline is not None and time.time() > deadline:
            raise KeyboardInterrupt()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True},
                       callback=callback)
        return res.x, -res.fun
    except KeyboardInterrupt:
        return x0, -objective(x0)
    except Exception:
        return x0, -objective(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    # Try multiple initialization strategies
    initializations = []
    
    # 1. Regular polygon
    initializations.append(_regular_polygon(N))
    
    # 2. Jittered regular polygons with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        initializations.append(_jittered_polygon(N, jitter=0.05, seed=seed))
        initializations.append(_jittered_polygon(N, jitter=0.1, seed=seed))
    
    # 3. Random convex position
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044]:
        initializations.append(_random_convex_position(N, seed=seed))
    
    # First pass: Quick evaluation of all initializations
    for init in initializations:
        if time.time() > deadline - 10:
            break
        score = _score(init)
        update_best(init, score)
    
    # Second pass: Local optimization from best initializations
    # Sort initializations by score
    scored_inits = []
    for init in initializations:
        score = _score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Take top initializations for detailed optimization
    top_k = min(8, len(scored_inits))
    for i, (score, init) in enumerate(scored_inits[:top_k]):
        if time.time() > deadline - 15:
            break
        
        # Try CMA-ES with different parameters
        for sigma0 in [0.05, 0.1, 0.2]:
            if time.time() > deadline - 10:
                break
            try:
                opt_points, opt_score = _cmaes_optimize(
                    _negative_score, init, sigma0=sigma0, 
                    maxiter=200, seed=42 + i, deadline=deadline - 5
                )
                update_best(opt_points, opt_score)
            except Exception:
                pass
        
        # Try Nelder-Mead
        if time.time() > deadline - 10:
            break
        try:
            opt_points, opt_score = _nelder_mead_optimize(
                _negative_score, init, maxiter=300, 
                xatol=1e-8, fatol=1e-10, deadline=deadline - 5
            )
            update_best(opt_points, opt_score)
        except Exception:
            pass
    
    # Third pass: Refine the best solution with more iterations
    if best_points is not None and time.time() < deadline - 20:
        # Final refinement with CMA-ES
        try:
            opt_points, opt_score = _cmaes_optimize(
                _negative_score, best_points, sigma0=0.05,
                maxiter=300, seed=999, deadline=deadline - 5
            )
            update_best(opt_points, opt_score)
        except Exception:
            pass
        
        # Final refinement with Nelder-Mead
        if time.time() < deadline - 10:
            try:
                opt_points, opt_score = _nelder_mead_optimize(
                    _negative_score, best_points, maxiter=500,
                    xatol=1e-9, fatol=1e-11, deadline=deadline - 5
                )
                update_best(opt_points, opt_score)
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
