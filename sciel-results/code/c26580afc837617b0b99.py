"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time


def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # in 2D, volume = area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    cha = _convex_hull_area(points)
    if cha < 1e-12:
        return 0.0
    mta = _min_triangle_area(points)
    return mta / cha


def _radial_to_points(coeffs, n_points=N):
    """
    Convert Fourier coefficients to points on a convex curve.
    coeffs[0] = base radius
    coeffs[1:] = amplitudes for harmonics 1..K
    We use a radial function r(theta) = r0 + sum_{k=1}^{K} a_k * cos(k*theta + phi_k)
    To keep convexity, we limit harmonics and amplitudes.
    """
    r0 = coeffs[0]
    # Number of harmonic pairs
    K = (len(coeffs) - 1) // 2
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.full(n_points, r0)
    for k in range(1, K + 1):
        a_k = coeffs[2 * k - 1]
        # Use a fixed phase for simplicity (or could optimize phase too)
        # Actually let's just use cos and sin separately for more flexibility
        # coeffs[2k-1] for cos(k*theta), coeffs[2k] for sin(k*theta)
        if 2 * k < len(coeffs):
            b_k = coeffs[2 * k]
        else:
            b_k = 0.0
        radii += a_k * np.cos(k * thetas) + b_k * np.sin(k * thetas)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])


def _objective(params):
    """Objective for differential evolution: minimize negative score."""
    points = _radial_to_points(params)
    s = _score(points)
    return -s


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s budget
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Differential evolution on Fourier coefficients
    # Use harmonics up to K=3 (6 coefficients) + base radius = 7 params
    K = 3
    n_params = 1 + 2 * K  # r0, then (a1,b1), (a2,b2), (a3,b3)
    
    # Bounds: r0 in [0.5, 1.5], amplitudes in [-0.3, 0.3]
    bounds = [(0.5, 1.5)]
    for _ in range(K):
        bounds.append((-0.3, 0.3))
        bounds.append((-0.3, 0.3))
    
    # Run differential evolution with time budget
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=True,
            callback=lambda xk, fk: None
        )
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            best_params = result.x
            best_points = _radial_to_points(best_params)
            best_score = _score(best_points)
    except Exception:
        pass
    
    # Strategy 2: Local refinement with Nelder-Mead or similar
    if best_points is not None and time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            # Convert to flat parameter vector for minimization
            from scipy.optimize import minimize
            
            def _flat_objective(flat_params):
                params = flat_params.copy()
                points = _radial_to_points(params)
                return -_score(points)
            
            # Start from best found
            init_params = np.array([best_params[0]] + list(best_params[1:]))
            # Reshape: r0, a1, b1, a2, b2, a3, b3
            flat_init = init_params.copy()
            
            res = minimize(
                _flat_objective,
                flat_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < -best_score:
                new_params = res.x
                new_points = _radial_to_points(new_params)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception:
            pass
    
    # Strategy 3: Try a regular polygon as baseline and compare
    theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])
    reg_score = _score(reg_points)
    
    if best_points is None or reg_score > best_score:
        best_points = reg_points
        best_score = reg_score
    
    # Strategy 4: Random restarts with local search if time permits
    if time.time() < deadline - 30.0:
        import random
        rng = random.Random(12345)
        for restart in range(20):
            if time.time() > deadline - 10.0:
                break
            # Random Fourier coefficients
            params = np.zeros(n_params)
            params[0] = rng.uniform(0.7, 1.3)
            for k in range(1, K + 1):
                params[2 * k - 1] = rng.uniform(-0.2, 0.2)
                params[2 * k] = rng.uniform(-0.2, 0.2)
            
            points = _radial_to_points(params)
            s = _score(points)
            if s > best_score:
                best_points = points
                best_score = s
            
            # Local search
            if time.time() > deadline - 15.0:
                break
            try:
                from scipy.optimize import minimize
                
                def _flat_obj2(flat_params):
                    pts = _radial_to_points(flat_params)
                    return -_score(pts)
                
                res = minimize(
                    _flat_obj2,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if res.fun < -best_score:
                    new_pts = _radial_to_points(res.x)
                    new_s = _score(new_pts)
                    if new_s > best_score:
                        best_points = new_pts
                        best_score = new_s
            except Exception:
                pass
    
    # Final safety: ensure we return something valid
    if best_points is None or len(best_points) != N:
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    return best_points
# EVOLVE-BLOCK-END
