# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

_N = 14


def _grad_score(points):
    """Compute score and gradient of (d_min/d_max)^2 w.r.t. point positions."""
    n = len(points)
    if n < 2:
        return 0.0, np.zeros_like(points)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0, np.zeros_like(points)
    
    score = (d_min / d_max) ** 2
    
    # Gradient: d/dx_i (d_min/d_max)^2
    # = 2*(d_min/d_max) * [d(d_min)/dx_i * (1/d_max) - d_min * d(d_max)/dx_i / (d_max^2)]
    # = 2*(d_min/d_max) * [d(d_min)/dx_i / d_max - d_min * d(d_max)/dx_i / d_max^2]
    
    # For d_min: gradient is (points[i] - points[j]) / d_min where (i,j) is the min pair
    # For d_max: gradient is (points[i] - points[j]) / d_max where (i,j) is the max pair
    
    min_idx = np.argmin(dists)
    i_min, j_min = np.unravel_index(min_idx, (n, n))
    
    max_idx = np.argmax(dists)
    i_max, j_max = np.unravel_index(max_idx, (n, n))
    
    grad = np.zeros_like(points)
    
    # Gradient of d_min
    diff_min = points[i_min] - points[j_min]
    grad[i_min] += diff_min / d_min
    grad[j_min] -= diff_min / d_min
    
    # Gradient of d_max
    diff_max = points[i_max] - points[j_max]
    grad[i_max] += diff_max / d_max
    grad[j_max] -= diff_max / d_max
    
    # Combine: d(score)/dx = 2*(d_min/d_max) * [grad_dmin/d_max - d_min*grad_dmax/d_max^2]
    factor = 2.0 * (d_min / d_max)
    grad = factor * (grad * (1.0/d_max - d_min/d_max**2))
    
    # Wait, that's not right. Let me redo:
    # score = (d_min/d_max)^2
    # d(score)/d(d_min) = 2*d_min/d_max^2
    # d(score)/d(d_max) = -2*d_min^2/d_max^3
    # 
    # d(d_min)/dx_i = (x_i - x_j)/d_min for the min pair (i,j)
    # d(d_max)/dx_i = (x_i - x_j)/d_max for the max pair (i,j)
    
    grad = np.zeros_like(points)
    
    # Contribution from d_min
    grad[i_min] += (2.0 * d_min / d_max**2) * diff_min / d_min
    grad[j_min] -= (2.0 * d_min / d_max**2) * diff_min / d_min
    
    # Contribution from d_max
    grad[i_max] += (-2.0 * d_min**2 / d_max**3) * diff_max / d_max
    grad[j_max] -= (-2.0 * d_min**2 / d_max**3) * diff_max / d_max
    
    return score, grad


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _spherical_to_cartesian(theta_phi, n=_N):
    """Convert (theta, phi) pairs to 3D unit sphere points."""
    thetas = theta_phi[0::2]
    phis = theta_phi[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])


def _neg_score_cartesian(points, n=_N):
    """Objective: negative of score for minimization, with sphere constraint handled by SLSQP."""
    points = points.reshape(n, 3)
    if not np.all(np.isfinite(points)):
        return 1.0
    return -_score(points)


def _neg_score_cartesian_grad(points, n=_N):
    """Objective and gradient for SLSQP."""
    points = points.reshape(n, 3)
    if not np.all(np.isfinite(points)):
        return 1.0, np.zeros_like(points)
    score, grad = _grad_score(points)
    return -score, -grad


def _sphere_constraints(n=_N):
    """Create sphere constraint functions for SLSQP."""
    def constraint(i):
        def c(points):
            points = points.reshape(n, 3)
            return np.sum(points[i] ** 2) - 1.0
        return c
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'eq', 'fun': constraint(i)})
    return constraints


def _soft_min_max_score(theta_phi, n=_N, sigma=0.05):
    """Soft surrogate using log-sum-exp for smooth optimization."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 1.0
    
    d_scaled = dists / sigma
    d_scaled = np.clip(d_scaled, -500, 500)
    
    # Soft min
    max_neg = np.max(-d_scaled)
    soft_min = -sigma * (max_neg + np.log(np.sum(np.exp(-d_scaled - max_neg))))
    
    # Soft max
    max_d = np.max(d_scaled)
    soft_max = sigma * (max_d + np.log(np.sum(np.exp(d_scaled - max_d))))
    
    if soft_max <= 0:
        return 1.0
    
    ratio = soft_min / soft_max
    return -(ratio ** 2)


def _soft_min_max_score_grad(theta_phi, n=_N, sigma=0.05):
    """Soft surrogate with gradient for L-BFGS-B."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0, np.zeros_like(theta_phi)
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 1.0, np.zeros_like(theta_phi)
    
    d_scaled = dists / sigma
    d_scaled = np.clip(d_scaled, -500, 500)
    
    # Soft min
    max_neg = np.max(-d_scaled)
    soft_min = -sigma * (max_neg + np.log(np.sum(np.exp(-d_scaled - max_neg))))
    
    # Soft max
    max_d = np.max(d_scaled)
    soft_max = sigma * (max_d + np.log(np.sum(np.exp(d_scaled - max_d))))
    
    if soft_max <= 0:
        return 1.0, np.zeros_like(theta_phi)
    
    ratio = soft_min / soft_max
    obj = -(ratio ** 2)
    
    # Numerical gradient (simplified - just return zero for now, use L-BFGS-B finite diff)
    return obj, np.zeros_like(theta_phi)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial guess."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    phis = golden_angle * np.arange(n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def _random_spherical(rng, n, bias='uniform'):
    """Generate random spherical coordinates with optional bias."""
    if bias == 'equator':
        thetas = rng.uniform(np.pi/4, 3*np.pi/4, n)
    elif bias == 'polar':
        thetas = np.concatenate([
            rng.uniform(0.1, np.pi/3, n//2),
            rng.uniform(2*np.pi/3, np.pi-0.1, n - n//2)
        ])
    else:
        thetas = rng.uniform(0.05, np.pi - 0.05, n)
    phis = rng.uniform(0, 2 * np.pi, n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def _coordinate_descent_polish(theta_phi, n=_N, max_iterations=500, time_limit=20.0):
    """Polish using coordinate descent on spherical coordinates with adaptive step size."""
    start = time.time()
    x = theta_phi.copy()
    current_score = _score(_spherical_to_cartesian(x, n))
    
    base_deltas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for it in range(max_iterations):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Optimize theta_i
            best_theta = x[2*i]
            best_theta_score = current_score
            for delta in base_deltas:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i] = best_theta + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > best_theta_score:
                        best_theta_score = new_score
                        best_theta = new_x[2*i]
                        improved = True
            x[2*i] = best_theta
            
            # Optimize phi_i
            best_phi = x[2*i+1]
            best_phi_score = current_score
            for delta in base_deltas:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i+1] = best_phi + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > best_phi_score:
                        best_phi_score = new_score
                        best_phi = new_x[2*i+1]
                        improved = True
            x[2*i+1] = best_phi
            
            current_score = best_theta_score
        
        if not improved:
            break
    
    return x


def _slsqp_polish(points, n=_N, time_limit=30.0):
    """Polish using SLSQP with sphere constraints and gradient."""
    start = time.time()
    x0 = points.flatten()
    
    try:
        constraints = _sphere_constraints(n)
        result = minimize(
            _neg_score_cartesian,
            x0,
            args=(n,),
            jac=lambda x, n: _neg_score_cartesian_grad(x, n)[1],
            method='SLSQP',
            constraints=constraints,
            options={
                'maxiter': 2000,
                'ftol': 1e-12,
            }
        )
        if result.success or result.fun < -_score(points):
            return result.x.reshape(n, 3)
    except Exception:
        pass
    
    return points


def _nelder_mead_polish(theta_phi, n=_N, time_limit=10.0):
    """Polish using Nelder-Mead with tight tolerances."""
    start = time.time()
    try:
        result = minimize(
            _neg_score_spherical,
            theta_phi,
            args=(n,),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 5000,
                'xatol': 1e-12,
                'fatol': 1e-12,
            }
        )
        return result.x
    except Exception:
        return theta_phi


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses multi-phase optimization:
    1. Multi-start with soft surrogate (Nelder-Mead on spherical coords)
    2. SLSQP with gradient on Cartesian coords with sphere constraints
    3. Coordinate descent polish
    4. Final SLSQP high-precision polish
    """
    n = _N
    deadline = time.time() + 580.0  # safety margin before 630s kill
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # Fibonacci sphere
    initial_guesses.append(_fibonacci_sphere(n))
    
    # Uniform random
    initial_guesses.append(_random_spherical(rng, n, bias='uniform'))
    
    # Equator-biased
    initial_guesses.append(_random_spherical(rng, n, bias='equator'))
    
    # Polar-biased
    initial_guesses.append(_random_spherical(rng, n, bias='polar'))
    
    # More random starts
    for _ in range(6):
        initial_guesses.append(_random_spherical(rng, n, bias='uniform'))
    
    # Process each initial guess
    for idx, theta_phi in enumerate(initial_guesses):
        if time.time() > deadline - 60:
            break
        
        # Phase 1: Soft surrogate optimization (Nelder-Mead on spherical coords)
        try:
            result_soft = minimize(
                _soft_min_max_score,
                theta_phi,
                args=(n, 0.05),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                }
            )
        except Exception:
            result_soft = type('obj', (), {'x': theta_phi})()
        
        if time.time() > deadline - 50:
            break
        
        # Convert to Cartesian for SLSQP
        points_soft = _spherical_to_cartesian(result_soft.x, n)
        # Normalize to unit sphere
        norms = np.linalg.norm(points_soft, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points_soft = points_soft / norms
        
        # Phase 2: SLSQP with gradient on Cartesian coords
        try:
            points_slqp = _slsqp_polish(points_soft, n, time_limit=25.0)
        except Exception:
            points_slqp = points_soft
        
        if time.time() > deadline - 30:
            break
        
        # Evaluate
        if np.all(np.isfinite(points_slqp)):
            score = _score(points_slqp)
            if score > best_score:
                best_score = score
                best_points = points_slqp.copy()
        
        # Phase 3: Coordinate descent polish (convert back to spherical)
        # Create spherical coords from best Cartesian
        pts = best_points if best_points is not None else points_slqp
        thetas = np.arccos(np.clip(pts[:, 2], -1, 1))
        phis = np.arctan2(pts[:, 1], pts[:, 0])
        theta_phi_cd = np.empty(2 * n)
        theta_phi_cd[0::2] = thetas
        theta_phi_cd[1::2] = phis
        
        polished_x = _coordinate_descent_polish(theta_phi_cd, n, max_iterations=200, time_limit=10.0)
        points_cd = _spherical_to_cartesian(polished_x, n)
        
        if np.all(np.isfinite(points_cd)):
            score_cd = _score(points_cd)
            if score_cd > best_score:
                best_score = score_cd
                best_points = points_cd.copy()
        
        # Phase 4: Final SLSQP high-precision polish
        if time.time() > deadline - 15:
            break
        
        if best_points is not None:
            try:
                final_points = _slsqp_polish(best_points, n, time_limit=15.0)
                if np.all(np.isfinite(final_points)):
                    final_score = _score(final_points)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_points.copy()
            except Exception:
                pass
    
    # If no valid result, return Fibonacci sphere
    if best_points is None:
        theta_phi = _fibonacci_sphere(n)
        best_points = _spherical_to_cartesian(theta_phi, n)
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = rng.randn(n, 3)
    
    # Ensure shape is correct
    if best_points.shape != (n, 3):
        best_points = rng.randn(n, 3)
    
    return best_points


# EVOLVE-BLOCK-END
