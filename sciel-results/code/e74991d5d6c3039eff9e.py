"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

def _score(points):
    """Compute (max pairwise dist / min pairwise dist)^2. Returns inf if any min dist is 0."""
    n = len(points)
    # Compute all pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Exclude diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_d = np.min(dist_sq)
    max_d = np.max(dist_sq)
    if min_d <= 0:
        return np.inf
    return (max_d / min_d) ** 2

def _soft_score(points, temperature=1.0):
    """
    Smooth surrogate: soft-max of log(dist_sq) and soft-min of log(dist_sq).
    Actually, we want to minimize (max/min)^2.
    Let's minimize log(max) - log(min).
    Soft max of x_i is (1/t) * log(sum(exp(t*x_i))).
    Soft min of x_i is -(1/t) * log(sum(exp(-t*x_i))).
    We want to minimize SoftMax(log d) - SoftMin(log d).
    This is differentiable.
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 0.0)
    
    # Avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-12)
    log_d = np.log(dist_sq)
    
    # Soft max
    t = 1.0 / temperature
    m = np.max(log_d)
    soft_max = m + (1.0 / t) * np.log(np.sum(np.exp(t * (log_d - m))))
    
    # Soft min
    m_min = np.min(log_d)
    soft_min = m_min - (1.0 / t) * np.log(np.sum(np.exp(-t * (log_d - m_min))))
    
    # Objective: SoftMax - SoftMin
    return soft_max - soft_min

def _construct_fourier(theta, coeffs):
    """
    Construct points from radial function r(theta) = sum c_k * cos(k*theta).
    coeffs[0] is r0 (mean radius).
    Only even harmonics to enforce central symmetry?
    Wait, central symmetry requires r(theta + pi) = r(theta).
    cos(k*(theta+pi)) = cos(k*theta + k*pi) = (-1)^k cos(k*theta).
    So for symmetry, we need odd harmonics to be 0.
    So we use only even k: 0, 2, 4, ...
    """
    r = np.zeros_like(theta)
    for k, c in enumerate(coeffs):
        if k == 0:
            r += c
        else:
            # k is the harmonic index. If we pass coeffs for k=0, 2, 4...
            # Let's just pass the actual k values.
            pass
    
    # Better: coeffs is a list of (k, c) pairs or just even harmonics.
    # Let's say coeffs[0] is c0, coeffs[1] is c2, coeffs[2] is c4, etc.
    # Harmonic k = 2*i for i >= 1.
    r = coeffs[0]
    for i in range(1, len(coeffs)):
        k = 2 * i
        r += coeffs[i] * np.cos(k * theta)
    return r

def _fourier_to_points(coeffs, n_points=16):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = _construct_fourier(theta, coeffs)
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _fourier_score(coeffs, temperature=1.0):
    points = _fourier_to_points(coeffs)
    return _soft_score(points, temperature)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Initial guess: Circle
    # coeffs: [r0, c2, c4, c6, c8]
    # r0 = 1, others 0 -> Circle
    initial_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try a few random initializations with low-order Fourier
    np.random.seed(42)
    
    candidates = []
    
    # 1. Circle
    candidates.append(initial_coeffs)
    
    # 2. Random perturbations
    for _ in range(20):
        c = initial_coeffs.copy()
        c[1:] += np.random.normal(0, 0.1, size=len(c)-1)
        candidates.append(c)
    
    # 3. Specific shapes (Square-like, Hexagon-like)
    # Square: r(theta) ~ sec(theta) approximation?
    # Let's just use random search + optimization.
    
    for i, c0 in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Optimize
            res = minimize(
                _fourier_score,
                c0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if time.time() > deadline - 5.0:
                break
                
            c_opt = res.x
            points = _fourier_to_points(c_opt)
            
            # Check distinctness
            if len(set(map(tuple, np.round(points, 10)))) < N:
                continue
                
            score = _score(points)
            
            if score < best_score:
                best_score = score
                best_points = points
                
        except Exception as e:
            pass
    
    # If Fourier approach failed or didn't find anything good, fallback to grid or random
    if best_points is None:
        # Simple random search
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            points = np.random.uniform(-1, 1, (N, 2))
            score = _score(points)
            if score < best_score:
                best_score = score
                best_points = points
        
        if best_points is None:
            # Fallback to grid
            side = int(np.ceil(N ** (1 / D)))
            grid = list(itertools.product(range(side), repeat=D))[:N]
            best_points = np.array(grid, dtype=float)
            
    # Final polish: Local optimization on the actual points
    if time.time() < deadline - 10.0:
        def _point_score(pts):
            return _soft_score(pts, temperature=0.1)
        
        def _point_grad(pts):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(pts)
            for i in range(N):
                for j in range(2):
                    pts_plus = pts.copy()
                    pts_plus[i, j] += eps
                    pts_minus = pts.copy()
                    pts_minus[i, j] -= eps
                    g[i, j] = (_point_score(pts_plus) - _point_score(pts_minus)) / (2 * eps)
            return g
        
        try:
            res = minimize(
                _point_score,
                best_points,
                jac=_point_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.success:
                pts_opt = res.x
                score_opt = _score(pts_opt)
                if score_opt < best_score:
                    best_points = pts_opt
                    best_score = score_opt
        except:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
