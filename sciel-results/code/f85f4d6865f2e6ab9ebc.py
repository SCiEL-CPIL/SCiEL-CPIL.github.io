"""Rewritten: spherical-harmonic density optimization + discrete polish."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
MAX_TIME = 150.0
SAFETY = 5.0  # seconds to reserve

def _energy(points):
    """Log energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    d2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-18))
    iu = np.triu_indices(n, 1)
    return math.fsum(math.log(1.0 / d[iu[0], iu[1]]) for _ in range(len(iu[0])))

def _energy_fast(points):
    """Vectorized energy for optimization (not used for final scoring)."""
    n = len(points)
    d2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-18))
    iu = np.triu_indices(n, 1)
    return -np.sum(np.log(d[iu]))

def _energy_grad(points):
    """Energy and gradient for discrete optimization."""
    n = len(points)
    d2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-18))
    iu = np.triu_indices(n, 1)
    e = -np.sum(np.log(d[iu]))
    # grad: for pair (i,j), dE/dd_ij = -1/d_ij, dd_ij/dx_i = (x_i - x_j)/d_ij
    grad = np.zeros_like(points)
    diff = points[iu[0]][:, None, :] - points[iu[1]][:, None, :]  # (P, 3)
    inv_d = 1.0 / d[iu][:, None]
    # contribution to x_i: -diff/d * (-1/d) = diff/d^2 ... let me redo
    # E = sum -log(d_ij), dE/dx_i = sum_{j>i} (-1/d_ij) * (x_i - x_j)/d_ij = -sum (x_i-x_j)/d_ij^2
    # dE/dx_j = sum_{i<j} (-1/d_ij) * (x_j - x_i)/d_ij = -sum (x_j-x_i)/d_ij^2 = sum (x_i-x_j)/d_ij^2
    contrib = diff * inv_d**2  # (x_i - x_j)/d^2
    np.add.at(grad, iu[0], -contrib)
    np.add.at(grad, iu[1], contrib)
    return e, grad

def _project(points):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Spherical harmonics up to L=4 for density parameterization
def _spherical_harmonics(theta, phi, L):
    """Return array of Y_lm for l=0..L, m=-l..l. theta=polar, phi=azimuthal."""
    from scipy.special import lpmv
    n_terms = (L+1)**2
    Y = np.zeros((len(theta), n_terms))
    idx = 0
    for l in range(L+1):
        for m in range(-l, l+1):
            abs_m = abs(m)
            # Y_lm(theta, phi) = N_lm * P_lm(cos(theta)) * exp(i*m*phi)
            # We'll use real spherical harmonics: combine m and -m
            # For simplicity, use complex and take real/imag parts
            P = lpmv(abs_m, l, np.cos(theta))
            norm = np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-abs_m)/math.factorial(l+abs_m))
            if m == 0:
                Y[:, idx] = norm * P
            elif m > 0:
                # Real part: sqrt(2)*norm*P*cos(m*phi), Imag: sqrt(2)*norm*P*sin(m*phi)
                Y[:, idx] = np.sqrt(2.0)*norm*P*np.cos(m*phi)
                Y[:, idx+1] = np.sqrt(2.0)*norm*P*np.sin(m*phi)
                idx += 2
            else:
                # m < 0: handled by positive m
                pass
            if m >= 0:
                idx += 1
    return Y[:, :idx]

def _density_energy(coeffs, L=4, n_quad=128):
    """Continuous energy functional for density p(x) = (1/4pi)(1 + sum c_k Y_k(x))."""
    # Quadrature points on sphere
    theta = np.linspace(0, np.pi, n_quad)
    phi = np.linspace(0, 2*np.pi, n_quad)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    TH_flat = TH.ravel()
    PH_flat = PH.ravel()
    X = np.column_stack([np.sin(TH_flat)*np.cos(PH_flat), np.sin(TH_flat)*np.sin(PH_flat), np.cos(TH_flat)])
    Y = _spherical_harmonics(TH_flat, PH_flat, L)
    # Density at quadrature points
    p = (1.0/(4*np.pi)) * (1.0 + Y @ coeffs)
    p = np.maximum(p, 1e-10)
    # Normalize density
    w = np.sin(TH_flat) * (np.pi/n_quad) * (2*np.pi/n_quad)
    p_norm = p / np.sum(p * w)
    # Pairwise distances
    n = len(X)
    d2 = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2.0 * X @ X.T
    d = np.sqrt(np.maximum(d2, 1e-15))
    np.fill_diagonal(d, 1.0)
    K = np.log(1.0 / d)
    # Energy = sum_{i,j} K_ij p_i p_j w_i w_j
    E = (p_norm * w) @ K @ (p_norm * w)
    return E

def _sample_from_density(coeffs, L, n_samples, rng):
    """Sample n points from density p(x) on sphere via rejection sampling."""
    # Use many candidate points, accept with probability proportional to p
    n_try = n_samples * 50
    theta = rng.uniform(0, np.pi, n_try)
    phi = rng.uniform(0, 2*np.pi, n_try)
    Y = _spherical_harmonics(theta, phi, L)
    p = (1.0/(4*np.pi)) * (1.0 + Y @ coeffs)
    p = np.maximum(p, 0.0)
    p_max = np.max(p)
    if p_max < 1e-15:
        return _fibonacci_spiral(n_samples)
    accept = rng.uniform(0, 1, n_try) < p / p_max
    accepted = np.column_stack([np.sin(theta[accept])*np.cos(phi[accept]),
                                  np.sin(theta[accept])*np.sin(phi[accept]),
                                  np.cos(theta[accept])])
    if len(accepted) < n_samples:
        # Fill with fibonacci
        remaining = _fibonacci_spiral(n_samples - len(accepted))
        accepted = np.vstack([accepted, remaining])
    return accepted[:n_samples]

def _gradient_descent(points, max_iters, lr, time_limit, start_time):
    """Gradient descent with adaptive learning rate and backtracking."""
    best_e = _energy_fast(points)
    best_pts = points.copy()
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        e, g = _energy_grad(points)
        if e < best_e - 1e-15:
            best_e = e
            best_pts = points.copy()
        # Adaptive learning rate
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10:
            break
        step = lr * g / g_norm
        # Backtracking
        alpha = 1.0
        for _ in range(30):
            new_pts = _project(points - alpha * step)
            new_e = _energy_fast(new_pts)
            if new_e < e - 1e-4 * alpha * np.sum(g * step):
                break
            alpha *= 0.5
        points = _project(points - alpha * step)
        # Decay learning rate
        if it % 50 == 0:
            lr *= 0.95
    return best_pts, best_e

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME - SAFETY
    rng = np.random.default_rng(42)

    # Phase 1: Spherical harmonic density optimization
    L = 4
    n_coeffs = (L+1)**2
    # Initialize coefficients to zero (uniform density)
    coeffs = np.zeros(n_coeffs)
    
    if time.time() < deadline - 30:
        # Optimize density coefficients
        def obj(c):
            return _density_energy(c, L, n_quad=96)
        
        res = minimize(obj, coeffs, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-4})
        coeffs = res.x

    # Sample from optimized density
    points = _sample_from_density(coeffs, L, N, rng)

    # Phase 2: Discrete gradient descent polish
    # Multiple restarts with different initial conditions
    candidates = [points.copy()]
    
    # Add fibonacci spiral as another candidate
    candidates.append(_fibonacci_spiral(N))
    
    # Add perturbed versions
    for _ in range(2):
        perturbed = points + rng.normal(0, 0.1, size=points.shape)
        candidates.append(_project(perturbed))

    best_points = None
    best_energy = np.inf

    for cand in candidates:
        if time.time() > deadline:
            break
        # Initial gradient descent
        pts, e = _gradient_descent(cand, max_iters=200, lr=0.01, time_limit=deadline - time.time() - 5, start_time=start_time)
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()

    # Phase 3: Fine polish on best candidate
    if time.time() < deadline - 10:
        pts, e = _gradient_descent(best_points, max_iters=500, lr=0.005, time_limit=deadline - time.time() - 2, start_time=start_time)
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()

    # Final projection and validation
    best_points = _project(best_points)
    # Ensure no points are too close
    d2 = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2.0 * best_points @ best_points.T
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 0))
    iu = np.triu_indices(N, 1)
    min_d = np.min(d[iu])
    if min_d < 1e-9:
        # Separate close points
        for _ in range(100):
            d2 = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2.0 * best_points @ best_points.T
            np.fill_diagonal(d2, 1.0)
            d = np.sqrt(np.maximum(d2, 0))
            iu = np.triu_indices(N, 1)
            if np.min(d[iu]) > 1e-9:
                break
            # Find closest pair and push apart
            idx = np.argmin(d[iu])
            i, j = iu[0][idx], iu[1][idx]
            direction = best_points[i] - best_points[j]
            direction = direction / (np.linalg.norm(direction) + 1e-15)
            best_points[i] += 1e-6 * direction
            best_points[j] -= 1e-6 * direction
            best_points = _project(best_points)

    return best_points
