"""Symmetry-reduced lattice optimization for logarithmic energy minimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and its gradient w.r.t. pts.
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient of -log(d_ij) w.r.t. x_i = -(x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist2 = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: for each pair (i,j), i<j:
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # d/dx_j [-log(d_ij)] = +(x_i - x_j) / d_ij^2
    grad = np.zeros_like(pts)
    diff_upper = diff[iu]  # (n(n-1)/2, 3)
    d2_upper = dist2[iu]  # (n(n-1)/2,)
    coeff = 1.0 / d2_upper  # (n(n-1)/2,)
    
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad, ii, -coeff[:, np.newaxis] * diff_upper)
    np.add.at(grad, jj, coeff[:, np.newaxis] * diff_upper)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Handle zero norms safely
    norms[norms == 0] = 1.0
    return pts / norms

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)

def _hilbert_curve_3d(n_samples):
    """Generate a 3D space-filling curve using a recursive Hilbert-like construction.
    Returns n_samples points in [0,1]^3.
    """
    # Use a simpler approach: Van der Corput sequence in 3D
    def van_der_corput(n, base):
        result = 0.0
        f = 1.0 / base
        while n > 0:
            result += f * (n % base)
            n //= base
            f /= base
        return result
    
    indices = np.arange(n_samples)
    x = np.array([van_der_corput(i, 2) for i in indices])
    y = np.array([van_der_corput(i, 3) for i in indices])
    z = np.array([van_der_corput(i, 5) for i in indices])
    return np.column_stack([x, y, z])

def _curve_to_sphere(params, n_points):
    """
    Map curve parameters to points on the sphere.
    params: array of shape (12,)
      - params[0:3]: rotation Euler angles (rx, ry, rz)
      - params[3:6]: scale factors for x, y, z of the base curve
      - params[6:9]: phase shifts for x, y, z
      - params[9:12]: local deformation amplitudes (ax, ay, az)
    """
    rx, ry, rz = params[0], params[1], params[2]
    sx, sy, sz = params[3], params[4], params[5]
    px, py, pz = params[6], params[7], params[8]
    ax, ay, az = params[9], params[10], params[11]
    
    # Generate base curve points in [0,1]^3
    base = _hilbert_curve_3d(n_points)
    
    # Apply phase shifts and scales
    t = np.arange(n_points, dtype=float) / n_points
    x = sx * base[:, 0] + px * np.sin(2 * np.pi * t)
    y = sy * base[:, 1] + py * np.sin(2 * np.pi * t)
    z = sz * base[:, 2] + pz * np.sin(2 * np.pi * t)
    
    # Apply local deformations
    x += ax * np.sin(4 * np.pi * t)
    y += ay * np.sin(4 * np.pi * t)
    z += az * np.sin(4 * np.pi * t)
    
    pts = np.column_stack([x, y, z])
    
    # Center and normalize to unit sphere
    center = np.mean(pts, axis=0)
    pts = pts - center
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    # Apply rotation
    rot = Rotation.from_euler('xyz', [rx, ry, rz])
    pts = rot.apply(pts)
    
    return _project_to_sphere(pts)

def _curve_energy_and_grad(params, n_points):
    """Compute energy and gradient w.r.t. curve parameters."""
    pts = _curve_to_sphere(params, n_points)
    energy, grad_pts = _energy_and_grad(pts)
    
    # Numerical gradient w.r.t. params
    grad = np.zeros(len(params))
    eps = 1e-7
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        e_plus = _energy_and_grad(_curve_to_sphere(params_plus, n_points))[0]
        grad[i] = (e_plus - energy) / eps
    
    return energy, grad

def _optimize_curve(deadline):
    """Optimize curve parameters to minimize energy."""
    # Initial guess: identity rotation, unit scales, zero phases, small deformations
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    
    def obj(x):
        return _curve_energy_and_grad(x, N)[0]
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-6})
        best_params = res.x
        pts = _curve_to_sphere(best_params, N)
        return pts
    except Exception:
        return _fibonacci_spiral(N)

def _lbfgs_optimize(pts, deadline):
    """Optimize using L-BFGS-B with sphere constraint via normalization."""
    n = pts.shape[0]
    
    def energy_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(energy_flat, x0, jac=grad_flat, 
                      method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-12})
        p_opt = _project_to_sphere(res.x.reshape(n, 3))
        e_opt = _energy_and_grad(p_opt)[0]
        return p_opt, e_opt
    except Exception:
        return pts, _energy_and_grad(pts)[0]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Step 1: Curve optimization to get a structured initial guess
    pts = _optimize_curve(deadline)
    
    # Ensure pts is valid
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        pts = _fibonacci_spiral(N)
        
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Step 2: Refine with L-BFGS on full point coordinates
    if time.time() < deadline - 10.0:
        p_lbfgs, e_lbfgs = _lbfgs_optimize(pts, deadline)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = p_lbfgs
    
    # Step 3: Try a few random restarts with L-BFGS if time permits
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > deadline - 15.0:
            break
            
        p = rng.standard_normal((N, 3))
        p = _project_to_sphere(p)
        
        p_lbfgs, e_lbfgs = _lbfgs_optimize(p, deadline)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = p_lbfgs

    # Final refinement
    if time.time() < deadline - 2.0:
        p_final, e_final = _lbfgs_optimize(best_pts, deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = p_final

    return _project_to_sphere(best_pts)
