"""Rewritten program exploiting scale invariance via spherical parameterization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import factorial
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 100
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials (physicists' convention) using recurrence
def _hermite(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(1, n):
        H2 = 2 * x * H1 - 2 * k * H0
        H0 = H1
        H1 = H2
    return H1

# Compute H0, H4, H8, H12
_H0 = _hermite(0, _x_gh)
_H4 = _hermite(4, _x_gh)
_H8 = _hermite(8, _x_gh)
_H12 = _hermite(12, _x_gh)

# Values at x=0
_H0_0 = 1.0
_H4_0 = _hermite(4, np.array([0.0]))[0]
_H8_0 = _hermite(8, np.array([0.0]))[0]
_H12_0 = _hermite(12, np.array([0.0]))[0]

# Derivatives of Hermite polynomials: d/dx H_n = 2n H_{n-1}
_dH0 = np.zeros_like(_x_gh)
_dH4 = 8 * _hermite(3, _x_gh)
_dH8 = 16 * _hermite(7, _x_gh)
_dH12 = 24 * _hermite(11, _x_gh)

# Stack all Hermite polynomials
_H_all = np.column_stack([_H0, _H4, _H8, _H12])
_dH_all = np.column_stack([_dH0, _dH4, _dH8, _dH12])
_x2 = _x_gh ** 2

# Build Gram matrices
# G[i,j] = ∫ H_i H_j e^{-x^2} dx
_G = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _G[i, j] = np.sum(_w_gh * _H_all[:, i] * _H_all[:, j])

# M[i,j] = ∫ x^2 H_i H_j e^{-x^2} dx
_M = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _M[i, j] = np.sum(_w_gh * _x2 * _H_all[:, i] * _H_all[:, j])

# D[i,j] = ∫ dH_i dH_j e^{-x^2} dx
_D = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _D[i, j] = np.sum(_w_gh * _dH_all[:, i] * _dH_all[:, j])

# E[i,j] = ∫ dH_i x H_j e^{-x^2} dx
_E = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _E[i, j] = np.sum(_w_gh * _dH_all[:, i] * _x_gh * _H_all[:, j])


def _coeffs_from_angles(theta, phi):
    """Convert spherical angles to 3D direction vector."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def _compute_bound_from_direction(d):
    """
    Compute the uncertainty product for a unit direction vector d = (d0, d4, d8).
    The constraint P(0) = 0 fixes a12 = -(d0*H0_0 + d4*H4_0 + d8*H8_0) / H12_0.
    Since the problem is scale-invariant, we can use unit vectors.
    """
    d0, d4, d8 = d[0], d[1], d[2]
    a12 = -(d0 * _H0_0 + d4 * _H4_0 + d8 * _H8_0) / _H12_0
    a = np.array([d0, d4, d8, a12])
    
    norm2 = a @ _G @ a
    if norm2 < 1e-30:
        return 1e10
    
    mom2 = a @ _M @ a
    var_x = mom2 / norm2
    
    momentum2 = a @ _D @ a - 2.0 * (a @ _E @ a) + mom2
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))


def _objective_spherical(theta, phi):
    """Objective function in spherical coordinates."""
    d = _coeffs_from_angles(theta, phi)
    return _compute_bound_from_direction(d)


def _fibonacci_sphere(n_points):
    """Generate n_points on the unit sphere using Fibonacci lattice."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        z = 1 - (2 * i + 1) / n_points
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)


def _local_refine(d0, max_iter=200):
    """Local refinement using gradient descent on the sphere."""
    best_d = d0.copy()
    best_val = _compute_bound_from_direction(best_d)
    
    # Convert to spherical coordinates
    if best_d[2] >= 1.0:
        theta = 0.0
        phi = 0.0
    elif best_d[2] <= -1.0:
        theta = np.pi
        phi = 0.0
    else:
        theta = np.arccos(np.clip(best_d[2], -1.0, 1.0))
        phi = np.arctan2(best_d[1], best_d[0])
    
    lr = 0.1
    for it in range(max_iter):
        # Compute gradient numerically
        eps = 1e-6
        f_theta_plus = _objective_spherical(theta + eps, phi)
        f_theta_minus = _objective_spherical(theta - eps, phi)
        f_phi_plus = _objective_spherical(theta, phi + eps)
        f_phi_minus = _objective_spherical(theta, phi - eps)
        
        grad_theta = (f_theta_plus - f_theta_minus) / (2 * eps)
        grad_phi = (f_phi_plus - f_phi_minus) / (2 * eps)
        
        theta -= lr * grad_theta
        phi -= lr * grad_phi
        
        # Update best
        new_d = _coeffs_from_angles(theta, phi)
        new_val = _compute_bound_from_direction(new_d)
        
        if new_val < best_val:
            best_val = new_val
            best_d = new_d.copy()
            lr = min(lr * 1.1, 0.5)
        else:
            lr = max(lr * 0.9, 1e-4)
            # Revert
            if best_d[2] >= 1.0:
                theta = 0.0
            elif best_d[2] <= -1.0:
                theta = np.pi
            else:
                theta = np.arccos(np.clip(best_d[2], -1.0, 1.0))
                phi = np.arctan2(best_d[1], best_d[0])
        
        if lr < 1e-6:
            break
    
    return best_d, best_val


def construct():
    start_time = time.time()
    deadline = 210.0
    safety_margin = 10.0
    
    # Phase 1: Fibonacci lattice search
    n_points = 5000
    lattice = _fibonacci_sphere(n_points)
    
    best_d = None
    best_val = float('inf')
    
    for i in range(n_points):
        if time.time() > deadline - safety_margin - 30:
            break
        d = lattice[i]
        val = _compute_bound_from_direction(d)
        if val < best_val:
            best_val = val
            best_d = d.copy()
    
    # Phase 2: Local refinement from best lattice point
    if best_d is not None and time.time() < deadline - safety_margin - 20:
        best_d, best_val = _local_refine(best_d, max_iter=500)
    
    # Phase 3: Multi-start refinement from diverse points
    if time.time() < deadline - safety_margin - 15:
        # Generate additional diverse starting points
        rng = np.random.default_rng(123)
        n_starts = 20
        for _ in range(n_starts):
            if time.time() > deadline - safety_margin - 10:
                break
            # Random point on sphere
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            
            d, val = _local_refine(v, max_iter=300)
            if val < best_val:
                best_val = val
                best_d = d.copy()
    
    # Phase 4: Final fine-tuning with BFGS in Cartesian space
    if time.time() < deadline - safety_margin - 5 and best_d is not None:
        try:
            def obj_cartesian(x):
                # Project to sphere
                norm = np.linalg.norm(x)
                if norm < 1e-10:
                    return 1e10
                return _compute_bound_from_direction(x / norm)
            
            res = minimize(obj_cartesian, best_d, method='BFGS',
                          options={'maxiter': 1000, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_d = res.x / np.linalg.norm(res.x)
        except Exception:
            pass
    
    # Phase 5: Final coordinate-wise polish
    if time.time() < deadline - safety_margin and best_d is not None:
        for _ in range(10):
            if time.time() > deadline - safety_margin:
                break
            improved = False
            for coord in range(3):
                # 1D search along this coordinate, keeping others fixed
                def obj_1d(t):
                    d_new = best_d.copy()
                    d_new[coord] = t
                    norm = np.linalg.norm(d_new)
                    if norm < 1e-10:
                        return 1e10
                    return _compute_bound_from_direction(d_new / norm)
                
                t0 = best_d[coord]
                try:
                    res_1d = minimize_scalar(obj_1d, bounds=(-1.5, 1.5), method='bounded',
                                            options={'xatol': 1e-12, 'maxiter': 200})
                    if res_1d.fun < best_val - 1e-14:
                        d_new = best_d.copy()
                        d_new[coord] = res_1d.x
                        norm = np.linalg.norm(d_new)
                        if norm > 1e-10:
                            d_new = d_new / norm
                            best_val = res_1d.fun
                            best_d = d_new
                            improved = True
                except Exception:
                    pass
            if not improved:
                break
    
    # Return the three coefficients (a0, a4, a8)
    if best_d is None:
        return np.array([0.0, 0.0, 0.0])
    return best_d

# EVOLVE-BLOCK-END
