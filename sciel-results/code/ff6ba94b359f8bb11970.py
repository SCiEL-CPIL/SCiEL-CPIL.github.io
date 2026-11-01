"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import math
from scipy.optimize import minimize
import time

N = 100


def _energy_and_grad(theta_phi):
    """Compute log energy and gradient w.r.t. (theta, phi) parameters.
    
    theta in [0, pi], phi in [0, 2*pi)
    x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
    """
    theta = theta_phi[0::2]
    phi = theta_phi[1::2]
    
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    x = st * cp
    y = st * sp
    z = ct
    
    # Compute pairwise distances
    # d_ij^2 = 2 - 2*(x_i*x_j + y_i*y_j + z_i*z_j)
    dots = x[:, None] * x[None, :] + y[:, None] * y[None, :] + z[:, None] * z[None, :]
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_vals = np.sqrt(d2[iu])
    energy = -np.sum(np.log(d_vals))
    
    # Gradient computation
    # d/dtheta_i of log(d_ij) for i != j
    # d_ij^2 = 2 - 2*(x_i x_j + y_i y_j + z_i z_j)
    # dd2/dtheta_i = -2*(dx_i/dtheta_i * x_j + dy_i/dtheta_i * y_j + dz_i/dtheta_i * z_j)
    # dx/dtheta = cos(theta)*cos(phi), dy/dtheta = cos(theta)*sin(phi), dz/dtheta = -sin(theta)
    # dlog(d)/dtheta_i = (1/d) * (1/(2d)) * dd2/dtheta_i = dd2/dtheta_i / (2*d^2)
    # But energy = -sum log(d), so grad_energy = -grad_log_d
    
    # For point i, gradient contribution from all j != i
    # Let's compute grad w.r.t. x, y, z first, then chain rule
    
    # grad_x_i = sum_{j!=i} d/dx_i [-log(d_ij)] = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Similarly for y, z
    
    # Vectorized: for each i, sum over j of (x_i - x_j)/d_ij^2
    # Note: d_ij^2 = 2 - 2*dots_ij
    
    # grad_x[i] = sum_j (x_i - x_j) / d2[i,j]  (j != i)
    # = x_i * sum_{j!=i} 1/d2[i,j] - sum_{j!=i} x_j/d2[i,j]
    
    # Build matrix of 1/d2 with diagonal 0
    inv_d2 = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / d2[mask]
    
    # sum_j 1/d2[i,j] for j != i
    sum_inv_d2 = inv_d2.sum(axis=1)
    # sum_j x_j / d2[i,j]
    sum_x_inv = inv_d2 @ x
    sum_y_inv = inv_d2 @ y
    sum_z_inv = inv_d2 @ z
    
    grad_x = x * sum_inv_d2 - sum_x_inv
    grad_y = y * sum_inv_d2 - sum_y_inv
    grad_z = z * sum_inv_d2 - sum_z_inv
    
    # Energy = -sum log(d), so grad_energy = -grad of sum log(d)
    # But wait: d/dx_i [-log(d_ij)] = -(1/d_ij) * d/dx_i [d_ij] = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # So grad_energy_x[i] = -sum_{j!=i} (x_i - x_j)/d2[i,j] = -grad_x[i]
    
    grad_energy_x = -grad_x
    grad_energy_y = -grad_y
    grad_energy_z = -grad_z
    
    # Now chain rule: d/dtheta = grad_x * dx/dtheta + grad_y * dy/dtheta + grad_z * dz/dtheta
    # dx/dtheta = cos(theta)*cos(phi) = ct*cp
    # dy/dtheta = cos(theta)*sin(phi) = ct*sp
    # dz/dtheta = -sin(theta) = -st
    
    grad_theta = grad_energy_x * ct * cp + grad_energy_y * ct * sp + grad_energy_z * (-st)
    
    # d/dphi:
    # dx/dphi = -sin(theta)*sin(phi) = -st*sp
    # dy/dphi = sin(theta)*cos(phi) = st*cp
    # dz/dphi = 0
    
    grad_phi = grad_energy_x * (-st * sp) + grad_energy_y * (st * cp)
    
    # Interleave back
    grad = np.zeros_like(theta_phi)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    
    return energy, grad


def _energy_only(theta_phi):
    e, _ = _energy_and_grad(theta_phi)
    return e


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_energy = np.inf
    best_points = None
    
    # Try multiple initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        elif restart == 1:
            # Slightly perturbed Fibonacci
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.normal(0, 0.05, N)
            r = np.sqrt(1.0 - z * z)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        else:
            # Random uniform on sphere
            u = np.random.uniform(-1, 1, N)
            v = np.random.uniform(0, 2*np.pi, N)
            s = np.sqrt(1 - u*u)
            x = s * np.cos(v)
            y = s * np.sin(v)
            z = u
        
        # Convert to theta, phi
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        
        theta_phi = np.zeros(2*N)
        theta_phi[0::2] = theta
        theta_phi[1::2] = phi
        
        # Optimize
        try:
            res = minimize(
                _energy_only,
                theta_phi,
                jac=_energy_and_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            
            e = res.fun
            if e < best_energy:
                best_energy = e
                # Convert back to Cartesian
                th = res.x[0::2]
                ph = res.x[1::2]
                st = np.sin(th)
                ct = np.cos(th)
                sp = np.sin(ph)
                cp = np.cos(ph)
                best_points = np.column_stack([st*cp, st*sp, ct])
        except Exception:
            continue
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 5.0:
        theta = np.arccos(np.clip(best_points[:, 2], -1.0, 1.0))
        phi = np.arctan2(best_points[:, 1], best_points[:, 0])
        theta_phi = np.zeros(2*N)
        theta_phi[0::2] = theta
        theta_phi[1::2] = phi
        
        try:
            res = minimize(
                _energy_only,
                theta_phi,
                jac=_energy_and_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                th = res.x[0::2]
                ph = res.x[1::2]
                st = np.sin(th)
                ct = np.cos(th)
                sp = np.sin(ph)
                cp = np.cos(ph)
                best_points = np.column_stack([st*cp, st*sp, ct])
        except Exception:
            pass
    
    # Validate and normalize
    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    norms = np.maximum(norms, 1e-15)
    best_points = best_points / norms[:, None]
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb
                best_points[j] += np.random.normal(0, 1e-6, 3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
