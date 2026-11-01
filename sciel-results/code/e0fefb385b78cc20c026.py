import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(theta, phi):
    """
    Calculate energy and gradient for points on sphere parameterized by (theta, phi).
    theta: (N,) array in [0, pi]
    phi: (N,) array in [0, 2*pi)
    """
    # Convert to Cartesian
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    
    # Compute pairwise distances squared
    # d2[i,j] = (x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2
    # d2[i,j] = 2 - 2*(x_i x_j + y_i y_j + z_i z_j)
    dot = x[:, None] * x[None, :] + y[:, None] * y[None, :] + z[:, None] * z[None, :]
    d2 = 2.0 - 2.0 * dot
    
    # Avoid division by zero / log(0)
    # d2 can be 0 if points coincide. We add a small epsilon.
    # The energy is -sum log(d). d = sqrt(d2).
    # log(d) = 0.5 * log(d2)
    # E = -0.5 * sum_{i<j} log(d2[i,j])
    
    # Mask upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d2_upper = d2[mask]
    
    # Clip d2 to avoid log(0) or negative due to precision
    d2_safe = np.maximum(d2_upper, 1e-12)
    
    energy = -0.5 * np.sum(np.log(d2_safe))
    
    # Gradient computation
    # E = -0.5 * sum_{i<j} log(d2[i,j])
    # dE/dx_k = -0.5 * sum_{j!=k} (1/d2[k,j]) * d(d2[k,j])/dx_k
    # d2[k,j] = 2 - 2 dot[k,j]
    # d(d2[k,j])/dx_k = -2 * d(dot[k,j])/dx_k = -2 * x_j
    # So dE/dx_k = -0.5 * sum_{j!=k} (1/d2[k,j]) * (-2 x_j) = sum_{j!=k} x_j / d2[k,j]
    
    # We need dE/dtheta and dE/dphi
    # dE/dtheta_k = sum_{j!=k} (1/d2[k,j]) * (x_j * dx_k/dtheta_k + y_j * dy_k/dtheta_k + z_j * dz_k/dtheta_k)
    # dE/dphi_k = sum_{j!=k} (1/d2[k,j]) * (x_j * dx_k/dphi_k + y_j * dy_k/dphi_k + z_j * dz_k/dphi_k)
    
    # dx/dtheta = cos(theta)cos(phi)
    # dy/dtheta = cos(theta)sin(phi)
    # dz/dtheta = -sin(theta)
    
    # dx/dphi = -sin(theta)sin(phi)
    # dy/dphi = sin(theta)cos(phi)
    # dz/dphi = 0
    
    # Let W[i,j] = 1/d2[i,j] for i!=j, 0 otherwise.
    # dE/dtheta_k = sum_j W[k,j] * (x_j cos_t_k cos_p_k + y_j cos_t_k sin_p_k - z_j sin_t_k)
    # dE/dphi_k = sum_j W[k,j] * (-x_j sin_t_k sin_p_k + y_j sin_t_k cos_p_k)
    
    W = np.zeros((N, N))
    np.fill_diagonal(W, 0.0)
    # Use d2_safe for W to avoid inf
    W = 1.0 / np.maximum(d2, 1e-12)
    np.fill_diagonal(W, 0.0)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    # Term for theta gradient
    # grad_theta[k] = sum_j W[k,j] * (x_j cos_t_k cos_p_k + y_j cos_t_k sin_p_k - z_j sin_t_k)
    # = cos_t_k * (cos_p_k * sum_j W[k,j] x_j + sin_p_k * sum_j W[k,j] y_j) - sin_t_k * sum_j W[k,j] z_j
    
    sum_Wx = W @ x
    sum_Wy = W @ y
    sum_Wz = W @ z
    
    grad_theta = cos_t * (cos_p * sum_Wx + sin_p * sum_Wy) - sin_t * sum_Wz
    
    # Term for phi gradient
    # grad_phi[k] = sum_j W[k,j] * (-x_j sin_t_k sin_p_k + y_j sin_t_k cos_p_k)
    # = sin_t_k * (-sin_p_k * sum_j W[k,j] x_j + cos_p_k * sum_j W[k,j] y_j)
    
    grad_phi = sin_t * (-sin_p * sum_Wx + cos_p * sum_Wy)
    
    return energy, np.concatenate([grad_theta, grad_phi])

def _objective(u):
    theta = u[:N]
    phi = u[N:]
    e, _ = _energy_and_grad(theta, phi)
    return e

def _jac(u):
    theta = u[:N]
    phi = u[N:]
    _, g = _energy_and_grad(theta, phi)
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_energy = float('inf')
    best_u = None
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    # Convert to theta, phi
    theta_fib = np.arccos(z)
    phi_fib = phi
    inits.append(np.concatenate([theta_fib, phi_fib]))
    
    # 2. Random
    for _ in range(4):
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        theta_rand = np.arccos(pts[:, 2])
        phi_rand = np.arctan2(pts[:, 1], pts[:, 0])
        phi_rand = np.mod(phi_rand, 2 * np.pi)
        inits.append(np.concatenate([theta_rand, phi_rand]))
    
    # 3. Fibonacci with jitter
    for _ in range(2):
        theta_jit = theta_fib + np.random.randn(N) * 0.1
        phi_jit = phi_fib + np.random.randn(N) * 0.1
        theta_jit = np.clip(theta_jit, 1e-6, np.pi - 1e-6)
        phi_jit = np.mod(phi_jit, 2 * np.pi)
        inits.append(np.concatenate([theta_jit, phi_jit]))
    
    for init_u in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, init_u, jac=_jac, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_u = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 30: # Stop starting new optimizations if time is low
             break

    # Polish the best result
    if best_u is not None and time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_u, jac=_jac, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_energy:
                best_energy = res.fun
                best_u = res.x
        except Exception:
            pass

    # Convert best_u back to Cartesian
    if best_u is None:
        # Fallback to Fibonacci
        best_u = inits[0]
        
    theta = best_u[:N]
    phi = best_u[N:]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y, z])
    
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
