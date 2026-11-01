"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _points_from_params(params):
    """Convert (theta, phi) parameters to Cartesian coordinates on the unit sphere."""
    thetas = params[0::2]
    phis = params[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])

def _energy_and_grad(params):
    """
    Compute the Riesz energy (s=2) and its gradient w.r.t. the parameters (theta, phi).
    Energy = sum_{i<j} 1 / ||p_i - p_j||^2
    """
    P = _points_from_params(params)
    n = len(P)
    
    # Compute pairwise distances squared
    # dist_sq[i, j] = ||P[i] - P[j]||^2
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum over i < j of 1 / dist_sq[i, j]
    # We compute sum over all i != j and divide by 2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    energy = 0.5 * np.sum(inv_dist_sq)
    
    # Gradient w.r.t. P
    # dE/dP_i = sum_{j != i} d/dP_i (1 / ||P_i - P_j||^2)
    # Let r_i = P_i - P_j. ||r_i||^2 = r_i dot r_i.
    # d/dP_i (1 / (r_i dot r_i)) = -1 / (r_i dot r_i)^2 * 2 * r_i
    # = -2 * r_i / dist_sq[i, j]^2
    
    # grad_P[i] = sum_{j != i} -2 * (P_i - P_j) / dist_sq[i, j]^2
    # Note: dist_sq is symmetric.
    # We can vectorize this.
    # inv_dist_sq_2 = 1 / dist_sq^2
    inv_dist_sq_2 = 1.0 / (dist_sq**2)
    np.fill_diagonal(inv_dist_sq_2, 0.0)
    
    # grad_P[i] = -2 * sum_j (P_i - P_j) * inv_dist_sq_2[i, j]
    # diff is P_i - P_j.
    # grad_P = -2 * (diff * inv_dist_sq_2[:,:,np.newaxis]).sum(axis=1)
    grad_P = -2.0 * np.sum(diff * inv_dist_sq_2[:, :, np.newaxis], axis=1)
    
    # Chain rule: dE/dparams = (dP/dparams)^T * grad_P
    # params = [theta_0, phi_0, theta_1, phi_1, ...]
    # P_i = [sin(t_i)cos(p_i), sin(t_i)sin(p_i), cos(t_i)]
    
    thetas = params[0::2]
    phis = params[1::2]
    
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    sin_p = np.sin(phis)
    cos_p = np.cos(phis)
    
    # dP_i/dtheta = [cos(t)cos(p), cos(t)sin(p), -sin(t)]
    # dP_i/dphi   = [-sin(t)sin(p), sin(t)cos(p), 0]
    
    dP_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    dP_dphi   = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(n)])
    
    # grad_params[2i]   = dot(dP_dtheta[i], grad_P[i])
    # grad_params[2i+1] = dot(dP_dphi[i], grad_P[i])
    
    grad_theta = np.sum(dP_dtheta * grad_P, axis=1)
    grad_phi   = np.sum(dP_dphi * grad_P, axis=1)
    
    grad_params = np.zeros(2 * n)
    grad_params[0::2] = grad_theta
    grad_params[1::2] = grad_phi
    
    return energy, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_energy = np.inf
    best_params = None
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to params (theta, phi)
    theta_init = np.arccos(P_init[:, 2])
    phi_init = np.arctan2(P_init[:, 1], P_init[:, 0])
    # Handle arctan2 range [-pi, pi] to [0, 2pi] for consistency, though bounds allow [0, 2pi]
    phi_init[phi_init < 0] += 2 * np.pi
    
    params_init = np.zeros(2 * N)
    params_init[0::2] = theta_init
    params_init[1::2] = phi_init
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, np.pi))      # theta
        bounds.append((0.0, 2 * np.pi))  # phi
    
    # Run multiple restarts with slight perturbations
    np.random.seed(42)
    num_restarts = 5
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        if k == 0:
            p0 = params_init.copy()
        else:
            # Add small random noise
            p0 = params_init + np.random.normal(0, 0.1, 2 * N)
            # Wrap phi
            p0[1::2] = np.mod(p0[1::2], 2 * np.pi)
            # Clip theta
            p0[0::2] = np.clip(p0[0::2], 0.01, np.pi - 0.01)
        
        try:
            res = minimize(
                _energy_and_grad, 
                p0, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds, 
                options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    # If we have time, do a final high-precision polish on the best result
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _energy_and_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds, 
                options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            if res_final.fun < best_energy:
                best_energy = res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial
        best_params = params_init

    P_final = _points_from_params(best_params)
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(P_final, axis=1, keepdims=True)
    P_final = P_final / norms
    
    return P_final
# EVOLVE-BLOCK-END
