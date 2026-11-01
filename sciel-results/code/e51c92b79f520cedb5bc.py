"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """Minimize logarithmic energy of N points on unit sphere via L-BFGS-B."""
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Fibonacci spiral initial guess
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute energy and gradient using spherical coordinates (theta, phi)
    # theta = polar angle [0, pi], phi = azimuthal angle [0, 2*pi]
    # This parameterization keeps points on the sphere automatically
    
    def _energy_and_grad(theta_phi):
        """theta_phi: (N, 2) array of (theta, phi) for each point."""
        if time.time() - t0 > DEADLINE:
            # Return large penalty to stop optimization
            return 1e30, np.zeros_like(theta_phi)
        
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        pts = np.column_stack([x, y, z])
        
        # Compute pairwise distances
        # d_ij^2 = 2 - 2*(x_i*x_j + y_i*y_j + z_i*z_j)
        dots = pts @ pts.T
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        
        # Ensure no zero distances
        d2 = np.maximum(d2, 1e-30)
        
        # Energy = sum_{i<j} log(1/sqrt(d2_ij)) = -0.5 * sum_{i<j} log(d2_ij)
        # Only upper triangle
        iu = np.triu_indices(N, k=1)
        log_d2 = np.log(d2[iu])
        energy = -0.5 * np.sum(log_d2)
        
        # Gradient with respect to theta and phi
        # d(energy)/d(theta_k) and d(energy)/d(phi_k)
        # E = -0.5 * sum_{i<j} log(d2_ij)
        # dE/dx_k = -0.5 * sum_{j!=k} (1/d2_kj) * 2*(x_k - x_j) / 2 ... 
        # Let's be more careful:
        # d2_kj = |x_k - x_j|^2
        # d(d2_kj)/dx_k = 2*(x_k - x_j)
        # dE/dx_k = -0.5 * sum_{j!=k} (1/d2_kj) * 2*(x_k - x_j)
        #         = -sum_{j!=k} (x_k - x_j) / d2_kj
        
        grad_cart = np.zeros_like(pts)
        for k in range(N):
            # Vector from j to k for all j != k
            diff = pts[k] - pts  # shape (N, 3), diff[j] = x_k - x_j
            # weights = 1/d2_kj for j != k
            w = np.zeros(N)
            w[np.arange(N) != k] = 1.0 / d2[k, np.arange(N)]
            w[k] = 0.0
            # grad_cart[k] = -sum_j w[j] * diff[j]
            grad_cart[k] = -np.sum(w[:, np.newaxis] * diff, axis=0)
        
        # Now convert Cartesian gradient to (theta, phi) gradient
        # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
        # dx/dtheta = cos(theta)*cos(phi), dx/dphi = -sin(theta)*sin(phi)
        # dy/dtheta = cos(theta)*sin(phi), dy/dphi = sin(theta)*cos(phi)
        # dz/dtheta = -sin(theta),           dz/dphi = 0
        
        grad_theta = np.zeros(N)
        grad_phi = np.zeros(N)
        for k in range(N):
            st = np.sin(theta[k])
            ct = np.cos(theta[k])
            sp = np.sin(phi[k])
            cp = np.cos(phi[k])
            
            gx, gy, gz = grad_cart[k]
            
            # Chain rule: dE/dtheta_k = gx * (dx/dtheta) + gy * (dy/dtheta) + gz * (dz/dtheta)
            grad_theta[k] = gx * ct * cp + gy * ct * sp + gz * (-st)
            
            # dE/dphi_k = gx * (dx/dphi) + gy * (dy/dphi) + gz * (dz/dphi)
            grad_phi[k] = gx * (-st * sp) + gy * (st * cp) + gz * 0.0
        
        grad = np.column_stack([grad_theta, grad_phi])
        
        return energy, grad
    
    # Set up bounds: theta in [0, pi], phi in [0, 2*pi]
    theta_bounds = (0.0, np.pi)
    phi_bounds = (0.0, 2.0 * np.pi)
    bounds = []
    for _ in range(N):
        bounds.append(theta_bounds)
        bounds.append(phi_bounds)
    
    # Initial guess from Fibonacci spiral
    pts_init = fibonacci_spiral(N)
    theta_init = np.arccos(np.clip(pts_init[:, 2], -1.0, 1.0))
    phi_init = np.arctan2(pts_init[:, 1], pts_init[:, 0])
    phi_init = np.where(phi_init < 0, phi_init + 2*np.pi, phi_init)
    tp_init = np.column_stack([theta_init, phi_init]).flatten()
    
    # Optimize
    best_energy = 1e30
    best_tp = tp_init.copy()
    
    result = minimize(
        _energy_and_grad,
        tp_init,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': 5000,
            'maxls': 50,
            'maxfun': 10000,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'iprint': 0
        }
    )
    
    if result.fun < best_energy:
        best_energy = result.fun
        best_tp = result.x
    
    # Try a second optimization from a perturbed start if time allows
    if time.time() - t0 < DEADLINE - 30:
        np.random.seed(42)
        tp_perturbed = best_tp.copy()
        tp_perturbed += np.random.normal(0, 0.05, size=tp_perturbed.shape)
        # Re-wrap angles
        tp_perturbed[1::2] = tp_perturbed[1::2] % (2*np.pi)
        tp_perturbed[0::2] = np.clip(tp_perturbed[0::2], 0.01, np.pi - 0.01)
        
        result2 = minimize(
            _energy_and_grad,
            tp_perturbed,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 3000,
                'maxls': 50,
                'maxfun': 6000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'iprint': 0
            }
        )
        
        if result2.fun < best_energy:
            best_energy = result2.fun
            best_tp = result2.x
    
    # Convert back to Cartesian
    theta = best_tp[0::2]
    phi = best_tp[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Nudge apart
                pts[j] = pts[i] + np.array([1e-6, 0, 0])
                pts[j] = pts[j] / np.linalg.norm(pts[j])
    
    return pts
# EVOLVE-BLOCK-END
