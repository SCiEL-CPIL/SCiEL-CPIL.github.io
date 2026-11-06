"""
Spherical harmonic density-based optimization for minimal logarithmic energy
of 100 points on the unit sphere.

Approach:
1. Use a spherical harmonic expansion to parameterize a probability density on
   the sphere. Optimize the coefficients to minimize the continuous log-energy
   integral, which helps avoid local minima.
2. Place N points via inverse CDF sampling from the optimized density.
3. Refine with direct point-based optimization (L-BFGS-B) using multiple
   restarts from different initializations (Fibonacci, random, SH-sampled).
"""

import numpy as np
from scipy.optimize import minimize
from math import fsum, log
import time
from scipy.special import lpmv

N = 100

def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # penalty for coincident points
            terms.append(log(1.0 / d))
    return fsum(terms)

def _energy_grad(points):
    """Compute gradient of the logarithmic energy with respect to point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -diff / d^2
            # d/dx_j log(1/|x_i - x_j|) = diff / d^2
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    return grad

def _spherical_harmonics(theta, phi, l_max):
    """
    Compute spherical harmonics Y_l^m(theta, phi) for l=0..l_max, m=-l..l.
    theta: polar angle (0 to pi), phi: azimuthal angle (0 to 2pi)
    Returns array of shape (len(theta), (l_max+1)^2) with real and imaginary parts
    stored as separate entries. For real-valued density, we use real spherical harmonics.
    """
    n_theta = len(theta)
    n_sh = (l_max + 1) ** 2
    sh = np.zeros((n_theta, n_sh))
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute Y_l^m(theta, phi)
            # Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * exp(i*m*phi)
            # For real spherical harmonics, we use:
            # If m > 0: sqrt(2) * Re(Y_l^m) = sqrt(2) * N * P_l^m * cos(m*phi)
            # If m < 0: sqrt(2) * Im(Y_l^m) = sqrt(2) * N * P_l^m * sin(|m|*phi)
            # If m == 0: Y_l^0 = N * P_l^0
            # where N = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
            
            abs_m = abs(m)
            # Compute associated Legendre polynomial P_l^m(cos(theta))
            # lpmv(m, l, x) computes P_l^m(x) with the Condon-Shortley phase
            p_lm = lpmv(abs_m, l, np.cos(theta))
            
            # Normalization factor
            import math
            log_norm = 0.5 * math.log((2*l+1)/(4*math.pi))
            log_norm += 0.5 * (math.lgamma(l - abs_m + 1) - math.lgamma(l + abs_m + 1))
            norm = math.exp(log_norm)
            
            if m == 0:
                sh[:, idx] = norm * p_lm
            elif m > 0:
                sh[:, idx] = math.sqrt(2) * norm * p_lm * np.cos(m * phi)
            else:  # m < 0
                sh[:, idx] = math.sqrt(2) * norm * p_lm * np.sin(abs_m * phi)
            
            idx += 1
    
    return sh

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=0):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _spherical_to_cart(theta, phi):
    """Convert spherical coordinates to Cartesian."""
    sin_t = np.sin(theta)
    return np.column_stack([
        sin_t * np.cos(phi),
        sin_t * np.sin(phi),
        np.cos(theta)
    ])

def _cart_to_spherical(pts):
    """Convert Cartesian to spherical coordinates (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi

def _optimize_from_initial(init_points, maxiter=2000):
    """Optimize point positions to minimize log-energy, keeping points on sphere."""
    pts = init_points.copy().astype(np.float64)
    
    # Convert to spherical coordinates
    theta, phi = _cart_to_spherical(pts)
    
    def _objective(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        return _energy(p)
    
    def _grad(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        g_cart = _energy_grad(p)
        # Convert Cartesian gradient to spherical coordinate gradient
        sin_t = np.sin(th)
        cos_t = np.cos(th)
        sin_p = np.sin(ph)
        cos_p = np.cos(ph)
        
        # d(p_i)/dtheta = [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
        dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
        g_th = np.sum(g_cart * dtheta, axis=1)
        
        # d(p_i)/dphi = [-sin(phi), cos(phi), 0]
        dphi = np.column_stack([-sin_p, cos_p, np.zeros(N)])
        g_ph = np.sum(g_cart * dphi, axis=1)
        
        return np.concatenate([g_th, g_ph])
    
    params0 = np.concatenate([theta, phi])
    
    result = minimize(_objective, params0, method='L-BFGS-B', 
                      jac=_grad, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    
    best_params = result.x
    best_energy = result.fun
    best_points = _spherical_to_cart(best_params[:N], best_params[N:])
    
    return best_points, best_energy

def _sh_density_init(l_max=8, n_quad=64):
    """
    Initialize points using a spherical harmonic density.
    Start with uniform density, then optimize SH coefficients to minimize
    continuous log-energy, and sample N points from the resulting density.
    """
    # Create a quadrature grid on the sphere
    # Use a simple grid: theta from 0 to pi, phi from 0 to 2pi
    theta_grid = np.linspace(0.01, np.pi - 0.01, n_quad)
    phi_grid = np.linspace(0, 2*np.pi, n_quad, endpoint=False)
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    theta_flat = THETA.flatten()
    phi_flat = PHI.flatten()
    
    # Compute spherical harmonics at quadrature points
    sh_vals = _spherical_harmonics(theta_flat, phi_flat, l_max)
    n_sh = sh_vals.shape[1]
    n_pts_quad = len(theta_flat)
    
    # Quadrature weights (simple uniform weighting, approximately correct for this grid)
    # dΩ = sin(theta) dtheta dphi
    weights = np.sin(theta_flat)
    weights /= weights.sum()  # normalize so sum(weights) = 1
    
    # Initial coefficients: uniform density corresponds to only c_00 being nonzero
    # Y_0^0 = 1/sqrt(4*pi), so rho = c_00 * Y_0^0 = 1/(4*pi)
    # We want rho = 1/(4*pi) for uniform, so c_00 = 1/sqrt(4*pi)
    import math
    c = np.zeros(n_sh)
    c[0] = 1.0 / math.sqrt(4 * math.pi)  # Y_0^0 = 1/sqrt(4pi), so rho = c[0]*Y_0^0 = 1/(4pi)
    
    # Optimize coefficients to minimize continuous log-energy
    # E = sum_{i,j} w_i * w_j * log(1/|p_i - p_j|) * (sum_k c_k Y_k(p_i)) * (sum_l c_l Y_l(p_j))
    # This is a quadratic form in c: E = c^T A c where A_{kl} = sum_{i,j} w_i w_j log(1/|p_i-p_j|) Y_k(p_i) Y_l(p_j)
    
    # Compute the matrix A
    # For efficiency, compute in chunks
    A = np.zeros((n_sh, n_sh))
    
    # Compute pairwise distances and log kernels
    # This is O(n_quad^2) which for n_quad=64 is 4096 points -> ~16M pairs, manageable
    cart_quad = _spherical_to_cart(theta_flat, phi_flat)
    
    # Compute log kernel matrix
    # L_{ij} = log(1/|p_i - p_j|)
    # To save memory, compute in blocks
    block_size = 512
    for i_start in range(0, n_pts_quad, block_size):
        i_end = min(i_start + block_size, n_pts_quad)
        for j_start in range(0, n_pts_quad, block_size):
            j_end = min(j_start + block_size, n_pts_quad)
            
            # Compute distances between points i_start:i_end and j_start:j_end
            diff = cart_quad[i_start:i_end, None, :] - cart_quad[None, j_start:j_end, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            
            # Avoid log(0) for i==j
            mask = np.ones_like(dists, dtype=bool)
            for ii in range(i_end - i_start):
                jj = j_start + ii
                if j_start <= jj < j_end:
                    mask[ii, jj - j_start] = False
            
            log_kernel = np.zeros_like(dists)
            log_kernel[mask] = np.log(1.0 / dists[mask])
            
            # Accumulate into A
            # A += sh[i_block] @ diag(w_i * w_j * log_kernel) @ sh[j_block]^T
            # More precisely: A_{kl} += sum_{i in block_i, j in block_j} w_i w_j log_kernel[i,j] sh[i,k] sh[j,l]
            
            sh_i = sh_vals[i_start:i_end, :]
            sh_j = sh_vals[j_start:j_end, :]
            
            # Weighted log kernel
            wk = weights[i_start:i_end, None] * weights[None, j_start:j_end] * log_kernel
            
            # A += sh_i^T @ wk @ sh_j
            A += sh_i.T @ wk @ sh_j
    
    # Minimize c^T A c subject to rho >= 0 and integral rho dOmega = 1
    # For simplicity, use gradient descent on c with projection
    # Gradient of c^T A c is 2*A*c
    # Constraint: sum_i w_i * (sum_k c_k Y_k(p_i)) = 1, i.e., sum_k c_k * (sum_i w_i Y_k(p_i)) = 1
    # For k=0: sum_i w_i Y_0^0(p_i) = sum_i w_i / sqrt(4pi) = 1/sqrt(4pi)
    # So constraint is c[0] / sqrt(4pi) = 1, i.e., c[0] = sqrt(4pi) for uniform
    # More generally: sum_k c_k * b_k = 1 where b_k = sum_i w_i Y_k(p_i)
    
    b = sh_vals.T @ weights  # b_k = sum_i w_i Y_k(p_i)
    
    # Simple gradient descent with projection onto the constraint and non-negativity
    c_curr = c.copy()
    lr = 0.01
    for it in range(100):
        grad = 2.0 * A @ c_curr
        c_curr = c_curr - lr * grad
        
        # Project onto constraint: sum_k c_k b_k = 1
        # Approximate: adjust c[0] since b[0] is the dominant term
        val = np.dot(c_curr, b)
        if abs(val) > 1e-10:
            c_curr = c_curr / val
        
        # Ensure density is non-negative: rho(p) = sum_k c_k Y_k(p) >= 0
        rho = sh_vals @ c_curr
        if np.any(rho < 0):
            # Scale down to make all non-negative
            min_rho = np.min(rho)
            if min_rho < 0:
                # Add a constant to make all positive, then renormalize
                c_curr[0] += -min_rho * math.sqrt(4 * math.pi)
                val = np.dot(c_curr, b)
                if abs(val) > 1e-10:
                    c_curr = c_curr / val
    
    # Now sample N points from the density rho
    rho = sh_vals @ c_curr
    rho = np.maximum(rho, 0)
    rho /= rho.sum()  # normalize
    
    # Inverse CDF sampling
    cdf = np.cumsum(rho)
    cdf /= cdf[-1]
    rng = np.random.RandomState(42)
    u = rng.uniform(0, 1, N)
    indices = np.searchsorted(cdf, u)
    indices = np.clip(indices, 0, n_pts_quad - 1)
    
    # Add small random perturbation to avoid exact duplicates
    theta_sampled = theta_flat[indices] + rng.normal(0, 0.01, N)
    phi_sampled = phi_flat[indices] + rng.normal(0, 0.01, N)
    theta_sampled = np.clip(theta_sampled, 0.001, np.pi - 0.001)
    phi_sampled = phi_sampled % (2 * np.pi)
    
    points = _spherical_to_cart(theta_sampled, phi_sampled)
    return points

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    Uses spherical harmonic density initialization plus multiple restarts
    with gradient-based optimization.
    """
    start_time = time.time()
    timeout = 100.0  # leave some margin for the 120s limit
    
    # Generate initializations
    inits = []
    
    # Fibonacci spiral
    inits.append(("fib", _fibonacci_spiral()))
    
    # Random initializations
    inits.append(("rand0", _random_points(42)))
    inits.append(("rand1", _random_points(123)))
    inits.append(("rand2", _random_points(999)))
    
    # Spherical harmonic density initialization
    try:
        sh_init = _sh_density_init(l_max=8, n_quad=48)
        inits.append(("sh", sh_init))
    except Exception:
        pass
    
    best_points = None
    best_energy = np.inf
    
    for name, init in inits:
        if time.time() - start_time > timeout:
            break
        
        try:
            pts, energy = _optimize_from_initial(init, maxiter=1500)
            # Ensure points are on the sphere
            pts = _project_to_sphere(pts)
            # Verify energy
            e = _energy(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral()
    
    # Final check: ensure valid points
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral()
    
    # Ensure shape is (N, 3)
    assert best_points.shape == (N, 3)
    
    return best_points
