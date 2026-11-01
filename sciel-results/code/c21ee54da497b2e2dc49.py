"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0, 1) for a given base."""
    seq = []
    f = 1.0 / base
    i = 1
    while len(seq) < n:
        result = 0.0
        i_temp = i
        while i_temp > 0:
            i_temp, digit = divmod(i_temp, base)
            result += f * digit
            f /= base
        seq.append(result)
        i += 1
    return seq


def _initialize_halton(n):
    """Initialize n points on the unit sphere using a 3D Halton sequence."""
    # Use three different prime bases for the three coordinates
    x = np.array(_halton_sequence(n, 2))
    y = np.array(_halton_sequence(n, 3))
    z = np.array(_halton_sequence(n, 5))
    
    # Map from [0,1]^3 to sphere surface
    # Use inverse CDF for uniform distribution on sphere
    # z is in [-1, 1], phi is in [0, 2*pi)
    z_sphere = 2.0 * z - 1.0  # z in [-1, 1]
    phi = 2.0 * math.pi * x
    
    # y is not directly used in standard spherical mapping, but we can use it for better dispersion
    # Actually, let's use a different approach: use all three coordinates to define direction
    # Normalize the 3D points to lie on the unit sphere
    points = np.column_stack([x, y, z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    points = points / norms
    return points


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    
    # Compute log(1/dist) = -log(dist)
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    log_terms = -np.log(dist[iu])
    
    # Use math.fsum for accurate summation as specified
    return math.fsum(log_terms.tolist())


def _cartesian_gradient(points):
    """Compute the gradient of the log energy w.r.t. Cartesian coordinates."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Gradient of -log(dist) w.r.t. points[i] is (points[i] - points[j]) / dist^2
    # Sum over j != i
    # dist^2
    dist2 = dist ** 2
    np.fill_diagonal(dist2, 1.0)  # Avoid division by zero
    
    # Gradient contribution: sum_j (points[i] - points[j]) / dist_ij^2
    # This is: points[i] * sum_j(1/dist2_ij) - sum_j(points[j] / dist2_ij)
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # sum_j(1/dist2_ij) for each i
    sum_inv = np.sum(inv_dist2, axis=1)
    
    # sum_j(points[j] / dist2_ij) for each i
    # This is a matrix-vector product: inv_dist2 @ points
    sum_pts = inv_dist2 @ points
    
    grad = points * sum_inv[:, np.newaxis] - sum_pts
    return grad


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms


def _riemannian_cg_step(points, max_iter=50, tol=1e-10):
    """
    Perform Riemannian conjugate gradient steps on the sphere.
    Uses exponential map for retraction.
    """
    n = len(points)
    x = points.copy()
    
    # Initial gradient in tangent space
    cart_grad = _cartesian_gradient(x)
    # Project to tangent space: grad_t = grad - (grad . x) * x
    proj = np.sum(cart_grad * x, axis=1, keepdims=True)
    tangent_grad = cart_grad - proj * x
    
    # Initial search direction
    d = -tangent_grad
    d_prev = d.copy()
    
    for it in range(max_iter):
        # Check convergence
        if np.linalg.norm(tangent_grad) < tol:
            break
        
        # Line search in tangent space
        best_step = 1e-4
        best_energy = _log_energy(x)
        
        for step in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
            # Exponential map: x_new = x * cos(step * ||d||) + d/||d|| * sin(step * ||d||)
            d_norm = np.linalg.norm(d, axis=1, keepdims=True)
            d_norm[d_norm < 1e-12] = 1.0
            d_unit = d / d_norm
            
            angle = step * np.linalg.norm(d, axis=1)
            x_new = x * np.cos(angle)[:, np.newaxis] + d_unit * np.sin(angle)[:, np.newaxis]
            x_new = _project_to_sphere(x_new)
            
            e_new = _log_energy(x_new)
            if e_new < best_energy:
                best_energy = e_new
                best_step = step
                x_candidate = x_new
            else:
                break
        
        if best_step == 1e-4 and _log_energy(x_candidate) >= best_energy - 1e-12:
            # Try smaller steps
            for step in [1e-5, 3e-5, 1e-6]:
                d_norm = np.linalg.norm(d, axis=1, keepdims=True)
                d_norm[d_norm < 1e-12] = 1.0
                d_unit = d / d_norm
                angle = step * np.linalg.norm(d, axis=1)
                x_candidate = x * np.cos(angle)[:, np.newaxis] + d_unit * np.sin(angle)[:, np.newaxis]
                x_candidate = _project_to_sphere(x_candidate)
                e_new = _log_energy(x_candidate)
                if e_new < best_energy:
                    best_energy = e_new
                    best_step = step
        
        x = x_candidate
        
        # New gradient
        cart_grad = _cartesian_gradient(x)
        proj = np.sum(cart_grad * x, axis=1, keepdims=True)
        tangent_grad_new = cart_grad - proj * x
        
        # Polak-Ribiere beta
        grad_diff = np.sum(tangent_grad_new * (tangent_grad_new - tangent_grad), axis=1)
        grad_norm_sq = np.sum(tangent_grad ** 2, axis=1)
        beta = np.maximum(0.0, grad_diff / (grad_norm_sq + 1e-12))
        
        # New direction
        d = -tangent_grad_new + beta[:, np.newaxis] * d_prev
        d_prev = d
        
        tangent_grad = tangent_grad_new
    
    return x


def construct():
    # Initialize with Halton sequence
    points = _initialize_halton(N)
    
    # Alternate between Cartesian projected gradient and Riemannian CG
    for outer_iter in range(30):
        # Cartesian projected gradient descent
        for inner_iter in range(20):
            grad = _cartesian_gradient(points)
            # Project gradient to tangent space
            proj = np.sum(grad * points, axis=1, keepdims=True)
            tangent_grad = grad - proj * points
            
            # Adaptive step size
            step = 1e-4 / (1.0 + 0.1 * outer_iter)
            
            # Update in Cartesian space
            points_new = points - step * tangent_grad
            points_new = _project_to_sphere(points_new)
            
            # Check if energy decreased
            e_old = _log_energy(points)
            e_new = _log_energy(points_new)
            
            if e_new < e_old:
                points = points_new
            else:
                # Backtrack
                for _ in range(5):
                    step *= 0.5
                    points_new = points - step * tangent_grad
                    points_new = _project_to_sphere(points_new)
                    e_new = _log_energy(points_new)
                    if e_new < e_old:
                        points = points_new
                        break
        
        # Riemannian conjugate gradient
        points = _riemannian_cg_step(points, max_iter=30, tol=1e-10)
    
    # Final projection to ensure unit norm
    points = _project_to_sphere(points)
    
    # Verify validity
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-12) or not np.all(np.isfinite(points)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return points
# EVOLVE-BLOCK-END
