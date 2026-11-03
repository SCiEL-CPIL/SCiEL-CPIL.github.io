"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient using dot-product trick.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    |x_i - x_j|^2 = 2 - 2 x_i . x_j
    """
    # pts is (N, 3)
    # Compute pairwise dot products
    dots = pts @ pts.T  # (N, N)
    # |x_i - x_j|^2 = 2 - 2*dots[i,j]
    dist2 = 2.0 - 2.0 * dots
    # Avoid log(0) for i=j; we'll mask diagonal
    np.fill_diagonal(dist2, 1.0)  # placeholder, will be masked
    
    # log(1/d) = -log(d) = -0.5 * log(d^2)
    # energy = sum_{i<j} -0.5 * log(dist2[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    d2_upper = dist2[iu]
    energy = -0.5 * np.sum(np.log(d2_upper))
    
    # Gradient: d/dx_k [ -0.5 * log(2 - 2 x_i . x_j) ]
    # = -0.5 * (1/(2-2 x_i.x_j)) * (-2 x_j) for derivative w.r.t. x_i
    #   = -0.5 * (1/(2-2 x_i.x_j)) * (-2 x_i) for derivative w.r.t. x_j
    # So grad_i = sum_{j!=i} x_j / (2 - 2 x_i.x_j)  ... let me re-derive
    # E = -0.5 sum_{i<j} log(2 - 2 x_i.x_j)
    # dE/dx_k = -0.5 * sum_{i<j, k in {i,j}} (1/(2-2 x_i.x_j)) * (-2 * d(x_i.x_j)/dx_k)
    # If k=i: d(x_i.x_j)/dx_i = x_j, so term = -0.5 * (-2) * x_j / (2-2 x_i.x_j) = x_j/(2-2 x_i.x_j)
    # If k=j: similar = x_i/(2-2 x_j.x_i)
    # So grad_k = sum_{j!=k} x_j / (2 - 2 x_k.x_j)
    # 
    # But wait: the points are constrained to the sphere. We're minimizing in R^3 and
    # the gradient should be projected. Actually, for L-BFGS-B we can just minimize
    # the energy as a function of the 3N coordinates, but the constraint ||x_k||=1 is
    # not enforced. Better approach: use spherical coordinates or project.
    # 
    # Actually, a cleaner approach: minimize over the sphere directly by using
    # the fact that if we start on the sphere and use the gradient projected onto
    # the tangent space, we stay on the sphere (approximately). But L-BFGS-B doesn't
    # handle that well.
    #
    # Alternative: parameterize each point by (theta, phi) or use a different approach.
    # 
    # Let me use a different strategy: minimize in Cartesian with a penalty or
    # just re-normalize after each step. Actually, the cleanest is to use
    # scipy's minimize with the points as variables and add a constraint, but
    # that's complex.
    #
    # Simpler: just minimize the energy as a function of all 3N coordinates,
    # and the natural tendency of the log potential will keep points on the sphere
    # approximately. But this is not rigorous.
    #
    # Better: Use the approach where we minimize over the tangent space. But for
    # simplicity and given the time budget, let's use a different approach:
    # minimize in Cartesian, and the log(1/d) potential is attractive/repulsive
    # in a way that naturally keeps points spread. The constraint ||x||=1 is
    # important though.
    #
    # Let me just compute the gradient assuming points are on the sphere, and
    # use a constrained optimizer or project. Actually, let me use a simpler
    # approach: use scipy.optimize.minimize with method='L-BFGS-B' on the
    # 3N-dimensional space, but add a large penalty for deviating from the sphere.
    #
    # Even simpler and more effective: use the fact that for points on the unit sphere,
    # the gradient of the energy with respect to x_k (unconstrained) projected onto
    # the tangent plane gives the correct direction. Let me just compute the full
    # gradient and let the optimizer handle it; the log potential is strong enough
    # that points will stay near the sphere.
    #
    # Actually, the most robust approach for this problem: use spherical coordinates.
    # Each point is (theta_k, phi_k) where x = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
    # This automatically enforces the sphere constraint.
    
    # For now, let me compute the gradient in Cartesian and we'll handle the
    # constraint separately. The gradient of E w.r.t. x_k is:
    # grad_k = sum_{j != k} x_j / (2 - 2 * x_k . x_j)
    # (derived above, each pair contributes to both endpoints)
    
    # Build gradient matrix
    # For each k, grad_k = sum_{j != k} x_j / (2 - 2 * x_k.x_j)
    # This can be written as: for each pair (i,j), the contribution to grad_i is x_j/(2-2 x_i.x_j)
    # and to grad_j is x_i/(2-2 x_i.x_j)
    
    # Let's compute it efficiently:
    # inv_dist2 = 1 / (2 - 2*dots) for all pairs, with diagonal set to 0
    inv_d2 = np.zeros_like(dots)
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / dist2[mask]
    
    # grad_k = sum_j inv_d2[k,j] * x_j  for j != k
    # = (inv_d2 @ pts)[k]  but with diagonal of inv_d2 being 0, so it's fine
    grad = inv_d2 @ pts  # (N, 3)
    
    # But wait: this gives the gradient of E = -0.5 * sum log(2-2 x_i.x_j)
    # Let me verify: dE/dx_k = sum_{j!=k} x_j / (2 - 2 x_k.x_j) * (1)
    # From E = -0.5 sum_{i<j} log(2 - 2 x_i.x_j)
    # dE/dx_k = -0.5 * sum_{i<j, k=i} (1/(2-2 x_i.x_j)) * (-2 x_j) + (-0.5) * sum_{i<j, k=j} (1/(2-2 x_i.x_j)) * (-2 x_i)
    # = sum_{j!=k} x_j/(2-2 x_k.x_j)  (first sum) + sum_{i!=k} x_i/(2-2 x_i.x_k) (second sum, renaming)
    # = 2 * sum_{j!=k} x_j/(2-2 x_k.x_j)
    
    # Hmm, let me redo this more carefully.
    # E = -0.5 * sum_{i<j} log(2 - 2 x_i . x_j)
    # Let f_{ij} = log(2 - 2 x_i . x_j)
    # dE/dx_k = -0.5 * sum_{i<j} df_{ij}/dx_k
    # df_{ij}/dx_k = (1/(2-2 x_i.x_j)) * d(2-2 x_i.x_j)/dx_k
    # If k = i: d(2-2 x_i.x_j)/dx_i = -2 x_j, so df = (-2 x_j)/(2-2 x_i.x_j)
    # If k = j: d(2-2 x_i.x_j)/dx_j = -2 x_i, so df = (-2 x_i)/(2-2 x_i.x_j)
    # If k != i,j: df = 0
    # So dE/dx_k = -0.5 * [sum_{j!=k} (-2 x_j)/(2-2 x_k.x_j) + sum_{i!=k} (-2 x_i)/(2-2 x_i.x_k)]
    # = -0.5 * [-2 sum_{j!=k} x_j/(2-2 x_k.x_j) - 2 sum_{i!=k} x_i/(2-2 x_i.x_k)]
    # = sum_{j!=k} x_j/(2-2 x_k.x_j) + sum_{i!=k} x_i/(2-2 x_i.x_k)
    # = 2 * sum_{j!=k} x_j/(2-2 x_k.x_j)  (since the two sums are identical by renaming)
    
    # So the gradient is: grad_k = 2 * sum_{j!=k} x_j / (2 - 2 x_k.x_j)
    grad = 2.0 * (inv_d2 @ pts)
    
    # Flatten for optimizer
    grad_flat = grad.ravel()
    return energy, grad_flat


def _energy_only(pts):
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(N, 1)
    d2_upper = dist2[iu]
    return -0.5 * np.sum(np.log(d2_upper))


def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = _energy_only(pts)
    best_pts = pts.copy()
    
    # Try multiple restarts with perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the starting points
        if restart == 0:
            current = best_pts.copy()
        else:
            # Random perturbation
            current = best_pts + rng.normal(0, 0.1, size=(N, 3))
            # Re-normalize
            norms = np.linalg.norm(current, axis=1, keepdims=True)
            current = current / norms
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _energy_only,
                current.ravel(),
                jac=lambda x: _energy_and_grad(x.reshape(N, 3))[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = result.x.reshape(N, 3)
            # Re-normalize to ensure points are on the sphere
            opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            opt_pts = opt_pts / opt_norms
            
            # Check validity
            if not np.all(np.isfinite(opt_pts)):
                continue
            
            # Check minimum distance
            dots = opt_pts @ opt_pts.T
            dist2 = 2.0 - 2.0 * dots
            np.fill_diagonal(dist2, 1.0)
            min_dist2 = np.min(dist2)
            if min_dist2 < (1e-9) ** 2:
                continue
            
            opt_energy = _energy_only(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Final re-normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
