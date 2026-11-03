"""Rewritten program: Riesz/Logarithmic energy minimisation via L-BFGS-B on sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)."""
    # pts shape (N, 3), assumed on unit sphere
    dists = cdist(pts, pts)
    # Take upper triangle
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_and_grad(pts):
    """Return energy and gradient w.r.t. unconstrained parameters (before projection).
    We use the fact that for points on the sphere, the gradient of the energy
    projected onto the tangent space is sufficient.
    
    For simplicity and robustness, we compute energy on projected points and
    use finite-difference-like approach via autograd-style, but since we don't have
    autograd, we'll use a trick: parameterize via Cartesian and use the fact that
    the gradient in Cartesian space projected to tangent gives the correct direction.
    
    Actually, let's just use scipy's minimize with bounds and handle the constraint
    by projecting after each step. We'll use a simple approach: minimize in Cartesian
    space with the constraint that points are on the sphere, using penalty or
    re-projection.
    
    Better approach: use the standard technique of minimizing on the sphere by
    computing the Riemannian gradient. For the log energy E = -sum_{i<j} log(d_ij),
    the gradient w.r.t. point i is:
      grad_i = sum_{j != i} (x_i - x_j) / (||x_i - x_j||^2)
    Then project this gradient onto the tangent space at x_i:
      grad_tangent_i = grad_i - (grad_i · x_i) * x_i
    """
    pts = _project_to_sphere(pts)
    energy = _log_energy(pts)
    
    # Compute gradient
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-30:
                continue
            # dE/dx_i = sum_{j != i} (x_i - x_j) / dist^2
            # Because E = -sum log(dist), dE/dx_i = -sum (1/dist) * (x_i - x_j)/dist = -sum (x_i-x_j)/dist^2
            # Wait: d/dx_i log(||x_i - x_j||) = (x_i - x_j)/||x_i - x_j||^2
            # So dE/dx_i = -sum_{j != i} (x_i - x_j)/||x_i - x_j||^2
            grad[i] -= d / dist_sq
    
    # Project gradient to tangent space
    for i in range(N):
        grad_dot = np.dot(grad[i], pts[i])
        grad[i] -= grad_dot * pts[i]
    
    return energy, grad


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using L-BFGS-B with custom gradient
    # We'll use a wrapper that handles the sphere constraint by re-projecting
    # and computing the Riemannian gradient
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        energy, grad = _energy_and_grad(pts)
        # The gradient we return should be for the unconstrained variables
        # Since we project, we need to be careful. Let's use a penalty-free approach:
        # Just return the tangent gradient, and let the optimizer work in Cartesian.
        # Actually, this is tricky. Let's use a different approach.
        return energy, grad.flatten()
    
    # Alternative: use scipy minimize with the projected gradient
    # Let's try a simpler approach: iterate with gradient descent with re-projection
    
    best_pts = x0.copy()
    best_energy = _log_energy(best_pts)
    
    # Run L-BFGS-B
    try:
        result = minimize(
            lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
            x0.flatten(),
            jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if result.success or result.fun < best_energy:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < best_energy:
                best_pts = candidate
                best_energy = cand_energy
    except Exception:
        pass
    
    # Additional local refinement with smaller steps
    for _ in range(3):
        try:
            result = minimize(
                lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
                best_pts.flatten(),
                jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            if result.fun < best_energy:
                candidate = _project_to_sphere(result.x.reshape(N, 3))
                cand_energy = _log_energy(candidate)
                if cand_energy < best_energy:
                    best_pts = candidate
                    best_energy = cand_energy
        except Exception:
            pass
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close together
    dists = cdist(best_pts, best_pts)
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        # Perturb slightly
        rng = np.random.default_rng(42)
        best_pts = best_pts + rng.normal(0, 1e-6, best_pts.shape)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
