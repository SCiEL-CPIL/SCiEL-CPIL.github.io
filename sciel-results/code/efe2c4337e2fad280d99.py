"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _log_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    x = np.asarray(x)
    n = x.shape[0]
    # Compute all pairwise distances
    # x has shape (n, 3)
    # distances[i,j] = |x_i - x_j|
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dist, 1.0)
    # Energy is sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    return -np.sum(np.log(d))

def _log_energy_grad(x):
    """Compute gradient of logarithmic energy on the sphere.
    
    The energy is E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    For a pair (i,j), the gradient of -log(|x_i - x_j|) with respect to x_i is:
        (x_i - x_j) / |x_i - x_j|^2
    And with respect to x_j is:
        (x_j - x_i) / |x_i - x_j|^2
    
    But we need to project onto the tangent space of the sphere at each point.
    For a point x_i on the unit sphere, the tangent space is {v : v · x_i = 0}.
    The projection of a vector g onto the tangent space is g - (g · x_i) x_i.
    
    However, since we're optimizing on the manifold, we can compute the Euclidean
    gradient and then project. But for L-BFGS-B which operates in Euclidean space,
    we need to be careful. Actually, let's use a different approach:
    
    We'll optimize in R^3 with a constraint that |x_i| = 1. But L-BFGS-B doesn't
    handle equality constraints well. Instead, let's use the fact that the gradient
    on the sphere is the tangential component.
    
    For L-BFGS-B to work, we'll compute the Euclidean gradient of the energy,
    but since we're on the sphere, we need to project. Actually, the standard
    approach is to compute the gradient as if in R^3, and the optimizer will
    naturally stay near the sphere if we initialize there and the gradient
    points inward/outward appropriately. But this is not guaranteed.
    
    A better approach: use the tangent space gradient. For each point x_i,
    the gradient of the energy with respect to x_i (in the ambient R^3) is:
        g_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (from the pair (i,j), the contribution to x_i's gradient is (x_i - x_j)/|x_i-x_j|^2)
    
    Then the Riemannian gradient (tangent component) is:
        grad_i = g_i - (g_i · x_i) x_i
    
    But L-BFGS-B doesn't know about the manifold. So we'll use a trick:
    we'll compute the Euclidean gradient, and rely on the fact that for points
    on the sphere, the radial component of the gradient is related to the
    constraint. Actually, the cleanest way is to use scipy's minimize with
    the method 'L-BFGS-B' and provide the gradient, but we need to ensure
    points stay on the sphere.
    
    Alternative: use a penalty or re-normalize after each step. But that's
    not differentiable.
    
    Let's use the following approach: compute the Euclidean gradient of the
    energy function. Since the energy is defined for points in R^3 (not just
    on the sphere), and we initialize on the sphere, the optimizer will
    move points. We'll add a small penalty for |x_i|^2 - 1 to keep them
    on the sphere. But that changes the problem.
    
    Actually, let's reconsider. The logarithmic energy is well-defined for
    any points in R^3 as long as no two points coincide. The constraint is
    that they lie on the unit sphere. So we can use L-BFGS-B with bounds
    or with a penalty.
    
    A common technique: use the fact that for points on the sphere, the
    gradient of the energy projected onto the tangent space gives the
    direction of steepest descent on the manifold. We can implement a
    custom step: take a step in the negative gradient direction, then
    project back to the sphere. But L-BFGS-B doesn't support this.
    
    Let's use a different optimizer that supports manifold optimization,
    or use a penalty method.
    
    For simplicity and to use L-BFGS-B, let's use a penalty approach:
    E_total(x) = E_log(x) + lambda * sum_i (|x_i|^2 - 1)^2
    
    But this may not be robust.
    
    Alternative: use scipy.optimize.minimize with method='SLSQP' which
    supports equality constraints. But SLSQP is slower.
    
    Let's try a hybrid: use L-BFGS-B on the tangent space. We can
    parameterize each point by its tangent space coordinates. But that's
    complex.
    
    Actually, for the purpose of this task, let's use the Euclidean gradient
    and hope that the optimizer stays near the sphere. We'll initialize on
    the sphere and use a small step size. If points drift off the sphere,
    we can re-normalize at the end.
    
    Let's compute the Euclidean gradient:
    For E = -sum_{i<j} log(|x_i - x_j|),
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    This is the standard gradient. Let's use this with L-BFGS-B.
    """
    x = np.asarray(x)
    n = x.shape[0]
    grad = np.zeros_like(x)
    
    # Compute pairwise distances and gradients
    for i in range(n):
        for j in range(i+1, n):
            diff = x[i] - x[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-20:
                # Avoid division by zero
                continue
            d = np.sqrt(d2)
            # Gradient contribution to x_i: diff / d^2 = diff / d2
            # Gradient contribution to x_j: -diff / d^2
            g = diff / d2
            grad[i] += g
            grad[j] -= g
    
    return grad

def _fibonacci_sphere():
    """Generate Fibonacci sphere points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    """Generate random points on the unit sphere."""
    x = rng.standard_normal((N, 3))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _optimize_from(x0, time_limit):
    """Optimize from initial configuration x0, returning best point and energy."""
    start_time = time.time()
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    try:
        result = minimize(
            _log_energy,
            x0,
            jac=_log_energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        if result.success or result.fun < best_e:
            best_x = result.x
            best_e = result.fun
    except Exception:
        pass
    
    # Re-normalize to ensure points are on the unit sphere
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x, best_e

def construct():
    """Main entry point: multi-start L-BFGS-B optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Fibonacci sphere
    initial_configs.append(_fibonacci_sphere())
    
    # 2. Random configurations
    for _ in range(5):
        initial_configs.append(_random_sphere(rng))
    
    # 3. Jittered Fibonacci
    fib = _fibonacci_sphere()
    jitter = rng.standard_normal(fib.shape) * 0.05
    jittered = fib + jitter
    norms = np.linalg.norm(jittered, axis=1, keepdims=True)
    initial_configs.append(jittered / norms)
    
    best_x = None
    best_e = np.inf
    
    for x0 in initial_configs:
        if time.time() >= deadline:
            break
        
        x_opt, e_opt = _optimize_from(x0, deadline)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
    
    # If no optimization was run (shouldn't happen), use Fibonacci
    if best_x is None:
        best_x = _fibonacci_sphere()
    
    # Final re-normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x
# EVOLVE-BLOCK-END
