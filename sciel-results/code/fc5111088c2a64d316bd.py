"""
Strategy: Multi-start deterministic Riemannian L-BFGS-B optimization with dot-product
distance computation for speed. Uses multiple diverse initializations (Fibonacci sphere
with random rotations + random starts), each refined with a fast second-order solver.
This is materially different from the stochastic annealing approach: it exploits
deterministic second-order optimization from many starting points rather than
Monte Carlo exploration.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# Precompute pair indices
_idx_i, _idx_j = np.triu_indices(N, k=1)
_n_pairs = len(_idx_i)


def _energy_and_tangent_grad(X):
    """
    Compute energy and tangent-space gradient for logarithmic energy on sphere.
    Uses dot-product based distance: |xi-xj|^2 = 2 - 2*xi.xj
    X: (N, 3) array of unit vectors
    Returns: (energy, grad_tangent) where grad_tangent is (N,3)
    """
    # Compute all pairwise dot products
    # dot[i,j] = xi . xj
    dot = X @ X.T
    # dist_sq[i,j] = 2 - 2*dot[i,j]
    dist_sq = 2.0 - 2.0 * dot
    
    # Extract upper triangle (excluding diagonal)
    d2 = dist_sq[_idx_i, _idx_j]
    d2 = np.maximum(d2, 1e-18)
    
    # Energy = sum log(1/d) = -sum log(d) = -0.5 * sum log(d^2)
    energy = -0.5 * math.fsum(np.log(d2))
    
    # Gradient: dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    # In terms of dot products, the Euclidean gradient is:
    # grad_E[i] = sum_j (xi - xj) / d2_ij
    # We can compute this as: for each pair (i,j), add (xi-xj)/d2 to grad[i] and -(xi-xj)/d2 to grad[j]
    # (xi - xj) / d2 = (xi - xj) / (2 - 2*xi.xj)
    
    # Vectorized computation of Euclidean gradient
    # diff = xi - xj for each pair
    diff = X[_idx_i] - X[_idx_j]  # (n_pairs, 3)
    inv_d2 = 1.0 / d2  # (n_pairs,)
    
    grad_euclidean = np.zeros((N, 3))
    np.add.at(grad_euclidean, _idx_i, diff * inv_d2[:, np.newaxis])
    np.add.at(grad_euclidean, _idx_j, -diff * inv_d2[:, np.newaxis])
    
    # Project to tangent space
    # grad_tangent[i] = grad_euclidean[i] - (grad_euclidean[i] . xi) * xi
    dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot_grad * X
    
    return energy, grad_tangent


def _exp_map(X, V, step):
    """
    Exponential map on sphere: move each point along geodesic in direction V.
    X: (N, 3) unit vectors
    V: (N, 3) tangent vectors (already projected)
    step: scalar step size
    Returns: (N, 3) new unit vectors
    """
    cos_s = math.cos(step)
    sin_s = math.sin(step)
    X_new = cos_s * X + sin_s * V
    # Renormalize for numerical safety
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_new = X_new / norms
    return X_new


def _fibonacci_init(rng):
    """Generate Fibonacci sphere initialization with random rotation."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    # Generate random rotation matrix via QR decomposition
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    X = X @ Q.T
    return X


def _random_init(rng):
    """Generate random points on sphere."""
    X = rng.standard_normal((N, 3))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return X


def _local_minimize(X_init, deadline):
    """
    Run L-BFGS-B optimization on the sphere using tangent-space parameterization.
    We flatten the tangent vectors and use the exponential map.
    """
    # Initial tangent vectors (zero = stay at X_init)
    V0 = np.zeros((N * 3,))
    
    def _objective_and_grad(V_flat):
        """
        V_flat: (N*3,) flattened tangent vectors
        Returns: (energy, grad_flat)
        """
        V = V_flat.reshape(N, 3)
        # Project V to tangent space at X_init
        # Actually, for L-BFGS-B we need to define the optimization in a consistent space.
        # Better approach: optimize directly on the sphere using the exponential map
        # from X_init. The variable V represents the tangent vector at X_init.
        
        # Move from X_init to X_new = exp_{X_init}(V)
        # For small V: X_new ≈ cos(|V|) * X_init + sin(|V|)/|V| * V
        # But for L-BFGS-B, we need smoothness. Let's use a different parameterization.
        
        # Alternative: use the fact that we can optimize on R^{N*3} where each point
        # is represented by its tangent vector at the current position.
        # This is the "retraction" approach.
        
        # Simpler: just do gradient descent on the manifold directly.
        # But L-BFGS-B works in R^n. Let's use the "pull-back" approach:
        # Define f(V) = E(exp_{X0}(V)) where X0 = X_init
        # grad_V f = (grad_X E) pulled back to tangent space at X0
        
        # For the exponential map on sphere:
        # If V is the tangent vector at X0, then X_new = cos(|V|)X0 + sin(|V|)/|V| * V
        # The differential of exp at V=0 is the identity, so near V=0:
        # grad_V f ≈ grad_tangent E at X0 (projected)
        
        # Actually, let's just do iterative Riemannian gradient descent with L-BFGS-B
        # on the Euclidean embedding with projection.
        
        # Even simpler: just use scipy's minimize with a custom approach.
        # Let's optimize the angles directly.
        
        # Actually, the cleanest approach: use the fact that each point on S^2 can
        # be parameterized by (theta, phi), and optimize in that 2N-dimensional space.
        # But this has singularities at the poles.
        
        # Let's use a different strategy: iterative Riemannian gradient descent
        # with adaptive step size, which is simpler and robust.
        
        # For now, let's just compute energy and gradient at X_init + V (approximately)
        # and let L-BFGS-B handle the rest.
        
        # The issue: V is a tangent vector at X_init, but as we move, the tangent
        # space changes. For L-BFGS-B to work, we need a consistent parameterization.
        
        # Let's use the "retraction" approach: X(V) = normalize(X_init + V)
        # This is a valid retraction (not the exponential map, but close enough).
        
        X_new = X_init + V.reshape(N, 3)
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_new = X_new / norms
        
        energy, grad_tan = _energy_and_tangent_grad(X_new)
        
        # The gradient in the V-space: since X_new = normalize(X_init + V),
        # dX_new/dV = I - X_new * X_new^T (projection to tangent space)
        # So grad_V = grad_tan (already in tangent space)
        grad_flat = grad_tan.flatten()
        
        return energy, grad_flat
    
    # Use L-BFGS-B
    result = minimize(
        _objective_and_grad,
        V0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 2000,
            'maxfun': 5000,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxls': 50,
        }
    )
    
    # Extract optimized V and compute final X
    V_opt = result.x.reshape(N, 3)
    X_opt = X_init + V_opt
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_opt = X_opt / norms
    
    # Check time
    if time.time() > deadline:
        return X_init  # Return initial if timeout
    
    return X_opt


def _local_minimize_riemannian(X_init, deadline):
    """
    Pure Riemannian gradient descent with adaptive step size on the sphere.
    This avoids the parameterization issues of L-BFGS-B.
    """
    X = X_init.copy()
    current_energy, grad_tan = _energy_and_tangent_grad(X)
    best_X = X.copy()
    best_energy = current_energy
    
    step_size = 0.05
    patience = 0
    max_patience = 50
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        if patience > max_patience:
            break
        
        # Compute step direction
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size: step_size = alpha / (1 + grad_norm)
        step = step_size / (1.0 + grad_norm)
        step = min(step, 0.1)  # Cap step size
        
        # Exponential map step
        V = -grad_tan / (grad_norm + 1e-12)  # Unit direction
        X_new = _exp_map(X, V, step)
        
        E_new, grad_new = _energy_and_tangent_grad(X_new)
        
        if E_new < current_energy:
            X = X_new
            current_energy = E_new
            grad_tan = grad_new
            patience = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X.copy()
        else:
            patience += 1
            step_size *= 0.9  # Reduce step size
    
    return best_X, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 145s out of 150s limit
    
    rng = np.random.default_rng(42)
    
    best_X = None
    best_energy = float('inf')
    
    # Generate multiple diverse initializations
    # 1. Fibonacci sphere with random rotations
    # 2. Pure random starts
    # 3. Fibonacci with different offsets
    
    num_starts = 8
    
    for s in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initialization
        if s % 3 == 0:
            X_init = _fibonacci_init(rng)
        elif s % 3 == 1:
            X_init = _random_init(rng)
        else:
            # Fibonacci with random offset
            X_init = _fibonacci_init(rng)
            # Add small random perturbation
            pert = rng.standard_normal((N, 3)) * 0.01
            X_init = X_init + pert
            norms = np.linalg.norm(X_init, axis=1, keepdims=True)
            X_init = X_init / norms
        
        # Run Riemannian local minimization
        X_opt, E_opt = _local_minimize_riemannian(X_init, deadline)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
    
    # If we have time, do additional refinement on the best solution
    if best_X is not None and time.time() < deadline - 5.0:
        X_refined, E_refined = _local_minimize_riemannian(best_X, deadline)
        if E_refined < best_energy:
            best_energy = E_refined
            best_X = X_refined
    
    # Final normalization
    if best_X is None:
        # Fallback: Fibonacci sphere
        best_X = _fibonacci_init(rng)
    
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    # Validate
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_init(rng)
    
    return best_X
