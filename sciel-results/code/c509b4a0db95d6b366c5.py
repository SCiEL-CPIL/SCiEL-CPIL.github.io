"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START
def _log_energy(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # X is (N, 3)
    # Compute pairwise distances
    # For efficiency, use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _log_energy_grad(X):
    """Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    grad = np.zeros_like(X)
    # For each pair i < j:
    # dE/dx_i += (x_i - x_j) / |x_i - x_j|^2
    # dE/dx_j += (x_j - x_i) / |x_i - x_j|^2
    for i in range(N):
        for j in range(i + 1, N):
            diff = X[i] - X[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-30:
                continue
            d = math.sqrt(d2)
            inv_d2 = 1.0 / d2
            grad[i] += diff * inv_d2
            grad[j] -= diff * inv_d2
    return grad


def _log_energy_grad_fast(X):
    """Vectorized gradient computation."""
    # diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid division by zero
    d2 = np.maximum(d2, 1e-30)
    inv_d2 = 1.0 / d2
    # Set diagonal to 0
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_j (X[i] - X[j]) / |X[i] - X[j]|^2
    # = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _golden_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _momentum_step(X, grad, v, lr, momentum, beta):
    """Momentum step with projection."""
    v = momentum * v - lr * grad
    X_new = X + v
    X_new = _project_to_sphere(X_new)
    return X_new, v


def _armijo_line_search(X, grad, direction, f0, c1=1e-4, rho=0.5, max_iter=30):
    """Armijo backtracking line search for sphere-constrained optimization.
    We move along the geodesic direction (tangent projected).
    """
    # Project direction to tangent space at X
    # For sphere constraint, tangent space at x is {v : x·v = 0}
    # direction is already in ambient space; project to tangent
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / np.maximum(X_norm, 1e-15)
    # Project direction to tangent
    dot = np.sum(X_unit * direction, axis=1, keepdims=True)
    direction_tangent = direction - X_unit * dot
    
    # Use exponential map (rotation) for sphere
    # For small steps, just do X_new = normalize(X + alpha * direction_tangent)
    alpha = 1.0
    for _ in range(max_iter):
        X_new = X + alpha * direction_tangent
        X_new = _project_to_sphere(X_new)
        f_new = _log_energy(X_new)
        # Check finite
        if not np.isfinite(f_new):
            alpha *= rho
            continue
        slope = np.sum(grad * direction_tangent)
        if f_new <= f0 + c1 * alpha * slope:
            return X_new, alpha
        alpha *= rho
    return X, alpha


def construct():
    start_time = time.time()
    
    # Start with golden spiral
    X = _golden_spiral()
    best_X = X.copy()
    best_E = _log_energy(X)
    
    # First pass: scipy L-BFGS-B style but we do our own since sphere constraint
    # Use gradient descent with momentum and Armijo
    
    # Try multiple initializations
    inits = [X.copy()]
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() - start_time > DEADLINE - 20:
            break
        inits.append(_random_sphere(N, seed))
    
    for init_idx, X0 in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        X = X0.copy()
        E = _log_energy(X)
        
        # Gradient descent with momentum
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(X)
        
        max_iter = 5000 if init_idx == 0 else 2000
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            grad = _log_energy_grad_fast(X)
            
            # Momentum update
            v = momentum * v - lr * grad
            X_new = X + v
            X_new = _project_to_sphere(X_new)
            
            E_new = _log_energy(X_new)
            
            if np.isfinite(E_new) and E_new < E:
                X = X_new
                E = E_new
                # Decay learning rate slowly
                if it % 500 == 0 and it > 0:
                    lr *= 0.95
            else:
                # Reject step, reduce learning rate
                lr *= 0.5
                v = 0.5 * v
                X_new = X + v
                X_new = _project_to_sphere(X_new)
                E_new = _log_energy(X_new)
                if np.isfinite(E_new) and E_new < E:
                    X = X_new
                    E = E_new
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    # Final polish: use scipy minimize with very tight tolerances
    if time.time() - start_time < DEADLINE - 15:
        # Use scipy's L-BFGS-B with penalty method for sphere constraint
        # Or use trust-constr
        from scipy.optimize import minimize as sp_minimize
        
        def obj_and_grad(x_flat):
            X = x_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            f = _log_energy(X)
            g = _log_energy_grad_fast(X)
            # Project gradient to tangent
            X_unit = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-15)
            dot = np.sum(X_unit * g, axis=1, keepdims=True)
            g_tangent = g - X_unit * dot
            return f, g_tangent.ravel()
        
        x0 = best_X.ravel()
        
        def callback(xk):
            return time.time() - start_time > DEADLINE - 3
        
        try:
            res = sp_minimize(
                lambda x: obj_and_grad(x)[0],
                x0,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-18,
                    'gtol': 1e-14,
                    'callback': callback
                }
            )
            if res.success and np.isfinite(res.fun):
                X_res = _project_to_sphere(res.x.reshape(N, 3))
                E_res = _log_energy(X_res)
                if E_res < best_E:
                    best_E = E_res
                    best_X = X_res
        except Exception:
            pass
    
    # Final check and return
    best_X = _project_to_sphere(best_X)
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Perturb slightly
                best_X[j] += 1e-8 * np.random.randn(3)
                best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
