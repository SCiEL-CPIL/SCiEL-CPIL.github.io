"""
Strategy: Riemannian Newton method with explicit Hessian for logarithmic energy on the sphere.
Uses second-order curvature information to take larger, more informed steps than gradient-based
methods. The Newton direction solves H_p * delta = -g_p where H_p is the Riemannian Hessian
projected onto the tangent space. A conjugate-gradient solver with adaptive damping ensures
robustness. Final L-BFGS-B polish extracts last bits of precision.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# Precompute pair indices
_idx_i, _idx_j = np.triu_indices(N, k=1)
_n_pairs = len(_idx_i)


def _energy(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    dot = X @ X.T
    dist_sq = 2.0 - 2.0 * dot
    d2 = dist_sq[_idx_i, _idx_j]
    d2 = np.maximum(d2, 1e-18)
    return -0.5 * math.fsum(np.log(d2))


def _energy_and_tangent_grad(X):
    """
    Compute energy and tangent-space gradient for logarithmic energy on sphere.
    X: (N, 3) array of unit vectors
    Returns: (energy, grad_tangent) where grad_tangent is (N,3)
    """
    dot = X @ X.T
    dist_sq = 2.0 - 2.0 * dot
    d2 = dist_sq[_idx_i, _idx_j]
    d2 = np.maximum(d2, 1e-18)
    
    energy = -0.5 * math.fsum(np.log(d2))
    
    diff = X[_idx_i] - X[_idx_j]
    inv_d2 = 1.0 / d2
    
    # Euclidean gradient of sum log(1/|x_i-x_j|) = sum 1/|x_i-x_j|^2 * (x_i - x_j)
    # Actually d/dx_i log(1/|x_i-x_j|) = (x_i - x_j) / |x_i-x_j|^2
    grad_euclidean = np.zeros((N, 3))
    np.add.at(grad_euclidean, _idx_i, diff * inv_d2[:, np.newaxis])
    np.add.at(grad_euclidean, _idx_j, -diff * inv_d2[:, np.newaxis])
    
    # Project to tangent space
    dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot_grad * X
    
    return energy, grad_tangent


def _riemannian_hessian_vecprod(X, V):
    """
    Compute Hessian-vector product for the Riemannian Hessian of the log energy.
    
    For f(x) = sum_{i<j} log(1/|x_i - x_j|), the Riemannian Hessian applied to a
    tangent vector V at X is:
    
    (H_X V)_k = sum_{j != k} [ (I - x_k x_k^T) ( (x_k - x_j)/|x_k-x_j|^2 * (V_k - V_j)
            + (x_k - x_j)/|x_k-x_j|^4 * ( (x_k-x_j)^T V_j - (x_k-x_j)^T V_k ) ) ]
    
    More precisely, the Riemannian Hessian of f at x in direction v is:
    H_X(v) = P_x [ sum_{j} D^2 f(x)_xj * v ] where we need to be careful.
    
    Let's use the formula: for f(x) = sum g_ij(x_i, x_j), the Riemannian Hessian
    applied to tangent vector V is:
    (H V)_i = P_i [ sum_j (D^2_{ii} g * V_i + D^2_{ij} g * V_j) - (grad_i f)^T V_i * x_i ]
    
    For g_ij = log(1/|x_i - x_j|):
    D^2_{ii} g = I/|x_i-x_j|^2 - (x_i-x_j)(x_i-x_j)^T / |x_i-x_j|^4
    D^2_{ij} g = -(x_i-x_j)(x_i-x_j)^T / |x_i-x_j|^4
    
    So sum_j D^2_{ij} g * V_j contributes to the Hessian.
    
    Let me implement this carefully.
    """
    dot = X @ X.T
    dist_sq = 2.0 - 2.0 * dot
    d2 = np.maximum(dist_sq, 1e-18)
    
    diff = X[_idx_i] - X[_idx_j]  # (n_pairs, 3)
    inv_d2 = 1.0 / d2
    inv_d4 = inv_d2 * inv_d2
    
    # For each pair (i, j) with i < j:
    # Contribution to Hessian-vector product at index i:
    # P_i [ (I * inv_d2 - diff*diff^T * inv_d4) * V_i - diff*diff^T * inv_d4 * V_j ]
    # Contribution at index j:
    # P_j [ (I * inv_d2 - diff*diff^T * inv_d4) * V_j + diff*diff^T * inv_d4 * V_i ]
    
    V_i = V[_idx_i]  # (n_pairs, 3)
    V_j = V[_idx_j]  # (n_pairs, 3)
    
    # diff * inv_d2 (shape n_pairs, 3)
    diff_inv_d2 = diff * inv_d2[:, np.newaxis]
    
    # diff*diff^T * inv_d4 * V: for each pair, (diff @ diff.T) @ V = diff * (diff @ V) * inv_d4
    diff_dot_Vi = np.sum(diff * V_i, axis=1)  # (n_pairs,)
    diff_dot_Vj = np.sum(diff * V_j, axis=1)  # (n_pairs,)
    
    # Hessian contribution at i before projection:
    # inv_d2 * V_i - diff * diff_dot_Vi * inv_d4 - diff * diff_dot_Vj * inv_d4
    # = inv_d2 * V_i - diff * (diff_dot_Vi + diff_dot_Vj) * inv_d4
    Hv_i = inv_d2[:, np.newaxis] * V_i - diff * (diff_dot_Vi + diff_dot_Vj)[:, np.newaxis] * inv_d4
    
    # Hessian contribution at j before projection:
    # inv_d2 * V_j - diff * diff_dot_Vj * inv_d4 + diff * diff_dot_Vi * inv_d4
    # = inv_d2 * V_j + diff * (diff_dot_Vi - diff_dot_Vj) * inv_d4
    Hv_j = inv_d2[:, np.newaxis] * V_j + diff * (diff_dot_Vi - diff_dot_Vj)[:, np.newaxis] * inv_d4
    
    # Accumulate
    H_V = np.zeros((N, 3))
    np.add.at(H_V, _idx_i, Hv_i)
    np.add.at(H_V, _idx_j, Hv_j)
    
    # Project to tangent space: P_k = I - x_k x_k^T
    dot_HV = np.sum(H_V * X, axis=1, keepdims=True)
    H_V_tangent = H_V - dot_HV * X
    
    return H_V_tangent


def _riemannian_newton_step(X, max_iter_cg=200, tol_cg=1e-10, deadline=None):
    """
    Compute a Riemannian Newton step using conjugate gradient on the Hessian.
    Solves H_X * delta = -g_X for delta in the tangent space.
    Returns (delta, success) where delta is the Newton direction (tangent vector).
    """
    E, g = _energy_and_tangent_grad(X)
    
    # If gradient is very small, we're at a minimum
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-12:
        return np.zeros_like(X), True
    
    # Conjugate gradient to solve H * delta = -g
    # H is the Riemannian Hessian (symmetric positive definite at a local min)
    b = -g  # (N, 3)
    
    # Initialize
    r = b.copy()
    p = r.copy()
    rs_old = np.sum(r * r)
    
    delta = np.zeros_like(X)
    
    for k in range(max_iter_cg):
        if deadline is not None and time.time() > deadline:
            break
        
        Hp = _riemannian_hessian_vecprod(X, p)
        p_Hp = np.sum(p * Hp)
        
        if abs(p_Hp) < 1e-16:
            # Hessian is singular or nearly so in this direction
            break
        
        alpha = rs_old / p_Hp
        
        # Damping: if alpha is too large, reduce it
        if abs(alpha) > 0.1:
            alpha = np.sign(alpha) * 0.1
        
        delta_new = delta + alpha * p
        r_new = r + alpha * Hp
        rs_new = np.sum(r_new * r_new)
        
        if rs_new < 0:
            # Hessian not positive definite, break
            break
        
        beta = rs_new / rs_old if rs_old > 0 else 0
        p = r_new + beta * p
        r = r_new
        rs_old = rs_new
        
        if math.sqrt(rs_new) < tol_cg * (np.sum(b * b) ** 0.5 + 1e-16):
            break
    
    # Project delta to tangent space
    dot_delta = np.sum(delta * X, axis=1, keepdims=True)
    delta = delta - dot_delta * X
    
    # Check if CG converged
    if rs_old < (tol_cg ** 2) * (np.sum(b * b) + 1e-16):
        return delta, True
    else:
        return delta, False


def _damped_newton_iteration(X, deadline, max_iter=100):
    """
    Riemannian Newton method with adaptive damping.
    """
    E_old = _energy(X)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute Newton direction
        delta, cg_ok = _riemannian_newton_step(X, max_iter_cg=100, deadline=deadline)
        
        if not cg_ok:
            # Fallback to steepest descent direction
            _, g = _energy_and_tangent_grad(X)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-14:
                break
            delta = -g / g_norm
        
        # Line search with backtracking
        E = E_old
        step_size = 1.0
        X_new = X
        found = False
        
        for ls in range(30):
            if time.time() > deadline:
                break
            
            # Exponential map: X_new = cos(step) * X + sin(step) * (delta / |delta|)
            d_norm = np.linalg.norm(delta)
            if d_norm < 1e-14:
                break
            
            d_unit = delta / d_norm
            actual_step = step_size * d_norm
            cos_s = math.cos(actual_step)
            sin_s = math.sin(actual_step)
            X_candidate = cos_s * X + sin_s * d_unit
            norms = np.linalg.norm(X_candidate, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_candidate = X_candidate / norms
            
            E_candidate = _energy(X_candidate)
            
            if E_candidate < E_old - 1e-16:
                X_new = X_candidate
                E = E_candidate
                found = True
                break
            else:
                step_size *= 0.5
        
        if not found:
            # Try a smaller step in the steepest descent direction
            _, g = _energy_and_tangent_grad(X)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-14:
                break
            d_unit = -g / g_norm
            for ls in range(20):
                actual_step = step_size * 0.1
                cos_s = math.cos(actual_step)
                sin_s = math.sin(actual_step)
                X_candidate = cos_s * X + sin_s * d_unit
                norms = np.linalg.norm(X_candidate, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X_candidate = X_candidate / norms
                E_candidate = _energy(X_candidate)
                if E_candidate < E_old - 1e-16:
                    X_new = X_candidate
                    E = E_candidate
                    found = True
                    break
                step_size *= 0.5
        
        if not found:
            break
        
        # Check convergence
        if abs(E_old - E) < 1e-14:
            X = X_new
            break
        
        X = X_new
        E_old = E
    
    return X


def _lbfgs_polish(X, deadline, max_iter=200):
    """Final L-BFGS-B polish for high precision."""
    def obj_and_grad(V_flat, X_init=X):
        if time.time() > deadline:
            E, _ = _energy_and_tangent_grad(X_init)
            return E, np.zeros(N * 3)
        
        V = V_flat.reshape(N, 3)
        X_new = X_init + V
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_new = X_new / norms
        E, grad_tan = _energy_and_tangent_grad(X_new)
        return E, grad_tan.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            np.zeros(N * 3),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 2,
                'ftol': 1e-18,
                'gtol': 1e-15,
                'maxls': 200,
                'iprint': -1,
            }
        )
        V_opt = result.x.reshape(N, 3)
        X_opt = X + V_opt
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_opt = X_opt / norms
    except Exception:
        X_opt = X
    
    return X_opt


def _fibonacci_init(rng, offset=0.0):
    """Generate Fibonacci sphere initialization with random rotation."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    X = X @ Q.T
    return X


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_X = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    starts = []
    for i in range(5):
        starts.append(_fibonacci_init(rng, offset=float(i) * 0.15))
    
    # Add a few random starts
    for _ in range(3):
        X_rand = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
        starts.append(X_rand / norms)
    
    # Phase 1: Riemannian Newton optimization from each start
    for s, X_init in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 20.0:
            break
        
        # Allocate time for Newton iterations
        newton_deadline = min(deadline - 10.0, elapsed + remaining / max(1, len(starts) - s))
        
        X_opt = _damped_newton_iteration(X_init, newton_deadline, max_iter=80)
        
        E_opt = _energy(X_opt)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
    
    # Phase 2: L-BFGS-B polish of the best solution
    if best_X is not None and time.time() < deadline - 10.0:
        X_polished = _lbfgs_polish(best_X, deadline - 5.0, max_iter=300)
        E_polished = _energy(X_polished)
        if E_polished < best_energy:
            best_energy = E_polished
            best_X = X_polished
    
    # Phase 3: Another Newton round from the polished solution
    if best_X is not None and time.time() < deadline - 15.0:
        X_newton2 = _damped_newton_iteration(best_X, deadline - 10.0, max_iter=50)
        E_newton2 = _energy(X_newton2)
        if E_newton2 < best_energy:
            best_energy = E_newton2
            best_X = X_newton2
    
    # Phase 4: Final L-BFGS-B polish
    if best_X is not None and time.time() < deadline - 5.0:
        X_final = _lbfgs_polish(best_X, deadline - 2.0, max_iter=200)
        E_final = _energy(X_final)
        if E_final < best_energy:
            best_energy = E_final
            best_X = X_final
    
    # Fallback
    if best_X is None:
        best_X = _fibonacci_init(rng)
    
    # Final normalization and validation
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_init(rng)
    
    return best_X
