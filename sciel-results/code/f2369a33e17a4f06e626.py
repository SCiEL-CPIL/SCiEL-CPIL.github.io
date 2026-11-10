# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial

K = 10
N_THETA = 70
N_PHI = 130
DIM = 2 * K + 1  # 21

# Precompute quadrature weights and nodes
_x, _w = np.polynomial.legendre.leggauss(N_THETA)
_cos_theta = _x
_theta_weights = _w * 0.5

# phi grid
_phi = np.arange(N_PHI) * (2.0 * np.pi / N_PHI)

# Precompute associated Legendre functions P_l^m(cos theta) with Condon-Shortley phase
_P = {}
for m in range(K + 1):
    _P[m] = lpmv(m, K, _cos_theta)

# Normalization constants N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm_const(m):
    m = abs(m)
    val = 21.0 / (4.0 * np.pi) * float(factorial(10 - m, exact=False) / factorial(10 + m, exact=False))
    return np.sqrt(val)

# Precompute Y_m values on the grid
_basis = np.zeros((DIM, N_THETA, N_PHI), dtype=np.float64)
for idx, m in enumerate(range(-K, K + 1)):
    am = abs(m)
    norm = _norm_const(am)
    Pm = _P[am]
    if m == 0:
        _basis[idx, :, :] = norm * Pm[:, None]
    elif m > 0:
        cos_term = np.cos(m * _phi)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * cos_term[None, :]
    else:
        sin_term = np.sin(am * _phi)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * sin_term[None, :]

# Quadrature weights on the sphere
_w_phi = 2.0 * np.pi / N_PHI
_sphere_weights = np.outer(_theta_weights, np.full(N_PHI, _w_phi))
_sphere_weights_flat = _sphere_weights.flatten()
N_GRID = N_THETA * N_PHI

# Flatten basis for matrix operations: shape (DIM, N_GRID)
_basis_flat = _basis.reshape(DIM, N_GRID)

def _eval_f(coeffs):
    """Evaluate f on the grid: returns array of shape (N_GRID,)"""
    return _basis_flat @ coeffs

def _compute_R(coeffs):
    """Compute ||f||_4 / ||f||_2 for given coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape != (DIM,):
        return np.inf
    if not np.all(np.isfinite(coeffs)):
        return np.inf
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.inf
    
    f = _eval_f(coeffs)
    
    # L2 norm squared
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    # L4 norm
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return np.inf
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm

def _objective_with_grad(coeffs):
    """
    Minimize ||f||_4 subject to ||f||_2 = 1.
    Returns (objective_value, gradient).
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    f = _eval_f(coeffs)
    
    # Current L2 norm
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return 1e10, np.zeros(DIM)
    l2_norm = np.sqrt(l2_sq)
    
    # Objective: minimize L4 norm (scale-invariant, so we can fix L2=1)
    # Actually, let's minimize L4/L2 directly via the ratio
    # But for gradient, let's work with: minimize ||f||_4^4 / ||f||_2^4
    # Or better: minimize ||f||_4 subject to ||f||_2 = 1
    
    # For gradient descent on the ratio, use:
    # R = ||f||_4 / ||f||_2
    # dR/dc = (1/R) * (d||f||_4/dc) - (||f||_4/||f||_2^2) * (d||f||_2/dc)
    # This is complex. Instead, minimize J = ||f||_4^4 / ||f||_2^4
    # Or simply: since R is scale invariant, minimize ||f||_4 with constraint ||f||_2 = 1
    
    # Let's use the approach: minimize ||f||_4^4 / (||f||_2^2)^2
    # This is scale invariant and smooth.
    
    l4_pow4 = np.sum(f ** 4 * _sphere_weights_flat)
    l2_sq_safe = max(l2_sq, 1e-30)
    J = l4_pow4 / (l2_sq_safe ** 2)
    
    # Gradient of J with respect to coeffs
    # dJ/dc_m = (1/l2^4) * d(l4^4)/dc_m - (2*l4^4/l2^6) * d(l2^2)/dc_m
    # d(l4^4)/dc_m = 4 * sum(f^3 * w * Y_m)
    # d(l2^2)/dc_m = 2 * sum(f * w * Y_m)
    
    f3w = f ** 3 * _sphere_weights_flat
    fw = f * _sphere_weights_flat
    
    grad_l4 = 4.0 * (_basis_flat.T @ f3w)
    grad_l2 = 2.0 * (_basis_flat.T @ fw)
    
    l2_sq_safe = max(l2_sq, 1e-30)
    grad_J = grad_l4 / (l2_sq_safe ** 2) - 2.0 * J * grad_l2 / l2_sq_safe
    
    return J, grad_J

def _projected_gradient_descent(start_coeffs, deadline, max_iter=5000, lr_init=0.01, lr_decay=0.999):
    """
    Projected gradient descent: minimize ||f||_4 subject to ||f||_2 = 1.
    After each step, project back to the unit L2 sphere.
    """
    x = np.array(start_coeffs, dtype=np.float64).copy()
    
    # Normalize to unit L2
    f = _eval_f(x)
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq > 0:
        x = x / np.sqrt(l2_sq)
    
    best_x = x.copy()
    best_J = _compute_R(x)  # Use actual R for comparison
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        J, grad = _objective_with_grad(x)
        
        # Gradient step
        x_new = x - lr * grad
        
        # Project to unit L2 sphere
        f_new = _eval_f(x_new)
        l2_sq_new = np.sum(f_new ** 2 * _sphere_weights_flat)
        if l2_sq_new > 0:
            x_new = x_new / np.sqrt(l2_sq_new)
        
        # Check improvement
        R_new = _compute_R(x_new)
        if R_new < best_J:
            best_x = x_new.copy()
            best_J = R_new
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= lr_decay
        
        x = x_new
    
    return best_x, best_J

def _lbfgs_optimize(start_coeffs, deadline):
    """LBFGS optimization with L2 constraint via projection."""
    from scipy.optimize import minimize
    
    def obj_and_grad(x):
        J, grad = _objective_with_grad(x)
        return J, grad
    
    x = np.array(start_coeffs, dtype=np.float64).copy()
    # Normalize to unit L2
    f = _eval_f(x)
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq > 0:
        x = x / np.sqrt(l2_sq)
    
    best_x = x.copy()
    best_R = _compute_R(x)
    
    try:
        if time.time() < deadline - 10:
            res = minimize(
                obj_and_grad, x, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500}
            )
            if np.all(np.isfinite(res.x)):
                # Project result to unit L2
                f_res = _eval_f(res.x)
                l2_sq_res = np.sum(f_res ** 2 * _sphere_weights_flat)
                if l2_sq_res > 0:
                    res.x = res.x / np.sqrt(l2_sq_res)
                R_res = _compute_R(res.x)
                if R_res < best_R:
                    best_x = res.x.copy()
                    best_R = R_res
    except Exception:
        pass
    
    return best_x, best_R

def _initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. All ones
    candidates.append(np.ones(DIM))
    
    # 2. Only m=0
    c = np.zeros(DIM)
    c[K] = 1.0
    candidates.append(c)
    
    # 3. Only highest |m|
    for sign in [1, -1]:
        c = np.zeros(DIM)
        idx = K + sign * K
        c[idx] = 1.0
        candidates.append(c)
    
    # 4. Alternating signs
    c = np.ones(DIM) * np.array([(-1) ** i for i in range(DIM)])
    candidates.append(c)
    
    # 5. Symmetric (even m only)
    c = np.zeros(DIM)
    for m in range(-K, K + 1, 2):
        c[m + K] = 1.0
    candidates.append(c)
    
    # 6. Antisymmetric (odd m only)
    c = np.zeros(DIM)
    for m in range(-K + 1, K + 1, 2):
        c[m + K] = 1.0
    candidates.append(c)
    
    # 7. Random normal
    for _ in range(30):
        c = rng.normal(size=DIM)
        candidates.append(c)
    
    # 8. Sparse random
    for _ in range(15):
        c = np.zeros(DIM)
        idxs = rng.choice(DIM, size=rng.integers(3, 8), replace=False)
        c[idxs] = rng.normal(size=len(idxs))
        candidates.append(c)
    
    # 9. Cosine-weighted
    m_vals = np.arange(-K, K + 1)
    candidates.append(np.cos(m_vals * 0.3))
    
    # 10. Sine-weighted
    candidates.append(np.sin(m_vals * 0.3))
    
    # 11. Exponentially decaying from center
    dist = np.abs(m_vals)
    candidates.append(np.exp(-0.2 * dist))
    
    # 12. Inverse of above
    candidates.append(1.0 / (1.0 + 0.5 * dist))
    
    # 13. Gaussian in m
    candidates.append(np.exp(-0.1 * m_vals ** 2))
    
    # 14. Polynomials in m
    for p in [2, 4]:
        candidates.append(1.0 + 0.1 * m_vals ** p)
    
    return candidates

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin
    
    # Phase 1: Generate and evaluate initial candidates
    candidates = _initial_candidates()
    best_coeffs = None
    best_R = np.inf
    
    for cand in candidates:
        if time.time() > deadline:
            break
        R = _compute_R(cand)
        if R < best_R:
            best_R = R
            best_coeffs = cand.copy()
    
    # Phase 2: Sort candidates by R and use top ones as starts
    scored = []
    for c in candidates:
        if time.time() > deadline - 30:
            break
        R = _compute_R(c)
        scored.append((R, c))
    scored.sort(key=lambda x: x[0])
    
    # Phase 3: Local optimization from multiple starts
    starts = [c for _, c in scored[:15]]
    
    # Add more diverse random starts
    rng = np.random.default_rng(123)
    for _ in range(15):
        c = rng.normal(size=DIM)
        starts.append(c)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        
        # Try projected gradient descent
        opt_x, opt_R = _projected_gradient_descent(start, deadline, max_iter=3000, lr_init=0.005)
        if opt_R < best_R:
            best_R = opt_R
            best_coeffs = opt_x.copy()
    
    # Phase 4: LBFGS from best solutions
    if best_coeffs is not None and time.time() < deadline - 60:
        # LBFGS from current best
        opt_x, opt_R = _lbfgs_optimize(best_coeffs, deadline)
        if opt_R < best_R:
            best_R = opt_R
            best_coeffs = opt_x.copy()
        
        # LBFGS from top few starts
        for _, c in scored[:5]:
            if time.time() > deadline - 30:
                break
            opt_x, opt_R = _lbfgs_optimize(c, deadline)
            if opt_R < best_R:
                best_R = opt_R
                best_coeffs = opt_x.copy()
    
    # Phase 5: Random perturbations around best and re-optimize
    if best_coeffs is not None:
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            noise_scale = 0.05 * np.linalg.norm(best_coeffs)
            perturbed = best_coeffs + rng.normal(0, noise_scale, size=DIM)
            opt_x, opt_R = _projected_gradient_descent(perturbed, deadline, max_iter=2000, lr_init=0.003)
            if opt_R < best_R:
                best_R = opt_R
                best_coeffs = opt_x.copy()
    
    # Phase 6: Final LBFGS polish
    if best_coeffs is not None and time.time() < deadline - 10:
        opt_x, opt_R = _lbfgs_optimize(best_coeffs, deadline)
        if opt_R < best_R:
            best_R = opt_R
            best_coeffs = opt_x.copy()
    
    # Validate final result
    if best_coeffs is None:
        best_coeffs = np.ones(DIM)
    
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    # Ensure validity
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(DIM)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1e-2
        else:
            best_coeffs = np.ones(DIM)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs
# EVOLVE-BLOCK-END
