# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time
from math import factorial, pi


def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid. Returns flattened basis matrix B and weights.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    N = np.zeros(11)
    for m in range(11):
        num = factorial(k - m)
        den = factorial(k + m)
        N[m] = np.sqrt((2 * k + 1) / (4 * pi) * (num / den))
        
    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    # m=0
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
        
    return B, weights


B_GLOBAL, W_GLOBAL = _precompute_basis()
N_PTS = B_GLOBAL.shape[0]

# Axisymmetric indices: m = 0, ±2, ±4, ±6, ±8, ±10
# In the 21-dim vector (index = m + 10):
# m=0 -> 10, m=2 -> 12, m=-2 -> 8, m=4 -> 14, m=-4 -> 6, m=6 -> 16, m=-6 -> 4, m=8 -> 18, m=-8 -> 2, m=10 -> 20, m=-10 -> 0
AXISYM_INDICES = np.array([10, 12, 8, 14, 6, 16, 4, 18, 2, 20, 0])
B_AXISYM = B_GLOBAL[:, AXISYM_INDICES]
N_AXISYM = len(AXISYM_INDICES)  # 11


def _expand_axisym(c_axisym):
    """Expand 11-dim axisymmetric coefficients to full 21-dim vector."""
    c_full = np.zeros(21)
    c_full[AXISYM_INDICES] = c_axisym
    return c_full


def _compute_ratio_from_f(f):
    """Compute R = ||f||_4 / ||f||_2 from function values f on the grid."""
    f2 = f * f
    norm2_sq = np.dot(W_GLOBAL, f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.dot(W_GLOBAL, f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_ratio(c_full):
    """Compute R for full 21-dim coefficient vector."""
    f = B_GLOBAL @ c_full
    return _compute_ratio_from_f(f)


def _objective_axisym(c_axisym):
    """Objective: R for axisymmetric coefficients (11-dim)."""
    c_full = _expand_axisym(c_axisym)
    f = B_AXISYM @ c_axisym
    return _compute_ratio_from_f(f)


def _objective_and_grad_axisym(c_axisym):
    """
    Compute R and its gradient with respect to axisymmetric coefficients.
    """
    c_full = _expand_axisym(c_axisym)
    f = B_AXISYM @ c_axisym
    f2 = f * f
    norm2_sq = np.dot(W_GLOBAL, f2)
    
    if norm2_sq < 1e-30:
        return np.inf, np.zeros(N_AXISYM)
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.dot(W_GLOBAL, f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    if norm4 < 1e-30:
        return np.inf, np.zeros(N_AXISYM)
    
    R = norm4 / norm2
    
    # d||f||_4 / dc = (1/4) ||f||_4^{-3} * d||f||_4^4/dc
    # d||f||_4^4/dc = 4 * B^T (W * f^3)
    # d||f||_2 / dc = (1/2) ||f||_2^{-1} * d||f||_2^2/dc
    # d||f||_2^2/dc = 2 * B^T (W * f)
    
    grad_4 = 4.0 * (B_AXISYM.T @ (W_GLOBAL * f * f2))
    grad_2 = 2.0 * (B_AXISYM.T @ (W_GLOBAL * f))
    
    grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
    
    return R, grad_R


def _normalize_unit_l2(c_axisym):
    """Normalize axisymmetric coefficients so that ||f||_2 = 1 on the sphere."""
    f = B_AXISYM @ c_axisym
    norm2_sq = np.dot(W_GLOBAL, f * f)
    if norm2_sq < 1e-30:
        return c_axisym
    return c_axisym / np.sqrt(norm2_sq)


def _gradient_descent_polish(c_axisym, deadline, max_iter=5000):
    """
    Refined gradient descent with adaptive step size and backtracking.
    Works on the 11-dim axisymmetric space with L2 normalization.
    """
    c = c_axisym.copy()
    c = _normalize_unit_l2(c)
    current_val = _objective_axisym(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        val, grad = _objective_and_grad_axisym(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        step = 0.1
        improved = False
        for _backtrack in range(50):
            c_try = c - step * grad
            c_try = _normalize_unit_l2(c_try)
            val_try = _objective_axisym(c_try)
            if val_try < current_val - 1e-18:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            if step < 1e-15:
                break
            step = 0.5
            for _backtrack in range(30):
                c_try = c + step * grad
                c_try = _normalize_unit_l2(c_try)
                val_try = _objective_axisym(c_try)
                if val_try < current_val - 1e-18:
                    c = c_try
                    current_val = val_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    return current_val, c


def _nelder_mead_minimize(func, x0, max_iter=5000, tol=1e-14, deadline=None, norm_func=None):
    """
    Nelder-Mead minimization with optional normalization callback.
    """
    n = len(x0)
    
    if norm_func is not None:
        x0 = norm_func(x0).copy()
    
    simplex = [x0.copy()]
    base_step = 0.05
    for i in range(n):
        xi = x0.copy()
        xi[i] += base_step
        if norm_func is not None:
            xi = norm_func(xi)
        simplex.append(xi)
    
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_val = min(fvals)
    best_point = simplex[np.argmin(fvals)].copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        if norm_func is not None:
            xr = norm_func(xr)
        fr = func(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            if norm_func is not None:
                xe = norm_func(xe)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            if norm_func is not None:
                xc = norm_func(xc)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    if norm_func is not None:
                        simplex[i] = norm_func(simplex[i])
                    fvals[i] = func(simplex[i])
        
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[np.argmin(fvals)].copy()
    
    return best_val, best_point


def _lgbfsb_optimize_axisym(c0, deadline):
    """
    Optimize using L-BFGS-B with gradient information in axisymmetric space.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = _normalize_unit_l2(c)
        val, grad = _objective_and_grad_axisym(c_norm)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = scipy_minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 8000}
        )
        if result.success or result.fun < 1e10:
            c_opt = _normalize_unit_l2(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0


def _generate_initial_points_axisym(n_random=80):
    """Generate diverse initial points in the 11-dim axisymmetric space."""
    inits = []
    
    # 1. Zonal harmonic only (m=0)
    c_zonal = np.zeros(N_AXISYM)
    c_zonal[0] = 1.0  # m=0 is first in AXISYM_INDICES
    inits.append(c_zonal)
    
    # 2. Each axisymmetric mode alone
    for i in range(N_AXISYM):
        c = np.zeros(N_AXISYM)
        c[i] = 1.0
        inits.append(c)
    
    # 3. Pairs of modes
    for i in range(N_AXISYM):
        for j in range(i+1, N_AXISYM):
            c = np.zeros(N_AXISYM)
            c[i] = 1.0
            c[j] = 1.0
            inits.append(c)
    
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(n_random):
        c = rng.normal(size=N_AXISYM)
        inits.append(c)
    
    # 5. Structured: equal weights on all modes
    c_eq = np.ones(N_AXISYM)
    inits.append(c_eq)
    
    # 6. Alternating signs
    c_alt = np.ones(N_AXISYM) * np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
    inits.append(c_alt)
    
    # 7. Only even m (zonal-like): m=0, ±2, ±4, ±6, ±8, ±10
    # These are all modes in axisymmetric space, so all are included
    
    # 8. Polynomial-like patterns
    for power in range(1, 5):
        c = np.zeros(N_AXISYM)
        for i in range(N_AXISYM):
            c[i] = (i + 1) ** power
        inits.append(c)
    
    return inits


def _momentum_gradient_descent(c_axisym, deadline, max_iter=10000):
    """
    Momentum-accelerated gradient descent with adaptive step size.
    """
    c = c_axisym.copy()
    c = _normalize_unit_l2(c)
    current_val = _objective_axisym(c)
    
    momentum = np.zeros(N_AXISYM)
    beta = 0.9
    lr = 0.05
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        val, grad = _objective_and_grad_axisym(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        momentum = beta * momentum - lr * grad
        c_try = c + momentum
        c_try = _normalize_unit_l2(c_try)
        val_try = _objective_axisym(c_try)
        
        if val_try < current_val:
            c = c_try
            current_val = val_try
        else:
            # Backtrack: reduce momentum
            momentum *= 0.5
            c_try = c + momentum
            c_try = _normalize_unit_l2(c_try)
            val_try = _objective_axisym(c_try)
            if val_try < current_val:
                c = c_try
                current_val = val_try
            else:
                # Reduce learning rate
                lr *= 0.5
                momentum = -lr * grad
                c_try = c + momentum
                c_try = _normalize_unit_l2(c_try)
                val_try = _objective_axisym(c_try)
                if val_try < current_val:
                    c = c_try
                    current_val = val_try
                else:
                    if lr < 1e-15:
                        break
    
    return current_val, c


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic that minimize ||f||_4 / ||f||_2.
    Exploits rotational symmetry to reduce search to 11-dim axisymmetric space.
    """
    start_time = time.time()
    deadline = start_time + 610  # Leave 20s margin for 630s limit
    
    best_R = np.inf
    best_c_axisym = None
    
    # Generate initial points
    inits = _generate_initial_points_axisym(n_random=80)
    
    # Phase 1: Quick screening with L-BFGS-B on all inits
    for c0 in inits:
        if time.time() > deadline - 60:
            break
            
        try:
            R, c_opt = _lgbfsb_optimize_axisym(c0, deadline)
            if np.isfinite(R) and R < best_R:
                best_R = R
                best_c_axisym = c_opt
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on best and nearby points
    if time.time() < deadline - 30:
        candidates = []
        if best_c_axisym is not None:
            candidates.append(best_c_axisym)
        
        # Add some random perturbations of the best
        if best_c_axisym is not None:
            rng = np.random.default_rng(999)
            for _ in range(10):
                c_pert = best_c_axisym + rng.normal(scale=0.1, size=N_AXISYM)
                candidates.append(c_pert)
        
        # Add top few from screening
        for c0 in inits[:5]:
            candidates.append(c0)
        
        for c0 in candidates:
            if time.time() > deadline - 20:
                break
            try:
                R, c_opt = _nelder_mead_minimize(
                    _objective_axisym, c0, max_iter=5000, tol=1e-14,
                    deadline=deadline, norm_func=_normalize_unit_l2
                )
                if np.isfinite(R) and R < best_R:
                    best_R = R
                    best_c_axisym = c_opt
            except Exception:
                pass
    
    # Phase 3: Momentum gradient descent polish
    if time.time() < deadline - 10 and best_c_axisym is not None:
        try:
            R, c_opt = _momentum_gradient_descent(best_c_axisym, deadline, max_iter=10000)
            if np.isfinite(R) and R < best_R:
                best_R = R
                best_c_axisym = c_opt
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with gradient descent
    if time.time() < deadline - 5 and best_c_axisym is not None:
        try:
            R, c_opt = _gradient_descent_polish(best_c_axisym, deadline, max_iter=5000)
            if np.isfinite(R) and R < best_R:
                best_R = R
                best_c_axisym = c_opt
        except Exception:
            pass
    
    # Expand to full 21-dim vector
    if best_c_axisym is not None:
        best_c_full = _expand_axisym(best_c_axisym)
    else:
        # Fallback: zonal harmonic
        best_c_full = np.zeros(21)
        best_c_full[10] = 1.0
    
    # Verify validity
    if len(best_c_full) != 21:
        best_c_full = np.zeros(21)
        best_c_full[10] = 1.0
    if np.any(np.isnan(best_c_full)) or np.any(np.isinf(best_c_full)):
        best_c_full = np.zeros(21)
        best_c_full[10] = 1.0
    max_abs = np.max(np.abs(best_c_full))
    if max_abs < 1e-3:
        best_c_full = best_c_full * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_c_full = best_c_full * (1.0 / max_abs)
    
    return best_c_full


# EVOLVE-BLOCK-END
