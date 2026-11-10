import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
import time


def _build_quadrature():
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi


def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)

    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        P = lpmv(abs_m, k, x_gl)

        if m == 0:
            vals = N_m * P
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()

    return Y


def _eval_R(c, Y, weights):
    f = Y @ c
    f2 = f * f
    norm2_sq = np.dot(weights, f2)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(np.dot(weights, f2 * f2)))
    return norm4 / norm2


def _eval_R_and_grad(c, Y, weights):
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    norm2_sq = np.dot(weights, f2)
    if norm2_sq <= 1e-15:
        return 1e10, np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4_sq = np.dot(weights, f2 * f2)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    # grad R = (1/(4*norm4^3*norm2)) * 4*Y^T(W*f^3) - (norm4/(2*norm2^3)) * 2*Y^T(W*f)
    term1 = Y.T @ (weights * f3)
    term2 = Y.T @ (weights * f)
    grad = term1 / (norm2 * norm4**3) - norm4 * term2 / (norm2**3)
    return norm4 / norm2, grad


def _power_iteration(c, Y, weights, n_iter=300, tol=1e-15):
    """
    Nonlinear power iteration toward critical points of R = ||f||_4/||f||_2.
    At a critical point, f satisfies an eigenvalue-like equation.
    """
    c = c.copy()
    c = c / np.linalg.norm(c)
    for it in range(n_iter):
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        c_new = Y.T @ (weights * f3)
        c_new_norm = np.linalg.norm(c_new)
        if c_new_norm < 1e-15:
            break
        c_new = c_new / c_new_norm
        delta = np.linalg.norm(c_new - c)
        c = c_new
        if delta < tol:
            break
    return c


def _gradient_descent_polish(c, Y, weights, deadline, max_iter=500):
    """
    Gradient descent with backtracking line search and adaptive step size.
    """
    c = c.copy()
    current_val = _eval_R(c, Y, weights)
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        val, grad = _eval_R_and_grad(c, Y, weights)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        step = 0.1
        improved = False
        for _backtrack in range(50):
            c_try = c - step * grad
            val_try = _eval_R(c_try, Y, weights)
            if val_try < current_val - 1e-18:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            if step < 1e-15:
                break
            # Try larger step
            step = 0.5
            for _backtrack in range(30):
                c_try = c - step * grad
                val_try = _eval_R(c_try, Y, weights)
                if val_try < current_val - 1e-18:
                    c = c_try
                    current_val = val_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    return current_val, c


def _nelder_mead_minimize(func, x0, max_iter=2000, tol=1e-14, deadline=None):
    """
    Nelder-Mead minimization for derivative-free polishing.
    """
    n = len(x0)
    x0 = x0.copy()
    
    simplex = [x0.copy()]
    base_step = 0.02
    for i in range(n):
        xi = x0.copy()
        xi[i] += base_step
        simplex.append(xi)
    
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_val = min(fvals)
    best_point = simplex[np.argmin(fvals)].copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
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
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[np.argmin(fvals)].copy()
    
    return best_val, best_point


def _lbfgsb_optimize(c0, Y, weights, deadline, max_iter=1000):
    """
    L-BFGS-B optimization with tight tolerances.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        val, grad = _eval_R_and_grad(c, Y, weights)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 3000}
        )
        if result.success or result.fun < 1e10:
            return result.fun, result.x
    except Exception:
        pass
    
    return np.inf, c0


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi

    best_coeffs = None
    best_val = np.inf

    def try_candidate(c0, use_nm=True):
        nonlocal best_coeffs, best_val
        if time.time() > deadline - 30.0:
            return
        c0 = np.asarray(c0, dtype=np.float64)
        c0 = c0 / np.linalg.norm(c0)
        
        # Phase 1: Power iteration to get near a critical point
        c = _power_iteration(c0, Y, weights, n_iter=200)
        
        # Phase 2: Gradient descent polish
        val, c = _gradient_descent_polish(c, Y, weights, deadline, max_iter=200)
        if val < best_val:
            best_val = val
            best_coeffs = c.copy()
        
        # Phase 3: L-BFGS-B with tight tolerances
        if time.time() < deadline - 20.0:
            val2, c2 = _lbfgsb_optimize(c, Y, weights, deadline, max_iter=500)
            if val2 < best_val:
                best_val = val2
                best_coeffs = c2.copy()
        
        # Phase 4: Nelder-Mead as derivative-free complement
        if use_nm and time.time() < deadline - 15.0:
            def nm_func(cc):
                return _eval_R(cc, Y, weights)
            val3, c3 = _nelder_mead_minimize(nm_func, c, max_iter=500, tol=1e-15, deadline=deadline)
            if val3 < best_val:
                best_val = val3
                best_coeffs = c3.copy()
        
        # Phase 5: Final L-BFGS-B polish from NM result
        if best_coeffs is not None and time.time() < deadline - 10.0:
            val4, c4 = _lbfgsb_optimize(best_coeffs, Y, weights, deadline, max_iter=200)
            if val4 < best_val:
                best_val = val4
                best_coeffs = c4.copy()

    # Structured starts: exploit symmetry
    starts = []
    for m_val in range(0, 11, 2):
        c = np.zeros(21)
        c[m_val + 10] = 1.0
        starts.append(c)
        if m_val > 0:
            c2 = np.zeros(21)
            c2[m_val + 10] = 1.0
            c2[-m_val + 10] = 1.0
            starts.append(c2)
            c3 = np.zeros(21)
            c3[m_val + 10] = 1.0
            c3[-m_val + 10] = -1.0
            starts.append(c3)

    # Mixed zonal + sectorial with various ratios
    for ratio in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        c = np.zeros(21)
        c[10] = 1.0
        c[20] = ratio
        starts.append(c)
        c2 = np.zeros(21)
        c2[10] = ratio
        c2[20] = 1.0
        starts.append(c2)

    # Mixed m=0 and m=4
    for ratio in [0.3, 0.5, 0.7, 1.0, 1.5]:
        c = np.zeros(21)
        c[10] = 1.0
        c[14] = ratio
        starts.append(c)
        c2 = np.zeros(21)
        c2[10] = ratio
        c2[14] = 1.0
        starts.append(c2)

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(12):
        starts.append(rng.normal(size=21))

    for c0 in starts:
        if time.time() > deadline - 35.0:
            break
        try_candidate(c0, use_nm=True)

    # Additional random starts if time permits
    if best_coeffs is not None and time.time() < deadline - 50.0:
        for _ in range(20):
            if time.time() > deadline - 25.0:
                break
            try_candidate(rng.normal(size=21), use_nm=True)

    # Final polish on best result
    if best_coeffs is not None and time.time() < deadline - 10.0:
        val_final, c_final = _lbfgsb_optimize(best_coeffs, Y, weights, deadline, max_iter=300)
        if val_final < best_val:
            best_val = val_final
            best_coeffs = c_final

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs
        else:
            best_coeffs[10] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs

    return best_coeffs
