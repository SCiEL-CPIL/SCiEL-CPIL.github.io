# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
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
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
        
    return B, weights

B_GLOBAL, W_GLOBAL = _precompute_basis()
N_PTS = B_GLOBAL.shape[0]
DIM = 21

def _compute_ratio(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    if norm2_sq < 1e-30:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2

def _objective_and_grad(c):
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    
    R = norm4 / norm2
    
    grad_4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    grad_2 = B_GLOBAL.T @ (W_GLOBAL * f)
    
    grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
    
    return R, grad_R

def _normalize_unit_l2(c):
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _lbfgsb_optimize(c0, deadline):
    """
    Use scipy L-BFGS-B with gradient for fast, high-precision convergence.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = _normalize_unit_l2(c)
        val, grad = _objective_and_grad(c_norm)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = scipy_minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 5000}
        )
        if result.fun < 1e10:
            c_opt = _normalize_unit_l2(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _nelder_mead_polish(c0, deadline, max_iter=2000):
    """
    Nelder-Mead for local polish, handling the non-smooth normalization boundary.
    """
    def func(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        c_norm = _normalize_unit_l2(c)
        val = _compute_ratio(c_norm)
        return val if np.isfinite(val) else 1e10
    
    n = len(c0)
    c0 = _normalize_unit_l2(c0).copy()
    
    simplex = [c0.copy()]
    base_step = 0.02
    for i in range(n):
        xi = c0.copy()
        xi[i] += base_step
        xi = _normalize_unit_l2(xi)
        simplex.append(xi)
    
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_val = min(fvals)
    best_point = simplex[np.argmin(fvals)].copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-14 * (1 + abs(fvals[0])):
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _normalize_unit_l2(xr)
        fr = func(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = _normalize_unit_l2(xe)
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
            xc = _normalize_unit_l2(xc)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _normalize_unit_l2(simplex[i])
                    fvals[i] = func(simplex[i])
        
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[np.argmin(fvals)].copy()
    
    return best_val, best_point

def _get_eigenvector_start():
    """
    Power iteration on the projected gradient of ||f||_4^4 to find a good initial guess.
    """
    dim = DIM
    np.random.seed(12345)
    c = np.random.randn(dim)
    c = _normalize_unit_l2(c)
    
    for _ in range(100):
        f = B_GLOBAL @ c
        f2 = f * f
        norm2_sq = np.sum(W_GLOBAL * f2)
        if norm2_sq < 1e-30:
            break
        
        grad_4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))
        proj = grad_4 - np.dot(grad_4, c) * c
        
        grad_norm = np.linalg.norm(proj)
        if grad_norm < 1e-15:
            break
            
        step = 0.01
        c_new = c - step * proj
        c_new = _normalize_unit_l2(c_new)
        
        if _compute_ratio(c_new) < _compute_ratio(c):
            c = c_new
        else:
            step *= 0.5
            c_new = c - step * proj
            c_new = _normalize_unit_l2(c_new)
            if _compute_ratio(c_new) < _compute_ratio(c):
                c = c_new
                
    return c

def _newton_step(c):
    """
    Newton's method step using the Hessian of the L4/L2 ratio.
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    
    if norm2_sq < 1e-30 or norm4_4 < 1e-30:
        return c
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    # Hessian of R = d2/dc^2 (||f||_4 / ||f||_2)
    # R = (sum w f^4)^(1/4) / (sum w f^2)^(1/2)
    # Let A = sum w f^4, B = sum w f^2
    # R = A^(1/4) B^(-1/2)
    # dR = (1/4)A^(-3/4) B^(-1/2) dA - (1/2)A^(1/4) B^(-3/2) dB
    # dA = 4 f^3 df, dB = 2 f df
    # dR = A^(-3/4) B^(-1/2) f^3 df - A^(1/4) B^(-3/2) f df
    # dR = (B f^3 - A f / B) / (A^(3/4) B^(3/2)) df  ... wait
    # dR = A^(-3/4) B^(-1/2) [f^3 - A/(4B) * 2f / 1] ... no
    # dA = 4 sum w f^3 df, dB = 2 sum w f df
    # dR = (1/4) A^(-3/4) B^(-1/2) * 4 sum w f^3 df - (1/2) A^(1/4) B^(-3/2) * 2 sum w f df
    # dR = A^(-3/4) B^(-1/2) sum w f^3 df - A^(1/4) B^(-3/2) sum w f df
    # grad_R = A^(-3/4) B^(-1/2) B^T (W f^3) - A^(1/4) B^(-3/2) B^T (W f)
    
    # Hessian:
    # H = d/dc [A^(-3/4) B^(-1/2) B^T (W f^3) - A^(1/4) B^(-3/2) B^T (W f)]
    # This is complex. Let's use a numerical Hessian or a simpler approach.
    # Actually, for a 21-dim problem, we can use a numerical Hessian.
    
    # Let's use a 5-point stencil numerical Hessian
    n = len(c)
    h = 1e-5
    H = np.zeros((n, n))
    f0 = _compute_ratio(c)
    
    for i in range(n):
        for j in range(i, n):
            c_pp = c.copy(); c_pp[i] += h; c_pp[j] += h
            c_pm = c.copy(); c_pm[i] += h; c_pm[j] -= h
            c_mp = c.copy(); c_mp[i] -= h; c_mp[j] += h
            c_mm = c.copy(); c_mm[i] -= h; c_mm[j] -= h
            
            f_pp = _compute_ratio(c_pp)
            f_pm = _compute_ratio(c_pm)
            f_mp = _compute_ratio(c_mp)
            f_mm = _compute_ratio(c_mm)
            
            if i == j:
                H[i, i] = (f_pp - 2*f0 + f_mm) / (h*h)
            else:
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4*h*h)
                H[j, i] = H[i, j]
    
    # Solve H delta = -grad
    try:
        val, grad = _objective_and_grad(c)
        # Add small regularization for stability
        H_reg = H + 1e-8 * np.eye(n)
        delta = np.linalg.solve(H_reg, -grad)
        
        # Damped Newton step
        step_size = 1.0
        c_new = c + step_size * delta
        c_new = _normalize_unit_l2(c_new)
        
        if _compute_ratio(c_new) < f0:
            return c_new
        else:
            # Try smaller step
            step_size = 0.5
            c_new = c + step_size * delta
            c_new = _normalize_unit_l2(c_new)
            if _compute_ratio(c_new) < f0:
                return c_new
            else:
                return c
    except np.linalg.LinAlgError:
        return c

def _generate_initial_points(dim):
    """Generate diverse initial points for optimization."""
    inits = []
    
    # 1. Zonal harmonic (m=0)
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(_normalize_unit_l2(c_zonal))
    
    # 2. Random normals
    np.random.seed(42)
    for _ in range(15):
        c = np.random.randn(dim)
        inits.append(_normalize_unit_l2(c))
        
    # 3. Sparse single-mode
    for i in range(dim):
        c = np.zeros(dim)
        c[i] = 1.0
        inits.append(_normalize_unit_l2(c))
        
    # 4. Sparse two-mode
    for i in range(0, dim, 2):
        for j in range(i+1, min(i+4, dim)):
            c = np.zeros(dim)
            c[i] = 1.0
            c[j] = 1.0
            inits.append(_normalize_unit_l2(c))
            
    # 5. Symmetric patterns
    c_sym = np.zeros(dim)
    c_sym[10] = 1.0  # m=0
    c_sym[0] = 1.0   # m=-10
    c_sym[20] = 1.0  # m=10
    inits.append(_normalize_unit_l2(c_sym))
    
    c_sym2 = np.zeros(dim)
    c_sym2[10] = 1.0  # m=0
    c_sym2[1] = 1.0   # m=-9
    c_sym2[19] = 1.0  # m=9
    inits.append(_normalize_unit_l2(c_sym2))
    
    return inits

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    dim = DIM
    
    best_coeffs = None
    best_val = np.inf
    
    # Get excellent starting point via power iteration
    c_eig = _get_eigenvector_start()
    val_eig = _compute_ratio(c_eig)
    best_val = val_eig
    best_coeffs = c_eig.copy()
    
    # Generate diverse initial points
    inits = _generate_initial_points(dim)
    
    # Evaluate and sort initial points
    init_vals = []
    for c_init in inits:
        val = _compute_ratio(c_init)
        init_vals.append((val, c_init))
        if val < best_val:
            best_val = val
            best_coeffs = c_init.copy()
            
    init_vals.sort(key=lambda x: x[0])
    
    # Optimize from best starts using L-BFGS-B
    n_starts = 20
    for val, c_start in init_vals[:n_starts]:
        if time.time() > deadline - 60:
            break
            
        try:
            val_opt, c_opt = _lbfgsb_optimize(c_start, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
        
        # Polish with Nelder-Mead
        if time.time() > deadline - 30:
            break
        try:
            val_polish, c_polish = _nelder_mead_polish(c_opt, deadline, max_iter=500)
            if val_polish < best_val:
                best_val = val_polish
                best_coeffs = c_polish.copy()
        except Exception:
            pass
        
        # Newton step to escape shallow minima
        if time.time() > deadline - 10:
            break
        try:
            c_newton = _newton_step(c_polish)
            val_newton = _compute_ratio(c_newton)
            if val_newton < best_val:
                best_val = val_newton
                best_coeffs = c_newton.copy()
                # Follow up with L-BFGS-B from the Newton result
                if time.time() < deadline - 20:
                    val_lbfgs, c_lbfgs = _lbfgsb_optimize(c_newton, deadline)
                    if val_lbfgs < best_val:
                        best_val = val_lbfgs
                        best_coeffs = c_lbfgs.copy()
        except Exception:
            pass

    # Final validation
    coeffs = best_coeffs.astype(np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
        
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3:
        if max_c > 0:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = np.ones(dim)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
        
    return coeffs

# EVOLVE-BLOCK-END
