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

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    """
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
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient for coefficients c.
    """
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
    
    G4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    G2 = B_GLOBAL.T @ (W_GLOBAL * f)
    
    grad_R = G4 / (norm2 * norm4**3) - (norm4 * G2) / (norm2**3)
    
    return R, grad_R

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _gradient_descent_polish(c, deadline, max_iter=1000, momentum=0.9):
    """
    Refined gradient descent with adaptive step size, backtracking, and momentum.
    """
    c = c.copy()
    current_val = _compute_ratio(c)
    velocity = np.zeros_like(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        # Adaptive step with momentum
        step = 0.1 / (1.0 + 0.1 * grad_norm)
        velocity = momentum * velocity - step * grad
        
        improved = False
        for _backtrack in range(60):
            c_try = c + velocity
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-18:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
            velocity = momentum * velocity - step * grad
        
        if not improved:
            # Fallback without momentum
            velocity = np.zeros_like(c)
            step = 0.01
            for _backtrack in range(40):
                c_try = c - step * grad
                c_try = _normalize_unit_l2(c_try)
                val_try = _compute_ratio(c_try)
                if val_try < current_val - 1e-18:
                    c = c_try
                    current_val = val_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    return current_val, c

def _power_iteration(c0, deadline, max_iter=500, tol=1e-10):
    """
    Nonlinear power iteration for the L4/L2 eigenproblem.
    """
    c = c0.copy()
    c = _normalize_unit_l2(c)
    
    prev_val = _compute_ratio(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        f = B_GLOBAL @ c
        force = B_GLOBAL.T @ (W_GLOBAL * f * f * f)
        
        c_new = force
        c_new = _normalize_unit_l2(c_new)
        
        val_new = _compute_ratio(c_new)
        
        if val_new < prev_val:
            prev_val = val_new
            c = c_new
        else:
            if abs(val_new - prev_val) < tol:
                break
            c_damped = 0.5 * (c + c_new)
            c_damped = _normalize_unit_l2(c_damped)
            val_damped = _compute_ratio(c_damped)
            if val_damped < prev_val:
                prev_val = val_damped
                c = c_damped
            else:
                break
                
    return prev_val, c

def _lgbfsb_optimize(c0, deadline, norm_func=None):
    """
    Optimize using L-BFGS-B with gradient information.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = norm_func(c)
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
            options={'maxiter': 5000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxfun': 20000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _wigner_d_vectorized(j, mp, m_array, beta):
    """
    Vectorized computation of d^j_{mp,m}(beta) for an array of m values.
    j=10, mp=10, m_array contains values from -10 to 10.
    """
    m = m_array.astype(float)
    k_min = np.maximum(0, m - mp)
    k_max = np.minimum(j + m, j - mp)
    
    # We need to iterate over k values. Since j=10, k ranges from 0 to 10.
    # For each k, we compute the term where all factorials are valid.
    total = np.zeros_like(m)
    
    cb = np.cos(beta / 2.0)
    sb = np.sin(beta / 2.0)
    
    # Precompute factorials for j=10
    # We need (j+mp)!, (j-mp)!, and for each m: (j+m)!, (j-m)!
    # And for each k: (j+m-k)!, (mp-m+k)!, (j-mp-k)!, k!
    
    # Since j=10 and mp=10:
    # (j+mp)! = 20!, (j-mp)! = 0! = 1
    # For m in -10..10: (j+m)! and (j-m)!
    
    # Let's compute this more carefully for j=10, mp=10
    # d^10_{10,m}(beta)
    
    # k must satisfy:
    # j+m-k >= 0 => k <= j+m
    # mp-m+k >= 0 => k >= m-mp = m-10
    # j-mp-k >= 0 => k <= j-mp = 0
    # k >= 0
    
    # So k must satisfy: max(0, m-10) <= k <= min(j+m, 0) = min(10+m, 0)
    # Since m <= 10, 10+m >= 0, so min(10+m, 0) = 0 when 10+m >= 0, i.e., m >= -10
    # And max(0, m-10): since m <= 10, m-10 <= 0, so max(0, m-10) = 0
    
    # So k = 0 is the only valid value for all m in -10..10!
    
    k = 0
    k_min_valid = np.maximum(0, m - mp)
    k_max_valid = np.minimum(j + m, j - mp)
    
    valid = (k >= k_min_valid) & (k <= k_max_valid)
    
    # For k=0:
    # num = (j+mp)! * (j-mp)! * (j+m)! * (j-m)!
    # den = (j+m-k)! * (mp-m+k)! * (j-mp-k)! * k!
    #     = (j+m)! * (mp-m)! * (j-mp)! * 0!
    
    # So the ratio simplifies to:
    # sqrt((j+mp)! * (j-mp)! * (j+m)! * (j-m)! / ((j+m)! * (mp-m)! * (j-mp)! * 1))
    # = sqrt((j+mp)! * (j-m)! / (mp-m)!)
    
    # For j=10, mp=10:
    # = sqrt(20! * (10-m)! / (10-m)!) = sqrt(20!)
    
    # Wait, that's constant! Let me recheck.
    
    # Actually: (j+mp)! = 20!, (j-mp)! = 0! = 1
    # (j+m)! = (10+m)!, (j-m)! = (10-m)!
    # (j+m-k)! = (10+m)!, (mp-m+k)! = (10-m)!, (j-mp-k)! = 0! = 1, k! = 1
    
    # So num = 20! * 1 * (10+m)! * (10-m)!
    # den = (10+m)! * (10-m)! * 1 * 1
    # ratio = 20!
    # sqrt(ratio) = sqrt(20!)
    
    # And the powers:
    # exp1 = 2j + m - mp - 2k = 20 + m - 10 - 0 = 10 + m
    # exp2 = mp - m + 2k = 10 - m + 0 = 10 - m
    
    # term = (-1)^0 * sqrt(20!) * cb^(10+m) * sb^(10-m)
    
    # So d^10_{10,m}(beta) = sqrt(20!) * cb^(10+m) * sb^(10-m) for valid k=0
    
    # But we need to check validity: k=0 must satisfy k_min <= 0 <= k_max
    # k_min = max(0, m-10), k_max = min(10+m, 0)
    # For m in -10..10: k_min = 0 (since m-10 <= 0), k_max = 0 (since 10+m >= 0 and min with 0 is 0)
    # So k=0 is always valid!
    
    sqrt_20fact = np.sqrt(np.prod(np.arange(1, 21)))
    
    result = sqrt_20fact * (cb ** (10 + m)) * (sb ** (10 - m))
    
    return result

def _generate_symmetric_initial_points(dim):
    """
    Generate initial points based on direct structural analysis of the problem.
    """
    inits = []
    
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    c_m10 = np.zeros(dim)
    c_m10[20] = 1.0
    c_m10[0] = 1.0
    inits.append(c_m10)
    
    c_all = np.ones(dim)
    inits.append(c_all)
    
    k = 10
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    c_inv_N = np.zeros(dim)
    c_inv_N[10] = 1.0 / N[0]
    for m in range(1, 11):
        c_inv_N[10+m] = 1.0 / N[m]
        c_inv_N[10-m] = 1.0 / N[m]
    inits.append(c_inv_N)
    
    c_alt = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_alt[i] = 1.0 if m % 2 == 0 else -1.0
    inits.append(c_alt)
    
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-m**2 / (2 * sigma**2))
        inits.append(c_gauss)
        
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-(m-10)**2 / (2 * sigma**2))
        inits.append(c_gauss)

    np.random.seed(42)
    for _ in range(15):
        c_rand = np.random.randn(dim)
        for m in range(1, 11):
            c_rand[10-m] = c_rand[10+m]
        inits.append(c_rand)
        
    c_cos = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_cos[i] = np.cos(m * np.pi / 20.0)
    inits.append(c_cos)
    
    c_sin = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_sin[i] = np.sin(m * np.pi / 20.0)
    inits.append(c_sin)
    
    # Rotated monomial seeds using vectorized Wigner d
    rng = np.random.RandomState(12345)
    n_rot_seeds = 60
    m_values = np.array(range(-10, 11))
    
    for seed_idx in range(n_rot_seeds):
        beta = rng.uniform(0, np.pi)
        alpha = rng.uniform(0, 2 * np.pi)
        
        d_vals = _wigner_d_vectorized(10, 10, m_values, beta)
        
        c_rot = np.zeros(dim)
        for idx, m in enumerate(m_values):
            d_val = d_vals[idx]
            c_rot[10 + m] = d_val * np.cos(m * alpha)
            if m != 0:
                c_rot[10 - m] = d_val * np.sin(m * alpha)
        
        c_rot_norm = _normalize_unit_l2(c_rot)
        if np.linalg.norm(c_rot_norm) > 1e-10:
            inits.append(c_rot_norm)
    
    return inits

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    dim = 21
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    # Generate diverse initial points
    inits = _generate_symmetric_initial_points(dim)
    
    # Evaluate all inits and keep track of best
    init_vals = []
    for c_init in inits:
        c_norm = _normalize_unit_l2(c_init)
        val = objective(c_norm)
        init_vals.append((val, c_norm))
        if val < best_val:
            best_val = val
            best_coeffs = c_norm.copy()
            
    # Sort initializers by value
    init_vals.sort(key=lambda x: x[0])
    
    # Phase 1: Power Iteration from top initial points
    n_starts_power = 20
    for val, c_start in init_vals[:n_starts_power]:
        if time.time() > deadline - 200:
            break
        try:
            val_pi, c_pi = _power_iteration(c_start, deadline, max_iter=200)
            if val_pi < best_val:
                best_val = val_pi
                best_coeffs = c_pi.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B optimization from best starts
    if time.time() < deadline - 100:
        try:
            val_opt, c_opt = _lgbfsb_optimize(best_coeffs, deadline, norm_func=_normalize_unit_l2)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
        
    for val, c_start in init_vals[:15]:
        if time.time() > deadline - 150:
            break
        if np.allclose(c_start, best_coeffs):
            continue
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 3: Gradient descent polishing on the best found so far
    if time.time() < deadline - 60:
        c_polish = best_coeffs.copy()
        val_polish, c_polish_opt = _gradient_descent_polish(c_polish, deadline, max_iter=3000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish_opt.copy()
        
        if time.time() < deadline - 30:
            try:
                val_polish2, c_polish2 = _lgbfsb_optimize(
                    best_coeffs, deadline, norm_func=_normalize_unit_l2
                )
                if val_polish2 < best_val:
                    best_val = val_polish2
                    best_coeffs = c_polish2.copy()
            except Exception:
                pass
        
        if time.time() < deadline - 10:
            c_final = best_coeffs.copy()
            val_final, c_final_opt = _gradient_descent_polish(c_final, deadline, max_iter=5000)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = c_final_opt.copy()
        
        # Phase 4: Multi-start L-BFGS-B from perturbed best solution
        if time.time() < deadline - 15:
            rng2 = np.random.RandomState(999)
            for pert_idx in range(5):
                if time.time() > deadline - 15:
                    break
                noise = rng2.randn(dim) * 0.05
                c_pert = best_coeffs + noise
                c_pert = _normalize_unit_l2(c_pert)
                try:
                    val_pert, c_pert_opt = _lgbfsb_optimize(
                        c_pert, deadline, norm_func=_normalize_unit_l2
                    )
                    if val_pert < best_val:
                        best_val = val_pert
                        best_coeffs = c_pert_opt.copy()
                except Exception:
                    pass

    # Final validation and normalization
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
