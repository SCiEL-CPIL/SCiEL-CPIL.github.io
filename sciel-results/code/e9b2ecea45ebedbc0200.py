# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_legendre

_DEADLINE_SECONDS = 620.0


def _build_harmonic_matrix():
    """Build the real orthonormal degree-10 spherical harmonic evaluation matrix."""
    k = 10
    Ntheta = 70
    Nphi = 130

    x, w = np.polynomial.legendre.leggauss(Ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi = 2.0 * np.pi / Nphi

    H = np.zeros((Ntheta, Nphi, 2 * k + 1), dtype=np.float64)

    P = np.zeros((Ntheta, k + 1), dtype=np.float64)
    for m in range(k + 1):
        P[:, m] = eval_legendre(k, m, x)

    def N_const(m):
        m = abs(int(m))
        lo = k - m + 1
        hi = k + m
        den = 1.0
        for j in range(lo, hi + 1):
            den *= j
        return np.sqrt((2 * k + 1) / (4.0 * np.pi) * (1.0 / den))

    for m in range(-k, k + 1):
        j = m + k
        if m == 0:
            H[:, :, j] = N_const(0) * P[:, 0][:, None] * np.ones((1, Nphi))
        elif m > 0:
            H[:, :, j] = np.sqrt(2.0) * N_const(m) * P[:, m][:, None] * np.cos(m * phi)[None, :]
        else:
            am = -m
            H[:, :, j] = np.sqrt(2.0) * N_const(am) * P[:, am][:, None] * np.sin(am * phi)[None, :]

    weights = np.outer(w, np.full(Nphi, dphi))
    return H, weights


def _ratio_from_coeffs(c, H_flat, w_flat):
    """Compute R = ||f||_4 / ||f||_2."""
    f = H_flat @ c
    f2 = f * f
    f4 = f2 * f2
    A = np.dot(f4, w_flat)
    B = np.dot(f2, w_flat)
    if A <= 0 or B <= 0:
        return np.inf
    return A ** 0.25 / np.sqrt(B)


def _power_iteration(c, H_flat, w_flat, max_iters=200, tol=1e-14, deadline=None):
    """
    Nonlinear power iteration for minimizing L4/L2 ratio.
    
    The map is: c_new = normalize(H^T (f^3 * w))
    This is the gradient ascent direction for E[f^4] at fixed ||c||=1,
    which is equivalent to minimizing L4/L2 since L2 is fixed on the sphere.
    
    We use extrapolation (Anderson-like) to accelerate convergence.
    """
    n = len(c)
    c = c.copy()
    cn = np.linalg.norm(c)
    if cn < 1e-15:
        return c
    
    c = c / cn
    best_c = c.copy()
    best_R = _ratio_from_coeffs(c, H_flat, w_flat)
    
    # Keep history for extrapolation
    history = [c.copy()]
    
    for it in range(max_iters):
        if deadline is not None and (time.time() - deadline[0]) > deadline[1]:
            break
        
        f = H_flat @ c
        f3 = f * f * f
        # Gradient of E[f^4] = 4 * H^T (f^3 * w)
        grad = H_flat.T @ (f3 * w_flat)
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Project gradient to tangent space
        c_dot_g = np.dot(c, grad)
        grad_tan = grad - c_dot_g * c
        gt_norm = np.linalg.norm(grad_tan)
        
        if gt_norm < tol:
            break
        
        # Try extrapolation if we have at least 2 previous iterates
        if len(history) >= 2:
            # Simple extrapolation: c_new = c + alpha * (c - c_prev)
            c_prev = history[-2]
            diff = c - c_prev
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-15:
                # Extrapolate along the difference direction
                alpha = min(1.0, 2.0 * gt_norm / (diff_norm + 1e-15))
                c_extrap = c + alpha * diff
                c_extrap = c_extrap / (np.linalg.norm(c_extrap) + 1e-15)
                
                # Check if extrapolated point is better
                R_extrap = _ratio_from_coeffs(c_extrap, H_flat, w_flat)
                if R_extrap < best_R:
                    c = c_extrap
                    best_c = c.copy()
                    best_R = R_extrap
                    history.append(c.copy())
                    if len(history) > 5:
                        history.pop(0)
                    continue
        
        # Standard power iteration step
        # c_new = normalize(c + step * grad_tan)
        # Use adaptive step size
        step = min(1.0, 1.0 / (gt_norm + 1e-10))
        
        c_trial = c + step * grad_tan
        cn_t = np.linalg.norm(c_trial)
        if cn_t < 1e-15:
            break
        c_trial = c_trial / cn_t
        
        R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
        
        if R_trial < best_R:
            c = c_trial
            best_c = c.copy()
            best_R = R_trial
            history.append(c.copy())
            if len(history) > 5:
                history.pop(0)
        else:
            # Try smaller step
            for _ in range(5):
                step *= 0.5
                c_trial = c + step * grad_tan
                cn_t = np.linalg.norm(c_trial)
                if cn_t < 1e-15:
                    break
                c_trial = c_trial / cn_t
                R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
                if R_trial < best_R:
                    c = c_trial
                    best_c = c.copy()
                    best_R = R_trial
                    history.append(c.copy())
                    if len(history) > 5:
                        history.pop(0)
                    break
            else:
                # No improvement, break
                break
    
    return best_c


def _gradient_polish(c, H_flat, w_flat, max_iters=50, deadline=None):
    """
    Final gradient-based polishing using trust-region with exact Hessian.
    """
    n = len(c)
    c = c.copy()
    cn = np.linalg.norm(c)
    if cn < 1e-15:
        return c
    c = c / cn
    
    best_c = c.copy()
    best_R = _ratio_from_coeffs(c, H_flat, w_flat)
    
    for it in range(max_iters):
        if deadline is not None and (time.time() - deadline[0]) > deadline[1]:
            break
        
        f = H_flat @ c
        f2 = f * f
        f4 = f2 * f2
        A = np.dot(f4, w_flat)
        B = np.dot(f2, w_flat)
        total_w = np.sum(w_flat)
        
        # Gradient of Var(f^2) = E[f^4] - (E[f^2])^2
        f3w = f * f2 * w_flat
        fw = f * w_flat
        grad_A = 4.0 * (H_flat.T @ f3w)
        grad_B = 2.0 * (H_flat.T @ fw)
        grad = grad_A - 2.0 * B / total_w * grad_B
        
        # Project to tangent
        c_dot_g = np.dot(c, grad)
        grad_tan = grad - c_dot_g * c
        g_norm = np.linalg.norm(grad_tan)
        
        if g_norm < 1e-14:
            break
        
        # Hessian
        f2w = f2 * w_flat
        H_diag = H_flat * f2w[:, None]
        Ht_H_f2w = H_flat.T @ H_diag
        
        HW = H_flat * w_flat[:, None]
        Ht_HW = H_flat.T @ HW
        
        dB_dc = 2.0 * (H_flat.T @ (f * w_flat))
        outer_B = np.outer(dB_dc, dB_dc)
        
        Hess = 12.0 * Ht_H_f2w - 4.0 * B / total_w * Ht_HW - 2.0 / total_w * outer_B
        
        P = np.eye(n) - np.outer(c, c)
        Hess_tan = P @ Hess @ P
        
        # LM solve
        I = np.eye(n)
        s = None
        for mu in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
            try:
                s = np.linalg.solve(Hess_tan + mu * I, -grad_tan)
                if np.linalg.norm(s) <= 1.0:
                    break
            except np.linalg.LinAlgError:
                continue
        
        if s is None:
            s = -grad_tan / (g_norm + 1e-30)
        
        s = s - np.dot(c, s) * c
        c_trial = c + s
        cn_t = np.linalg.norm(c_trial)
        if cn_t < 1e-15:
            break
        c_trial = c_trial / cn_t
        
        R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
        if R_trial < best_R:
            c = c_trial
            best_c = c.copy()
            best_R = R_trial
        else:
            # Try half step
            c_trial = c + 0.5 * s
            cn_t = np.linalg.norm(c_trial)
            if cn_t > 1e-15:
                c_trial = c_trial / cn_t
                R_trial = _ratio_from_coeffs(c_trial, H_flat, w_flat)
                if R_trial < best_R:
                    c = c_trial
                    best_c = c.copy()
                    best_R = R_trial
                else:
                    break
            else:
                break
    
    return best_c


def _make_initializations(k, rng, n):
    """Generate diverse starting points on the unit sphere."""
    inits = []
    
    # Single modes
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v.copy())
    
    # Pairs (cos + sin for same |m|)
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0 / np.sqrt(2)
        v[-m + k] = 1.0 / np.sqrt(2)
        inits.append(v.copy())
    
    # All ones normalized
    v = np.ones(n)
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # Even m only
    v = np.zeros(n)
    for m in range(-k, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # Odd m only
    v = np.zeros(n)
    for m in range(-k + 1, k + 1, 2):
        v[m + k] = 1.0
    v /= np.linalg.norm(v)
    inits.append(v.copy())
    
    # m=0 + high m combinations
    for m in [2, 4, 6, 8, 10]:
        for n_m in [2, 4, 6, 8, 10]:
            if m != n_m:
                v = np.zeros(n)
                v[0 + k] = 1.0
                v[m + k] = 1.0
                v[-m + k] = 1.0
                v[n_m + k] = 0.5
                v[-n_m + k] = 0.5
                v /= np.linalg.norm(v)
                inits.append(v.copy())
    
    # Structured: emphasize certain m values
    for weight_m in [2, 4, 6, 8, 10]:
        v = np.zeros(n)
        for m in range(-k, k + 1):
            v[m + k] = 1.0 if abs(m) == weight_m else (0.3 if abs(m) == 0 else 0.1)
        v /= np.linalg.norm(v)
        inits.append(v.copy())
    
    # Random
    for _ in range(30):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        inits.append(v.copy())
    
    # Sparse random
    for _ in range(15):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(3, 8), replace=False)
        v[idx] = rng.normal(size=len(idx))
        v /= np.linalg.norm(v)
        inits.append(v.copy())
    
    return inits


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m = -10..10) of a degree-10 real spherical
    harmonic f = sum_m c_m Y_m that MINIMIZES the norm ratio
        R = ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}
    
    Uses nonlinear power iteration (Rayleigh quotient iteration) on the unit
    sphere, which is a fundamentally different approach from gradient-based
    methods. The power iteration map c -> normalize(H^T (f^3 w)) converges
    quadratically to the flat-harmonic fixed point.
    """
    start_time = time.time()
    k = 10
    n = 2 * k + 1

    H, weights = _build_harmonic_matrix()
    H_flat = H.reshape(-1, n)
    w_flat = weights.reshape(-1)

    rng = np.random.default_rng(12345)

    def remaining():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    inits = _make_initializations(k, rng, n)

    best_c = np.ones(n, dtype=np.float64) / np.sqrt(n)
    best_R = _ratio_from_coeffs(best_c, H_flat, w_flat)

    # Phase 1: Power iteration from diverse starts
    for init_idx, c0 in enumerate(inits):
        if remaining() < 20.0:
            break

        c = c0.copy()
        cn = np.linalg.norm(c)
        if cn > 1e-15:
            c = c / cn

        # Run power iteration
        c_result = _power_iteration(c, H_flat, w_flat, max_iters=300, tol=1e-13,
                                    deadline=(start_time, _DEADLINE_SECONDS - 30.0))
        
        R = _ratio_from_coeffs(c_result, H_flat, w_flat)
        if R < best_R:
            best_R = R
            best_c = c_result.copy()

        # Phase 2: Gradient polish from the power iteration result
        if remaining() > 40.0:
            c_polished = _gradient_polish(c_result, H_flat, w_flat, max_iters=80,
                                           deadline=(start_time, _DEADLINE_SECONDS - 20.0))
            R_pol = _ratio_from_coeffs(c_polished, H_flat, w_flat)
            if R_pol < best_R:
                best_R = R_pol
                best_c = c_polished.copy()

    # Phase 3: Additional random restarts with power iteration + polish
    n_extra = 20
    for _ in range(n_extra):
        if remaining() < 30.0:
            break
        
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        
        c_result = _power_iteration(v, H_flat, w_flat, max_iters=200, tol=1e-13,
                                     deadline=(start_time, _DEADLINE_SECONDS - 40.0))
        
        R = _ratio_from_coeffs(c_result, H_flat, w_flat)
        if R < best_R:
            best_R = R
            best_c = c_result.copy()
        
        if remaining() > 50.0:
            c_polished = _gradient_polish(c_result, H_flat, w_flat, max_iters=60,
                                           deadline=(start_time, _DEADLINE_SECONDS - 30.0))
            R_pol = _ratio_from_coeffs(c_polished, H_flat, w_flat)
            if R_pol < best_R:
                best_R = R_pol
                best_c = c_polished.copy()

    # Final deep polish
    if remaining() > 60.0:
        c_final = _gradient_polish(best_c, H_flat, w_flat, max_iters=150,
                                    deadline=(start_time, _DEADLINE_SECONDS - 15.0))
        R_final = _ratio_from_coeffs(c_final, H_flat, w_flat)
        if R_final < best_R:
            best_R = R_final
            best_c = c_final.copy()

    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c


# EVOLVE-BLOCK-END
