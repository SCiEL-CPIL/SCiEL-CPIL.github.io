import numpy as np
from scipy.optimize import differential_evolution, minimize
from math import lgamma, exp
import time

def _compute_c4(roots):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    P(x) = L_1^{(24)}(x) * prod_{i=1}^{12} (x - r_i)^2
    Note: The task specifies m=12, n=1. The Laguerre part is likely L_m^(n-1) or similar.
    Based on inspiration 1, it uses L_1^{(24)}? No, that was a specific guess.
    Let's look at the task: "Laguerre-polynomial construction ... m=12 and dimension n=1".
    Usually P(x) = L_m^{(alpha)}(x) * prod (x-r_i)^2.
    If n=1, the weight is x^1 e^{-x}? Or is n the degree?
    Inspiration 1 says: P(x) = L_1^{(24)}(x) ... this looks specific to a different parameter set or a mistake in the inspiration.
    However, Inspiration 2 says: P(t) = prod_i (t - x_i)^2 * L_m^(n)(t).
    Let's stick to the general form: P(t) = L_m^{(alpha)}(t) * Q(t).
    The norm is in L2(w) where w(t) = t^n e^{-t}.
    If n=1, w(t) = t e^{-t}.
    The Laguerre polynomial L_m^{(alpha)} is orthogonal with weight t^alpha e^{-t}.
    So if the weight is t^1 e^{-t}, then alpha=1.
    So P(t) = L_{12}^{(1)}(t) * prod_{i=1}^{12} (t - r_i)^2.
    
    Let's verify the degree.
    L_12 has degree 12.
    Q(t) has degree 24.
    P(t) has degree 36.
    We need ||P||^2_{L2(t e^{-t})}.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    # Ensure distinct roots? The task says "double roots", so r_i are the locations.
    # They should be positive.
    
    # Build Q(t) = prod_i (t - r_i)^2
    q_coeffs = np.array([1.0])
    for r in roots:
        new_coeffs = np.zeros(len(q_coeffs) + 2)
        for i in range(len(q_coeffs)):
            new_coeffs[i] += q_coeffs[i]
            new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
            new_coeffs[i+2] += r*r * q_coeffs[i]
        q_coeffs = new_coeffs
    
    # Get coefficients of L_12^{(1)}(t)
    # L_n^{(alpha)}(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
    # Or use scipy.special.eval_genlaguerre? No, we need coefficients for norm calculation.
    # Let's compute coefficients of L_12^{(1)}(t).
    # L_n^{(alpha)}(x) = sum_{k=0}^n (-1)^k * Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1) * k!) * x^k
    # For n=12, alpha=1.
    m = 12
    alpha = 1
    l_coeffs = np.zeros(m + 1) # l_coeffs[k] is coeff of x^k
    for k in range(m + 1):
        # Term: (-1)^k * binom(m+alpha, m-k) * 1/k!
        # binom(13, 12-k) * 1/k!
        # Using log gamma for stability
        log_val = lgamma(m + alpha + 1) - lgamma(m - k + 1) - lgamma(k + alpha + 1) - lgamma(k + 1)
        val = exp(log_val)
        if k % 2 == 1:
            val = -val
        l_coeffs[k] = val
    
    # Multiply L(t) and Q(t) to get P(t) coefficients
    # l_coeffs has degree 12 (index 0..12)
    # q_coeffs has degree 24 (index 0..24)
    # p_coeffs will have degree 36
    n_l = len(l_coeffs)
    n_q = len(q_coeffs)
    p_coeffs = np.zeros(n_l + n_q - 1)
    for i in range(n_l):
        if abs(l_coeffs[i]) < 1e-15:
            continue
        for j in range(n_q):
            if abs(q_coeffs[j]) < 1e-15:
                continue
            p_coeffs[i + j] += l_coeffs[i] * q_coeffs[j]
    
    # c[k] is coefficient of x^k in P(t)
    c = p_coeffs
    
    # Calculate Norm Squared: Integral P(t)^2 t^alpha e^{-t} dt
    # P(t)^2 = (sum c_k t^k)^2 = sum_{k,j} c_k c_j t^{k+j}
    # Integral t^{k+j} t^alpha e^{-t} dt = Gamma(k+j+alpha+1)
    max_deg = len(c) - 1
    norm_sq = 0.0
    for k in range(max_deg + 1):
        ck = c[k]
        if abs(ck) < 1e-15:
            continue
        for j in range(k, max_deg + 1):
            cj = c[j]
            if abs(cj) < 1e-15:
                continue
            # k+j+alpha+1
            arg = k + j + alpha + 1
            if arg > 170: # lgamma overflows? No, exp(lgamma) overflows.
                # We need to handle large numbers.
                # log_norm_sq approach?
                # For now, let's assume it fits in float64. 36+36+1+1 = 74. Gamma(74) is ~ 10^100.
                # c_k can be large.
                # Let's use log-sum-exp if necessary, but direct sum might work if we normalize?
                # Actually, 1/sqrt(norm_sq) will be very small.
                # Let's try to compute in log domain to be safe.
                pass
            
            val = ck * cj * exp(lgamma(arg))
            if k != j:
                val *= 2.0
            norm_sq += val
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
    
    return 1.0 / np.sqrt(norm_sq)

def _log_objective(log_roots):
    """
    Objective function for optimization in log-space to ensure positivity.
    """
    roots = np.exp(log_roots)
    # Check for distinctness? The problem implies distinct locations for double roots.
    # If roots are too close, numerical issues might arise, but mathematically it's fine.
    # However, if roots are equal, the polynomial has a root of multiplicity 4.
    # The task asks for "12 positive double-root locations". This usually implies distinct locations.
    # Let's add a penalty if they are too close to avoid numerical instability in coefficient calculation.
    if np.any(roots > 1e8) or np.any(roots < 1e-8):
        return 1e10
    
    # Sort to check differences? No, the objective is symmetric.
    # But for numerical stability in _compute_c4, large roots cause large coefficients.
    # Log-space helps keep them in a reasonable range.
    
    return _compute_c4(roots)

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 570.0 # 60s buffer
    
    # Initial guess: Laguerre nodes scaled? Or arithmetic?
    # Seed was arithmetic: 4, 8, ..., 48.
    # Inspiration 1 used Laguerre roots scaled to match sum.
    # Let's start with the Laguerre roots of L_12^{(1)}? Or L_12?
    # The roots of L_n^{(alpha)} are in (0, infinity).
    # Let's use the roots of L_12^{(1)} as a starting point.
    from scipy.special import roots_genlaguerre
    try:
        initial_roots, _ = roots_genlaguerre(12, 1)
    except:
        initial_roots = np.linspace(1, 12, 12)
    
    initial_log_roots = np.log(initial_roots)
    
    # Define bounds for log_roots
    # Roots should be positive.
    # Lower bound: 1e-4 -> log ~ -9
    # Upper bound: 1e6 -> log ~ 13
    bounds = [(-9.0, 13.0)] * 12
    
    best_c4 = 1e10
    best_roots = initial_roots.copy()
    
    # 1. Try local refinement from initial guess
    try:
        result = minimize(
            _log_objective,
            initial_log_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_c4:
            best_c4 = result.fun
            best_roots = np.exp(result.x)
    except Exception:
        pass
    
    # 2. Global search using Differential Evolution
    # DE is robust and handles the non-convexity well.
    # Use a smaller population and generations to fit in time.
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _log_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False, # We will polish later
                update_interval=10,
                callback=lambda x, convergence: (time.time() > deadline - 60) # Stop if close to deadline
            )
            if de_result.fun < best_c4:
                best_c4 = de_result.fun
                best_roots = np.exp(de_result.x)
        except Exception:
            pass
    
    # 3. Final local polish from the best found so far
    if time.time() < deadline - 10:
        try:
            start_x = np.log(best_roots)
            result2 = minimize(
                _log_objective,
                start_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < best_c4:
                best_c4 = result2.fun
                best_roots = np.exp(result2.x)
        except Exception:
            pass
        
        # Try Nelder-Mead as a final check
        if time.time() < deadline - 5:
            try:
                result3 = minimize(
                    _log_objective,
                    np.log(best_roots),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'fatol': 1e-14, 'xatol': 1e-14}
                )
                if result3.fun < best_c4:
                    best_c4 = result3.fun
                    best_roots = np.exp(result3.x)
            except Exception:
                pass
    
    # Validate final result
    if np.any(best_roots <= 0) or not np.all(np.isfinite(best_roots)):
        return [float(x) for x in initial_roots]
        
    return [float(x) for x in best_roots]
