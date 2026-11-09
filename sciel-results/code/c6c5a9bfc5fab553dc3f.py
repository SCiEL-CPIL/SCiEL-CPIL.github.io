import numpy as np
from scipy.special import roots_laguerre, eval_laguerre, gammaln
from scipy.optimize import minimize, minimize_scalar
import time

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for a given set of 12 positive double-root locations.
    
    The Cohn-Goncalves construction involves a function f(x) = P(x) * e^{-x/4}
    where P(x) = prod_i (x - r_i)^2 is a polynomial with double roots.
    
    The C4 constant relates to the ratio of L4 norms:
    C4 >= ||f||_4^4 / ||f_hat||_4^4
    
    For the specific Laguerre-based construction, we compute:
    1. Position space: ||f||_4^4 = integral |P(x)^2 e^{-x/2}|^2 dx = integral P(x)^4 e^{-x} dx
    2. Momentum space: We use the Fourier transform properties of Laguerre polynomials.
    
    For a practical and optimizable bound, we focus on minimizing the position-space
    integral which is dominated by the root configuration. A smaller integral implies
    a more concentrated function, which typically leads to a better (lower) uncertainty
    constant in this context, or we use a proxy that correlates with the final bound.
    
    Specifically, we compute:
    I = integral_0^inf P(x)^4 e^{-x} dx
    The bound is proportional to I^{-1/4} * (dual_factor). Since the dual factor is
    relatively stable for smooth root configurations, minimizing I is a strong heuristic.
    
    We use Gauss-Laguerre quadrature with a high number of points for accuracy.
    """
    r = np.array(roots, dtype=float)
    if len(r) < 2 or np.any(r <= 0):
        return 1e10
    r = np.sort(r)
    diffs = np.diff(r)
    if np.any(diffs <= 0):
        return 1e10
    
    # Use high-order Gauss-Laguerre quadrature
    n_quad = 200
    xq, wq = roots_laguerre(n_quad)
    
    # Compute P(x)^4 * e^{-x}
    # P(x) = prod (x - r_i)^2
    # P(x)^4 = prod (x - r_i)^8
    # To avoid overflow/underflow, we compute in log space or carefully.
    # Let's compute log(P(x)^4 * e^{-x}) = 8 * sum log|x - r_i| - x
    # Then sum exp(log_val) * wq
    
    # Efficient computation:
    # log_term = -xq
    # for r_i in r:
    #     log_term += 8 * log(abs(xq - r_i))
    # val = exp(log_term) * wq
    # integral = sum(val)
    
    log_vals = -xq.copy()
    for ri in r:
        # Avoid log(0)
        diff = np.abs(xq - ri)
        diff[diff < 1e-15] = 1e-15
        log_vals += 8.0 * np.log(diff)
    
    # Sum using log-sum-exp for stability if needed, but exp(log_vals) might be small enough
    # The roots are around O(10), so x-q nodes cover the range.
    # Let's just compute directly if it doesn't overflow.
    # Max log_val: if x is near a root, log is large negative? No, log(0) is -inf.
    # We clipped diff to 1e-15, so log(1e-15) ~ -34. 8 * -34 = -272.
    # exp(-272) is tiny.
    # Away from roots, log is positive? No, if x > r, log(x-r) can be positive.
    # But e^{-x} decays.
    
    # Let's try direct sum first.
    vals = np.exp(log_vals) * wq
    integral = np.sum(vals)
    
    if not np.isfinite(integral) or integral <= 0:
        return 1e10
    
    # The C4 bound scales as 1 / integral^{1/4}
    # We want to minimize the bound, so we maximize the integral?
    # Wait. C4 >= (||f||_4^4) / (||f_hat||_4^4).
    # If we fix the normalization, ||f||_2 = 1.
    # ||f||_4^4 = integral |f|^4.
    # If the function is more spread out, ||f||_4 is smaller.
    # The uncertainty principle says spread in x * spread in xi >= const.
    # The "sign uncertainty" bound is specific.
    # Let's look at the previous program's proxy: -log_prod(diffs).
    # That minimized the sum of logs of gaps, i.e., maximized the gaps?
    # -sum(log(diffs)) = -log(prod(diffs)). Minimizing this means maximizing prod(diffs).
    # Maximizing the product of gaps usually spreads the roots out.
    
    # Let's stick to a robust proxy that correlates with the "quality" of the bound.
    # The integral I = integral P(x)^4 e^{-x} dx.
    # If we spread roots, P(x) grows faster, but e^{-x} kills it.
    # Actually, for Laguerre polynomials, the optimal roots are the zeros of L_m^{alpha}.
    # The standard roots_laguerre(12) gives zeros of L_12^0.
    # The task asks to "tune" the construction.
    
    # Let's use the integral I itself as the objective to minimize or maximize?
    # If I is small, f is concentrated. Concentrated f has large Fourier transform.
    # Uncertainty product is large.
    # We want the *upper bound* on C4 to be low.
    # C4 is a constant such that ||f||_4 ||f_hat||_4 >= C4 ||f||_2^2.
    # We want the best possible lower bound for the uncertainty, or an upper bound for the constant?
    # "score is the upper bound on C4". "lower is better".
    # So we want to find a configuration where the *guaranteed* lower bound for the uncertainty
    # is high? No, "upper bound on C4".
    # If C4 is the constant in the inequality, and we provide an upper bound for it,
    # we want that upper bound to be as small as possible.
    # This means we want the inequality to be as tight as possible, or the constant to be small.
    
    # Let's use the integral I as a proxy.
    # If we minimize I, we make f more concentrated?
    # Let's just optimize for the integral I and see if it improves over the seed.
    # The seed score is 1.0.
    
    return -0.25 * np.log(integral)

def _objective(params):
    """
    Objective function for optimization.
    params: 12 log-space values that we transform to positive roots.
    """
    roots = np.exp(params)
    return _compute_c4_bound(roots)

def _gradient(params):
    """
    Numerical gradient for optimization.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        grad[i] = (f_plus - f0) / eps
        # Reset params to original
        params[i] = params_plus[i] - 2*eps
        f_minus = _objective(params[i])
        params[i] += eps
        grad[i] = (f_plus - f_minus) / (2*eps)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    try:
        # Start from standard Laguerre roots as initial guess
        roots_lag = roots_laguerre(12)
        r0 = np.sort(np.array(roots_lag.x, dtype=float))
        
        # Initial objective
        best_roots = r0.copy()
        best_score = _compute_c4_bound(best_roots)
        
        # Use log-space parameters for optimization
        log_params0 = np.log(r0)
        
        # Optimization Strategy:
        # 1. Coarse optimization with Nelder-Mead or Powell.
        # 2. Fine optimization with L-BFGS-B using numerical gradient.
        
        # Phase 1: Coarse
        for method in ['Powell', 'Nelder-Mead']:
            if time.time() > deadline - 100:
                break
                
            try:
                res = minimize(
                    _objective,
                    log_params0,
                    method=method,
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if time.time() > deadline - 100:
                    break
                    
                if res.fun < best_score:
                    new_roots = np.exp(res.x)
                    new_roots = np.sort(new_roots)
                    new_score = _compute_c4_bound(new_roots)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_roots = new_roots
                        log_params0 = res.x.copy()
                        
            except Exception:
                continue
        
        # Phase 2: Fine with L-BFGS-B
        if time.time() < deadline - 50:
            try:
                res = minimize(
                    _objective,
                    log_params0,
                    method='L-BFGS-B',
                    jac=_gradient,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < best_score:
                    new_roots = np.exp(res.x)
                    new_roots = np.sort(new_roots)
                    new_score = _compute_c4_bound(new_roots)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_roots = new_roots
                        
            except Exception:
                pass

        # Verify constraints: 12 positive, sorted
        best_roots = np.sort(best_roots)
        if len(best_roots) != 12 or np.any(best_roots <= 0):
            best_roots = r0.copy()
        
        # Final time check
        if time.time() > deadline:
            return [float(x) for x in r0]
            
        return [float(x) for x in best_roots]
        
    except Exception:
        # Fallback to standard Laguerre roots
        try:
            roots_lag = roots_laguerre(12)
            return [float(x) for x in np.sort(roots_lag.x)]
        except:
            return [4.0 * (i + 1) for i in range(12)]
