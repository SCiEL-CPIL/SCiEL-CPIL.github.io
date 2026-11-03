"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import eval_genlaguerre, gamma
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _laguerre_zeros_approx(m, n, num_points=5000):
    """Approximate zeros of L_m^{(n)}(x) by scanning for sign changes."""
    # The zeros of L_m^{(n)} are in (0, infinity)
    # Use a reasonable range
    x_max = 2 * m + n + 10
    x = np.linspace(1e-10, x_max, num_points)
    vals = eval_genlaguerre(m, n, x)
    
    # Find sign changes
    signs = np.sign(vals)
    zero_indices = []
    for i in range(len(signs) - 1):
        if signs[i] * signs[i+1] < 0:
            # Bisection
            a, b = x[i], x[i+1]
            fa, fb = vals[i], vals[i+1]
            for _ in range(60):
                mid = (a + b) / 2
                fm = eval_genlaguerre(m, n, mid)
                if fa * fm <= 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
            zero_indices.append((a + b) / 2)
    
    return np.array(zero_indices)


def _compute_C4_upper_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound given double-root locations.
    The refined Cohn-Goncalves construction uses a polynomial P(x) = prod_i (x - r_i)^2 * L_m^{(n)}(x)
    or similar. The C4 constant relates to the integral of |P|^2 against a weight.
    
    For the Fourier sign uncertainty problem with m double roots, the bound on C4
    is derived from the ratio of norms. A common form is:
    C4 <= (sum of certain terms involving roots) / (base constant)
    
    Based on the Laguerre polynomial construction for uncertainty inequalities:
    The bound is typically proportional to 1 / prod_i r_i^{alpha} * f(roots)
    
    Let's use the standard form where the bound decreases as roots are optimized.
    A reasonable model: C4_bound = K / prod_i r_i^2 * (1 + sum_i 1/r_i)
    or more likely, it's related to the norm of the constructed polynomial.
    
    For a concrete computable bound, we use:
    C4 <= (gamma(m+n+1) / gamma(n+1)) * 1 / prod_i r_i^{2} * sum_j (1/r_j)
    This is a heuristic that captures the right dependencies.
    """
    roots = np.array(roots)
    m = 12
    n = 1
    
    # The C4 upper bound for the refined inequality
    # Based on the Laguerre construction, the bound is:
    # C4 = (norm of constructed poly)^2 / (norm of base poly)^2
    # A simplified but monotone measure:
    log_prod = np.sum(2.0 * np.log(roots))
    sum_inv = np.sum(1.0 / roots)
    
    # The bound should decrease when roots are larger (more spread out)
    # and when they are well-separated
    bound = np.exp(-log_prod) * sum_inv * gamma(m + n + 1) / gamma(n + 1)
    
    return bound


def _dyson_equilibrium(m=12, n=1):
    """
    Find equilibrium positions of a Dyson gas with repulsion exponent 2
    and linear confining potential. This models the optimal root placement.
    
    The force balance for each particle i:
    sum_{j != i} 2/(x_i - x_j) + confining_force = 0
    
    For Laguerre-type weights, the confining potential is related to the
    asymptotic density of zeros.
    """
    # Initial guess: spread roots over expected range
    x_max_est = 2 * m + n + 5
    x = np.linspace(1.0, x_max_est, m)
    
    # Iterate to equilibrium
    for iteration in range(1000):
        forces = np.zeros(m)
        for i in range(m):
            repulsion = 0.0
            for j in range(m):
                if i != j:
                    repulsion += 2.0 / (x[i] - x[j])
            # Confining force: for Laguerre weight x^n e^{-x}, 
            # the effective potential derivative is -n/x + 1
            confining = 1.0 - n / x[i]
            forces[i] = repulsion + confining
        
        # Update positions
        x_new = x - 0.01 * forces
        x_new = np.maximum(x_new, 1e-6)  # Keep positive
        
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    
    # Sort and ensure positive
    x = np.sort(x)
    return x


def _optimize_roots(m=12, n=1):
    """
    Optimize the double-root locations to minimize the C4 upper bound.
    Uses multiple strategies: Dyson equilibrium, numerical optimization,
    and refinement.
    """
    best_roots = None
    best_score = np.inf
    
    # Strategy 1: Dyson gas equilibrium
    try:
        roots_dyson = _dyson_equilibrium(m, n)
        if len(roots_dyson) == m and np.all(roots_dyson > 0):
            score = _compute_C4_upper_bound(roots_dyson, m, n)
            if score < best_score:
                best_score = score
                best_roots = roots_dyson.copy()
    except Exception:
        pass
    
    # Strategy 2: Numerical optimization starting from Dyson solution
    if best_roots is not None:
        try:
            result = minimize(
                lambda r: _compute_C4_upper_bound(r, m, n),
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.success or result.fun < best_score:
                roots_opt = np.sort(result.x)
                if np.all(roots_opt > 0) and len(roots_opt) == m:
                    score = _compute_C4_upper_bound(roots_opt, m, n)
                    if score < best_score:
                        best_score = score
                        best_roots = roots_opt.copy()
        except Exception:
            pass
    
    # Strategy 3: Try with Laguerre zeros as initial guess
    try:
        lag_zeros = _laguerre_zeros_approx(m, n)
        if len(lag_zeros) >= m:
            # Use first m zeros or interpolate
            if len(lag_zeros) > m:
                # Resample to get m points
                idx = np.linspace(0, len(lag_zeros)-1, m).astype(int)
                lag_zeros = lag_zeros[idx]
            score = _compute_C4_upper_bound(lag_zeros, m, n)
            if score < best_score:
                best_score = score
                best_roots = lag_zeros.copy()
    except Exception:
        pass
    
    # Strategy 4: Uniform spacing refinement
    try:
        x_max_est = 2 * m + n + 5
        roots_uniform = np.linspace(1.0, x_max_est, m)
        result = minimize(
            lambda r: _compute_C4_upper_bound(r, m, n),
            roots_uniform,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < best_score:
            roots_opt = np.sort(result.x)
            if np.all(roots_opt > 0) and len(roots_opt) == m:
                score = _compute_C4_upper_bound(roots_opt, m, n)
                if score < best_score:
                    best_score = score
                    best_roots = roots_opt.copy()
    except Exception:
        pass
    
    if best_roots is None:
        # Fallback
        best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    return list(best_roots)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive floats.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    m = 12
    n = 1
    
    try:
        roots = _optimize_roots(m, n)
    except Exception:
        roots = [4.0 * (i + 1) for i in range(12)]
    
    # Ensure we return exactly 12 positive values
    roots = [float(r) for r in roots]
    while len(roots) < 12:
        roots.append(roots[-1] + 1.0)
    roots = roots[:12]
    roots = [max(r, 1e-6) for r in roots]
    roots = sorted(roots)
    
    return roots

# EVOLVE-BLOCK-END
