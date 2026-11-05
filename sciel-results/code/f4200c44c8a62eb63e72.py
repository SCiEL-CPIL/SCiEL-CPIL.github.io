import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.special import roots_laguerre
import time
import math

def _compute_c4_bound_laguerre(roots):
    """Compute the C4 upper bound using Laguerre basis norm.
    
    For P(x) = prod(x - r_i) (degree 12), the C4 bound is related to
    1 / ||P||^2 where ||.|| is the Laguerre inner product with alpha=1.
    ||P||^2 = sum_k c_k^2 * (k+1) where P = sum c_k L_k^{(1)}.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    if np.any(np.diff(roots) <= 1e-12):
        return np.inf
    
    # Build polynomial coefficients in standard basis (ascending)
    coeffs_desc = np.poly(roots)
    p = coeffs_desc[::-1]  # ascending
    
    # Precompute A and its inverse once (cached)
    if not hasattr(_compute_c4_bound_laguerre, '_A_T_inv'):
        A = np.zeros((13, 13))
        for k in range(13):
            for j in range(k + 1):
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
        try:
            _compute_c4_bound_laguerre._A_T_inv = np.linalg.inv(A.T)
        except np.linalg.LinAlgError:
            _compute_c4_bound_laguerre._A_T_inv = np.linalg.pinv(A.T)
    
    c = _compute_c4_bound_laguerre._A_T_inv @ p
    
    # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    norm_sq = np.sum(c**2 * (np.arange(13) + 1))
    
    if norm_sq <= 1e-10 or np.isinf(norm_sq):
        return np.inf
    
    return 1.0 / norm_sq

def _dyson_equations(roots):
    """Dyson gas equilibrium equations: sum_{j != i} 2/(r_i - r_j) = 1 for each i.
    This is the force balance for a Dyson gas with linear confining potential V(x) = x
    and interaction strength 2 (double roots)."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    residuals = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                s += 2.0 / (roots[i] - roots[j])
        residuals[i] = s - 1.0
    return residuals

def _solve_dyson_gas(init_roots, max_iter=500, tol=1e-14):
    """Solve Dyson gas equilibrium using Newton's method in log-space."""
    r = np.array(init_roots, dtype=np.float64).copy()
    log_r = np.log(r)
    
    for it in range(max_iter):
        # Compute residuals and Jacobian in log-space
        res = np.zeros(12)
        J = np.zeros((12, 12))
        
        for i in range(12):
            s = 0.0
            for j in range(12):
                if j != i:
                    s += 2.0 / (r[i] - r[j])
            res[i] = s - 1.0
            
            for k in range(12):
                if k == i:
                    # d/d(log r_i) of sum_{j != i} 2/(r_i - r_j) = sum_{j != i} 2 * (-1)/(r_i - r_j)^2 * r_i
                    # = -2 * r_i * sum_{j != i} 1/(r_i - r_j)^2
                    ds = 0.0
                    for j in range(12):
                        if j != i:
                            ds += 1.0 / (r[i] - r[j])**2
                    J[i, k] = -2.0 * r[i] * ds
                else:
                    # d/d(log r_k) of sum_{j != i} 2/(r_i - r_j) for j != i
                    # Only the term j=k contributes: d/d(log r_k) [2/(r_i - r_k)] = 2 * (-1)/(r_i - r_k)^2 * (-r_k) = 2*r_k/(r_i - r_k)^2
                    J[i, k] = 2.0 * r[k] / (r[i] - r[k])**2
        
        if np.max(np.abs(res)) < tol:
            break
        
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            # Regularize
            try:
                delta = np.linalg.solve(J + 1e-10 * np.eye(12), -res)
            except np.linalg.LinAlgError:
                break
        
        # Damping to ensure positivity
        step = 1.0
        for _ in range(50):
            new_log = log_r + step * delta
            if np.any(new_log < -30) or np.any(new_log > 30):
                step *= 0.5
                continue
            new_r = np.exp(new_log)
            if np.all(new_r > 0) and np.all(np.diff(np.sort(new_r)) > 1e-14):
                break
            step *= 0.5
        
        log_r = log_r + step * delta
        r = np.exp(log_r)
        r = np.sort(r)
        log_r = np.log(r)
    
    return r

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0
    
    def time_ok():
        return (time.time() - start_time) < (deadline - 30.0)
    
    # Initial guess: standard Laguerre roots of degree 12
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_laguerre(best_roots)
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Phase 1: Solve Dyson gas equilibrium from multiple initial conditions
    rng = np.random.default_rng(42)
    
    initial_guesses = [x0.copy()]
    
    # Scaled Laguerre roots
    for scale in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
        scaled = x0 * scale
        scaled = np.maximum(scaled, 1e-3)
        initial_guesses.append(np.sort(scaled))
    
    # Perturbed Laguerre roots
    for scale in [0.1, 0.3, 0.5, 1.0]:
        perturbed = x0 + scale * rng.standard_normal(12)
        perturbed = np.maximum(perturbed, 1e-3)
        initial_guesses.append(np.sort(perturbed))
    
    # Geometric sequences
    for start in [0.5, 1.0, 2.0]:
        for ratio in [1.2, 1.4, 1.6, 1.8]:
            geom = start * (ratio ** np.arange(12))
            if geom[0] > 1e-3 and geom[-1] < 1e6:
                initial_guesses.append(np.sort(geom))
    
    # Random positive roots
    for _ in range(10):
        rand = np.exp(rng.uniform(-2.0, 5.0, size=12))
        initial_guesses.append(np.sort(rand))
    
    for guess in initial_guesses:
        if not time_ok():
            break
        
        try:
            sol = _solve_dyson_gas(guess, max_iter=300, tol=1e-14)
            if np.all(sol > 0) and np.all(np.diff(sol) > 1e-10):
                score = _compute_c4_bound_laguerre(sol)
                if score < best_score:
                    best_score = score
                    best_roots = sol.copy()
        except Exception:
            pass
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Phase 2: Local polish in log-space from the best Dyson solution
    best_log = np.log(best_roots)
    
    def _objective_log(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        if np.any(np.diff(np.sort(roots)) <= 1e-14):
            return 1e10
        return _compute_c4_bound_laguerre(np.sort(roots))
    
    try:
        result = minimize(
            _objective_log,
            best_log,
            method='L-BFGS-B',
            bounds=[(-5.0, 15.0)] * 12,
            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16}
        )
        if result.success or result.fun < best_score:
            candidate = np.exp(result.x)
            candidate = np.sort(candidate)
            if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                score = _compute_c4_bound_laguerre(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
    except Exception:
        pass
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Phase 3: Re-solve Dyson from polished solution for consistency
    try:
        sol2 = _solve_dyson_gas(best_roots, max_iter=200, tol=1e-15)
        if np.all(sol2 > 0) and np.all(np.diff(sol2) > 1e-10):
            score2 = _compute_c4_bound_laguerre(sol2)
            if score2 < best_score:
                best_score = score2
                best_roots = sol2.copy()
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    if time_ok():
        try:
            best_log = np.log(best_roots)
            result_final = minimize(
                _objective_log,
                best_log,
                method='L-BFGS-B',
                bounds=[(-10.0, 20.0)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-19, 'gtol': 1e-17}
            )
            candidate = np.exp(result_final.x)
            candidate = np.sort(candidate)
            if np.all(candidate > 0) and np.all(np.diff(candidate) > 1e-10):
                score = _compute_c4_bound_laguerre(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
