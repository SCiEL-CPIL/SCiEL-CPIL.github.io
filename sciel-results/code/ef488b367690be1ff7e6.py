import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_laguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 600.0

    m = 12
    n = 1

    def compute_c4_bound(roots):
        """
        Compute the C4 upper bound using Laguerre-weighted norm ratio.
        C4 ~ ||P||_{L^2(e^{-x})} / ||P'||_{L^2(e^{-x})}
        where P(x) = prod_{i=1}^{12} (x - a_i)^2
        """
        roots = np.array(roots, dtype=float)
        if np.any(roots <= 0) or np.any(np.diff(np.sort(roots)) <= 0):
            return 1e10

        # Use Gauss-Laguerre quadrature with sufficient nodes
        # For 12 double roots, P is degree 24, P' is degree 23
        # Need enough quadrature points for accuracy
        n_q = 200
        x_q, w_q = np.polynomial.laguerre.laggauss(n_q)

        # Compute log(P(x)^2 * e^{-x}) = 4*sum(log|x-a_i|) - x
        # P(x)^2 = prod (x - a_i)^4
        diff = x_q[:, None] - roots[None, :]
        log_P2_e = 4.0 * np.log(np.abs(diff)) - x_q[:, None]
        log_P2_e = np.sum(log_P2_e, axis=1)
        log_P2_e = np.clip(log_P2_e, -700, 700)
        P2_e = np.exp(log_P2_e)
        norm_P2 = np.sum(w_q * P2_e)

        # P'(x) = P(x) * sum_i 2/(x - a_i)
        S = np.sum(2.0 / diff, axis=1)
        log_P2prime_e = log_P2_e + 2.0 * np.log(np.abs(S))
        log_P2prime_e = np.clip(log_P2prime_e, -700, 700)
        P2prime_e = np.exp(log_P2prime_e)
        norm_P2prime = np.sum(w_q * P2prime_e)

        if norm_P2prime <= 0 or norm_P2 <= 0:
            return 1e10

        return np.sqrt(norm_P2) / np.sqrt(norm_P2prime)

    # Test with known configurations to validate
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    scale10_roots = np.array([40.0 * (i + 1) for i in range(12)])

    c4_seed = compute_c4_bound(seed_roots)
    c4_s10 = compute_c4_bound(scale10_roots)

    print(f"C4 seed: {c4_seed:.6e}, C4 s10: {c4_s10:.6e}")
    print(f"Ratio seed/s10: {c4_seed/c4_s10:.4f}")

    best_roots = scale10_roots[:]
    best_val = c4_s10

    # Define objective with log-parameterization for positivity
    def objective(theta):
        if time.time() - t0 > deadline - 10:
            return 1e10
        roots = np.sort(np.exp(theta))
        return compute_c4_bound(roots)

    # Start from scale-10 arithmetic
    theta0 = np.log(scale10_roots)

    # Nelder-Mead optimization
    try:
        result = minimize(objective, theta0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_val:
            best_roots = np.sort(np.exp(result.x))
            best_val = result.fun
            print(f"NM improved: {best_val:.6e}")
    except Exception as e:
        print(f"NM failed: {e}")

    # L-BFGS-B
    try:
        result2 = minimize(objective, theta0, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14})
        if result2.fun < best_val:
            best_roots = np.sort(np.exp(result2.x))
            best_val = result2.fun
            print(f"L-BFGS-B improved: {best_val:.6e}")
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")

    # Try Laguerre zeros as starting points with various scales
    try:
        lag_zeros = roots_laguerre(12)[0]
        lag_zeros = sorted(lag_zeros)
        for target_max in [48, 100, 200, 480, 500, 800]:
            scale = target_max / lag_zeros[-1]
            lag_roots = np.array([z * scale for z in lag_zeros])
            val = compute_c4_bound(lag_roots)
            if val < best_val:
                best_roots = lag_roots[:]
                best_val = val
                print(f"Laguerre max={target_max} improved: {best_val:.6e}")
    except Exception as e:
        print(f"Laguerre failed: {e}")

    # Fine-tune around best found
    if time.time() - t0 < deadline - 60:
        try:
            theta_best = np.log(best_roots)
            result3 = minimize(objective, theta_best, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result3.fun < best_val:
                best_roots = np.sort(np.exp(result3.x))
                best_val = result3.fun
                print(f"Polish improved: {best_val:.6e}")
        except Exception as e:
            print(f"Polish failed: {e}")

    # Final validation
    if np.any(best_roots <= 0):
        best_roots = scale10_roots[:]

    return [float(r) for r in best_roots]
