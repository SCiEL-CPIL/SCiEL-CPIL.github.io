import numpy as np
import time
import math

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 580.0

    m = 12
    n = 1

    # The previous best (33.3095) used alpha=23 scaled to max=48000.
    # Let's try a completely different approach:
    # 1. Try alpha=1 (the natural Cohn-Goncalves choice for n=1)
    # 2. Try different scaling strategies
    # 3. Try non-uniform spacing patterns that might be better

    # For n=1, the natural alpha is 2n-1 = 1
    # Let's compute roots for alpha=1 and try various scales
    from scipy.special import roots_genlaguerre

    # Get roots for alpha=1 (natural choice)
    roots_a1 = sorted(float(r) for r in roots_genlaguerre(m, 1.0)[0])
    # Get roots for alpha=23 (previous best)
    roots_a23 = sorted(float(r) for r in roots_genlaguerre(m, 23.0)[0])

    # Try various configurations and estimate their quality
    # The score appears to be an upper bound on C4, which depends on the polynomial
    # Let's compute an actual bound using the Laguerre polynomial construction

    def compute_C4_bound(roots):
        """
        Compute an upper bound on C4 for the given double-root locations.
        Using the Laguerre polynomial construction, C4 is bounded by
        a function of the roots. For the Fourier sign uncertainty problem,
        the bound involves the ratio of certain integrals.
        
        For a polynomial P with double roots at r_1 < r_2 < ... < r_m,
        P(x) = prod_i (x - r_i)^2 * (some factor)
        
        The C4 bound in the Cohn-Goncalves construction relates to
        the L^4 norm of the sign function vs the polynomial.
        
        A simpler proxy: the bound scales with max_root / (sum of spacing^2)^(1/2)
        or similar. Let's use a heuristic based on observed scores.
        """
        # Heuristic: score ≈ max_root / (arithmetic_mean_spacing)^2 * constant
        # But from observations, alpha=23 scaled to 48000 gave 33.3
        # Let's try to compute something more principled
        
        # The actual C4 bound for the Laguerre construction involves:
        # C4 <= K * (max_i r_i) / (prod_i r_i)^(1/m) * f(spacing)
        # where K is a constant depending on m
        
        # A better heuristic: the bound is minimized when roots are
        # spaced to balance the polynomial's growth at infinity vs
        # its values near the roots
        
        # For now, use: score ≈ max_root / (geometric_mean(roots))^2
        # This rewards large max but penalizes small roots
        log_roots = np.log(np.array(roots))
        geom_mean = np.exp(np.mean(log_roots))
        return roots[-1] / (geom_mean ** 2)

    # Actually, let me think about this differently.
    # The seed (arithmetic 4,8,...,48) gives score 1.0
    # alpha=1 unscaled gives some score
    # alpha=23 scaled to 48000 gives 33.3
    
    # The score is the upper bound on C4, and we want it LOWER.
    # Wait - the task says "score is the upper bound on C4" and 
    # "the evaluator measures the raw value (lower is better)"
    # But then "search score is seed/value, so 1.0 equals the seed program"
    
    # So if seed gives upper bound = S_seed, and our program gives S_ours,
    # then search score = S_seed / S_ours
    # If search score > 1, our upper bound is smaller (better)
    
    # Current best search score is 33.3, meaning our upper bound is 
    # 33.3 times smaller than the seed's upper bound.
    
    # We want to minimize the upper bound, i.e., maximize the search score.
    
    # Let me try a direct optimization: use scipy.optimize to find
    # root positions that minimize a proxy for the C4 upper bound.
    
    # The C4 upper bound for the Laguerre construction with double roots
    # at positions r_1 < ... < r_12 can be written as:
    # C4 <= A(m) * max(r_i) / (sum_{i=1}^{m} 1/r_i)^(1/m) * g(spacing)
    
    # Let's use a simpler proxy and optimize it.
    
    def proxy_score(roots_array):
        """Proxy for the C4 upper bound (lower is better)."""
        roots_array = np.asarray(roots_array, dtype=float)
        if any(roots_array <= 0):
            return np.inf
        # The bound involves the polynomial's behavior
        # A good proxy: max_root / (prod roots)^(1/m)
        log_sum = np.sum(np.log(roots_array))
        geom_mean = np.exp(log_sum / len(roots_array))
        return roots_array[-1] / geom_mean

    # Try several candidate configurations
    candidates = []

    # 1. alpha=1, various scales
    for scale in [1, 2, 5, 10, 20, 50, 100]:
        r = [x * scale for x in roots_a1]
        candidates.append(r)

    # 2. alpha=23, various scales
    for scale in [100, 500, 1000, 2000, 5000, 10000, 20000, 48000]:
        r = [x * (scale / roots_a23[-1]) for x in roots_a23]
        candidates.append(r)

    # 3. Arithmetic progression with various spacing
    for spacing in [1, 2, 5, 10, 20, 50, 100]:
        r = [spacing * (i + 1) for i in range(m)]
        candidates.append(r)

    # 4. Geometric progression
    for ratio in [1.1, 1.2, 1.5, 2.0]:
        r = [1.0 * (ratio ** i) for i in range(m)]
        candidates.append(r)

    # Evaluate all candidates
    best_score = np.inf
    best_roots = None
    for r in candidates:
        s = proxy_score(np.array(r))
        if s < best_score:
            best_score = s
            best_roots = r

    # Now do a local optimization starting from the best candidate
    from scipy.optimize import minimize

    def objective(x):
        # x is the log of the roots (to ensure positivity)
        roots = np.exp(x)
        if not all(roots > 0) or not all(roots < 1e6):
            return 1e10
        return proxy_score(roots)

    # Start from best_roots
    x0 = np.log(np.array(best_roots, dtype=float))
    
    # Constrain roots to be in increasing order and positive
    # Use log-space parameterization: x_i = log(r_i), with constraint x_i < x_{i+1}
    
    def constrained_objective(x):
        roots = np.exp(x)
        # Penalty for non-increasing roots
        diffs = np.diff(roots)
        if any(diffs <= 0):
            return 1e10
        if roots[-1] > 1e6 or roots[0] < 1e-6:
            return 1e10
        return proxy_score(roots)

    # Optimize
    try:
        result = minimize(constrained_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.success or result.fun < 1e9:
            optimized_roots = sorted([float(np.exp(v)) for v in result.x])
            if all(r > 0 for r in optimized_roots) and len(optimized_roots) == m:
                opt_score = proxy_score(np.array(optimized_roots))
                if opt_score < best_score:
                    best_score = opt_score
                    best_roots = optimized_roots
    except Exception:
        pass

    if time.time() - t0 > deadline - 10:
        return best_roots

    # Also try a global optimization with a few different initializations
    try:
        from scipy.optimize import differential_evolution
        
        bounds = [(0, 13)] * m  # log-space, roots from 1 to ~e^13
        
        def de_objective(x):
            roots = np.exp(x)
            diffs = np.diff(roots)
            if any(diffs <= 0):
                return 1e10
            if roots[-1] > 1e6 or roots[0] < 1e-6:
                return 1e10
            return proxy_score(roots)
        
        # Run DE with limited evaluations
        de_result = differential_evolution(de_objective, bounds, maxiter=100,
                                            popsize=10, seed=42, tol=1e-6)
        if de_result.fun < 1e9:
            de_roots = sorted([float(np.exp(v)) for v in de_result.x])
            if all(r > 0 for r in de_roots) and len(de_roots) == m:
                de_score = proxy_score(np.array(de_roots))
                if de_score < best_score:
                    best_score = de_score
                    best_roots = de_roots
    except Exception:
        pass

    if time.time() - t0 > deadline - 5:
        return best_roots

    # Final validation
    best_roots = sorted(best_roots)
    assert len(best_roots) == 12
    assert all(r > 0 for r in best_roots)
    
    return best_roots
