import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 590.0  # safety margin
    
    m = 12
    # Seed: uniform grid 4, 8, ..., 48
    seed_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    def compute_score(roots):
        """
        Compute the C4 upper bound from the 12 double-root locations.
        The polynomial is P(x) = prod_{i=1}^{12} (x - r_i)^2
        C4 is likely related to some norm or integral of P or its derivative.
        
        For Fourier sign uncertainty, a common bound involves:
        - The number of sign changes (fixed at 12 for double roots)
        - The magnitude of the polynomial at certain points
        - An integral bound
        
        Let's try: C4 = max |P(x)| over x in [0, max_roots + margin]
        Or: C4 = integral of |P(x)| * weight
        """
        roots = np.sort(roots)
        # Evaluate P(x) = prod (x - r_i)^2 at a grid
        x = np.linspace(0, roots[-1] + 5, 1000)
        log_p = np.zeros_like(x)
        for r in roots:
            log_p += 2 * np.log(np.abs(x - r) + 1e-15)
        p = np.exp(log_p)
        # Try different scoring functions
        # Option 1: max |P|
        score1 = np.max(p)
        # Option 2: integral of P
        score2 = np.trapz(p, x)
        # Option 3: max of P' / P (log derivative)
        # Option 4: condition number related
        # For now, use max as proxy
        return score1
    
    def objective(roots):
        """Minimize the C4 bound."""
        roots = np.sort(roots)
        # Ensure positive
        if np.any(roots <= 0):
            return 1e10
        # Compute a proxy for C4
        # The C4 in Fourier uncertainty often involves:
        # C4 = sup_x |f(x)| * (something involving the roots)
        # Let's try: the product of (r_i) or sum of (1/r_i) or similar
        # Actually, for sign uncertainty, the bound often scales with
        # the spread of the roots or the product
        
        # Try: minimize the maximum gap between consecutive roots
        # (more uniform might be better, or worse)
        gaps = np.diff(roots)
        return np.max(gaps)
    
    # Since we don't know the exact scoring, let's try a few structural candidates
    # and return the one that seems most "refined"
    
    candidates = []
    
    # Candidate 1: Seed (uniform grid)
    candidates.append(seed_roots.copy())
    
    # Candidate 2: Chebyshev nodes on [4, 48]
    theta = np.linspace(0, np.pi, m)
    cheb = 0.5 * (48 - 4) * (1 + np.cos(theta)) + 4
    cheb = np.sort(cheb)
    candidates.append(cheb)
    
    # Candidate 3: Laguerre roots with alpha=0
    try:
        from scipy.special import roots_laguerre
        lag_roots, _ = roots_laguerre(m)
        # Scale to similar range as seed
        lag_roots_scaled = lag_roots * (48 / lag_roots[-1])
        candidates.append(lag_roots_scaled)
    except:
        pass
    
    # Candidate 4: Laguerre roots with alpha=14 (from current program)
    try:
        from scipy.special import roots_laguerre
        lag_roots_14, _ = roots_laguerre(m, 14.0)
        candidates.append(lag_roots_14)
    except:
        pass
    
    # Candidate 5: Slightly perturbed uniform grid (harmonic spacing)
    # r_i = 4 * i / (1 - i/(m+1)) or similar
    i = np.arange(1, m+1)
    harmonic = 4.0 * i / (1.0 - i / (m + 2.0))
    candidates.append(harmonic)
    
    # Candidate 6: Geometric spacing
    geo = 4.0 * (48.0/4.0)**(np.arange(m)/(m-1))
    candidates.append(geo)
    
    # Candidate 7: Roots of derivative of Laguerre polynomial
    try:
        from scipy.special import roots_laguerre
        # L_13 roots, use first 12
        lag13, _ = roots_laguerre(13)
        candidates.append(lag13[:12])
    except:
        pass
    
    # Now, since we don't know the exact score function, let's try to optimize
    # using a reasonable proxy. The most likely C4 for Fourier sign uncertainty
    # with double roots involves the polynomial P(x) = prod (x-r_i)^2.
    # A common bound is C4 = max_x |P(x)| * exp(-x/2) or similar weighted norm.
    
    def weighted_norm(roots, weight_alpha=0.5):
        """Compute max |P(x)| * exp(-alpha*x) as a proxy for C4."""
        roots = np.sort(roots)
        x = np.linspace(0, roots[-1] + 10, 2000)
        log_p = np.zeros_like(x)
        for r in roots:
            log_p += 2 * np.log(np.abs(x - r) + 1e-15)
        log_weighted = log_p - weight_alpha * x
        return np.max(np.exp(log_weighted))
    
    # Evaluate all candidates with the weighted norm proxy
    best_roots = seed_roots.copy()
    best_score = weighted_norm(seed_roots)
    
    for cand in candidates:
        if len(cand) != m:
            continue
        if np.any(cand <= 0):
            continue
        s = weighted_norm(cand)
        if s < best_score:
            best_score = s
            best_roots = cand.copy()
    
    # Try local optimization from the best candidate
    if time.time() - start_time < deadline - 60:
        try:
            def obj(roots):
                roots = np.sort(roots)
                if np.any(roots <= 0):
                    return 1e10
                return weighted_norm(roots)
            
            result = minimize(
                obj, 
                best_roots, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < best_score:
                opt_roots = np.sort(result.x)
                if np.all(opt_roots > 0):
                    best_roots = opt_roots
                    best_score = result.fun
        except:
            pass
    
    # Final check
    if time.time() - start_time > deadline:
        return list(seed_roots)
    
    result = sorted(best_roots)
    return [float(r) for r in result]
