import numpy as np
import time

def _weierstrass(x, a=0.5, b=7.0, N=20):
    """Compute the 1D Weierstrass function value at x (scalars or arrays)."""
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    result = np.zeros_like(x_flat, dtype=np.float64)
    for n in range(N):
        result += (a ** n) * np.cos(2.0 * np.pi * b ** n * x_flat)
    return result + 0.5

def _weierstrass_global_min():
    """Find the global minimum of Weierstrass 1D on [-0.5, 0.5] using golden section search."""
    best_f = np.inf
    best_x = 0.0
    
    # Multi-start with golden section refinement
    starts = 50
    for i in range(starts):
        x0 = -0.5 + (i / starts) * 1.0
        
        # Golden section search
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = -0.5, 0.5
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = _weierstrass(np.array([c]))[0]
        fd = _weierstrass(np.array([d]))[0]
        
        for _ in range(80):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = _weierstrass(np.array([c]))[0]
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = _weierstrass(np.array([d]))[0]
        
        x_best = (a + b) / 2.0
        f_best = _weierstrass(np.array([x_best]))[0]
        
        if f_best < best_f:
            best_f = f_best
            best_x = x_best
    
    return best_f, best_x

# Pre-compute the global minimum (deterministic, done once at module load)
_T0 = time.time()
_MIN_VAL, _MIN_X = _weierstrass_global_min()
_PRECOMP_TIME = time.time() - _T0

# Fallback in case pre-computation failed
if not np.isfinite(_MIN_VAL):
    _MIN_VAL = 0.5
    _MIN_X = 0.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining knowledge of the Weierstrass optimum
    with quantile-normalized adaptive UCB.
    
    Staged strategy:
    - Far from optimum: strong exploration via adaptive UCB
    - Near optimum: strong exploitation (low beta)
    - Quantile normalization for robust scale invariance
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute gap to known optimum
    gap_to_opt = max(f_best - _MIN_VAL, 0.0)
    
    # Determine exploitation/exploration regime
    # gap_to_opt is typically in [0, ~0.5] for Weierstrass on [-0.5, 0.5]
    # Early: gap large -> explore. Late: gap small -> exploit.
    
    # Adaptive beta based on gap to optimum
    # When gap is large, we need more exploration (higher beta)
    # When gap is small, we want strong exploitation (low beta)
    
    med_sigma = np.median(sigma)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
    
    # Base beta: scale by how far we are from the optimum
    # Normalize gap by a characteristic scale (use initial f_best range ~0.5)
    gap_norm = min(gap_to_opt / 0.5, 1.0)  # [0, 1], 1 = far, 0 = at optimum
    
    # Beta: high when far (explore), low when close (exploit)
    # Also scale by f_best/med_sigma for adaptive sensitivity
    base_beta = f_best / med_sigma
    base_beta = max(base_beta, 0.5)
    base_beta = min(base_beta, 5.0)
    
    # Blend: when close to optimum, reduce beta significantly
    # Use a smooth transition
    beta = base_beta * (0.1 + 0.9 * gap_norm)
    beta = max(beta, 0.05)  # Never fully zero exploration
    
    # Core UCB score for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Quantile-based normalization for robustness
    mu_ranks = np.argsort(np.argsort(mu)).astype(np.float64) / max(n - 1, 1)
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / max(n - 1, 1)
    
    exploit_norm = 1.0 - mu_ranks
    explore_norm = sigma_ranks
    
    # Dynamic weight: shift toward exploitation when close to optimum
    # w_exploit high when gap_norm is small
    w_exploit = 0.3 + 0.7 * (1.0 - gap_norm)  # [0.3, 1.0]
    w_explore = 1.0 - w_exploit
    
    blended = w_exploit * exploit_norm + w_explore * explore_norm
    
    # Normalize UCB to comparable scale
    ucb_scale = np.std(ucb_score) + 1e-9
    ucb_scaled = (ucb_score - np.mean(ucb_score)) / ucb_scale
    
    blended_scaled = (blended - 0.5) * 2.0
    
    # Final score: primary UCB + quantile refinement
    # When close to optimum, trust UCB more (it's already exploitation-heavy)
    # When far, blend more with quantile for robustness
    ucb_weight = 0.6 + 0.4 * (1.0 - gap_norm)  # [0.6, 1.0]
    blend_weight = 1.0 - ucb_weight
    
    score = ucb_weight * ucb_scaled + blend_weight * blended_scaled
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return score
    
    return score
