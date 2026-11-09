"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import factorial, gamma

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0   # safety margin before the 630 s wall-clock kill
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _laguerre_zeros(n, alpha=0.0):
    """Compute zeros of Laguerre polynomial L_n(x) using scipy."""
    from scipy.special import roots_laguerre
    z, w = roots_laguerre(n, alpha)
    return sorted(z)

def _compute_bound(r):
    """
    Compute a proxy for the Cohn-Goncalves upper bound on C4.
    
    The constructed polynomial is P(x) = prod_{i=1}^{12} (x - r_i)^2
    The bound likely involves the L^2 norm of P against the Laguerre weight,
    or some related functional.
    
    For the Fourier sign uncertainty problem, the bound is related to:
    - The value of the polynomial at specific points
    - Integrals against the weight function
    
    Let's use: bound = sqrt(int_0^inf P(x)^2 e^{-x} dx)
    This is the L^2 norm of P under the Laguerre weight.
    """
    r = np.asarray(r, dtype=float)
    # Build polynomial coefficients for prod (x - r_i)^2
    # Start with 1
    coeffs = np.array([1.0])
    for ri in r:
        # (x - ri)^2 = x^2 - 2*ri*x + ri^2
        quad = np.array([1.0, -2.0*ri, ri*ri])
        coeffs = np.convolve(coeffs, quad)
    
    # coeffs is for P(x) = sum coeffs[k] * x^k (ascending order)
    # We need int_0^inf P(x)^2 e^{-x} dx
    # P(x)^2 has degree 48
    # int_0^inf x^j e^{-x} dx = j!
    
    # Compute P(x)^2 coefficients
    p2_coeffs = np.convolve(coeffs, coeffs)
    # p2_coeffs[k] is coefficient of x^k
    
    # Integral = sum_k p2_coeffs[k] * k!
    total = 0.0
    for k in range(len(p2_coeffs)):
        total += p2_coeffs[k] * math.factorial(k)
    
    if total < 0:
        total = 0.0
    return math.sqrt(total)

def _construct_core(seed_value):
    """
    Use Laguerre polynomial zeros as initial guess, then optimize.
    
    Key insight: The Cohn-Goncalves construction for Fourier sign uncertainty
    works best when the double roots are placed at or near the zeros of the
    appropriate orthogonal polynomial (Laguerre in this case).
    """
    np.random.seed(seed_value)
    
    # Get zeros of Laguerre polynomial L_12(x)
    lag_zeros = np.array(_laguerre_zeros(M))
    
    # The seed uses arithmetic spacing 4,8,...,48
    seed_roots = np.array([4.0 * (i + 1) for i in range(NROOTS)])
    
    # Try several candidate sets:
    candidates = []
    
    # 1. Laguerre zeros
    candidates.append(lag_zeros.copy())
    
    # 2. Seed
    candidates.append(seed_roots.copy())
    
    # 3. Laguerre zeros scaled by various factors
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
        candidates.append(lag_zeros * scale)
    
    # 4. Laguerre zeros with additive shift
    for shift in [0.5, 1.0, 2.0, 4.0, 8.0]:
        candidates.append(lag_zeros + shift)
    
    # 5. Mix of Laguerre zeros and seed
    for t in [0.25, 0.5, 0.75]:
        candidates.append(t * lag_zeros + (1-t) * seed_roots)
    
    # 6. Zeros of L_24 (since we have 12 double roots, maybe L_24 is relevant)
    try:
        lag24_zeros = np.array(_laguerre_zeros(24))
        candidates.append(lag24_zeros.copy())
        for scale in [0.5, 1.0, 2.0]:
            candidates.append(lag24_zeros * scale)
    except:
        pass
    
    # Evaluate all candidates
    best_score = float('inf')
    best_roots = seed_roots
    
    for cand in candidates:
        if _deadline_reached():
            break
        cand = np.sort(cand)
        if cand[0] <= 0 or cand[-1] > 500:
            continue
        try:
            score = _compute_bound(cand)
            if score < best_score:
                best_score = score
                best_roots = cand.copy()
        except:
            pass
    
    # Now optimize from the best starting point using Nelder-Mead
    if not _deadline_reached():
        def objective(r):
            r = np.sort(r)
            if r[0] <= 0:
                return 1e10
            return _compute_bound(r)
        
        try:
            res = minimize(objective, best_roots, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
            if res.success or res.fun < best_score:
                best_roots = np.sort(res.x)
                best_score = res.fun
        except:
            pass
    
    # Final optimization with more iterations if time permits
    if not _deadline_reached() and (time.time() - _START_TIME) < _HARD_DEADLINE * 0.6:
        def objective(r):
            r = np.sort(r)
            if r[0] <= 0:
                return 1e10
            return _compute_bound(r)
        
        try:
            res = minimize(objective, best_roots, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.fun < best_score:
                best_roots = np.sort(res.x)
                best_score = res.fun
        except:
            pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    # The seed (arithmetic spacing 4,8,...,48) scores 1.0.
    # Previous optimization attempts made the score worse (0.9817).
    # Let's try a few simple alternative patterns and see if any beat the seed.
    
    try:
        result = _construct_core(seed_value=0)
    except Exception:
        result = [4.0 * (i + 1) for i in range(NROOTS)]

    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = [4.0 * (i + 1) for i in range(NROOTS)]

    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
    return out
