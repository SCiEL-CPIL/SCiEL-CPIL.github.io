"""Optimized program: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s (evaluator kills at 150s, but we want buffer)


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si == 0:
        return np.inf
    return _autoconv_max_abs(h) / si


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0/1 values)."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    # Efficient generation using iterative doubling
    seq = np.array([0])
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def _smooth_surrogate_refine(h, max_iter=200, time_budget=None):
    """
    Use smooth surrogate (log-sum-exp) for the max in the objective.
    Minimizes logsumexp(|ac|) / (sum h)^2 via subgradient-like steps.
    """
    h = h.copy().astype(float)
    best_h = h.copy()
    best_score = _score(best_h)
    
    def smooth_obj(x):
        """Smooth approximation of max|autoconv| / (sum x)^2."""
        ac = fftconvolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        # Log-sum-exp for stable computation
        m = np.max(abs_ac)
        if m == 0:
            lse = 0.0
        else:
            lse = m + np.log(np.sum(np.exp(abs_ac - m)))
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf, np.zeros_like(x)
        obj = lse / (s * s)
        # Subgradient approximation: gradient of lse w.r.t. x
        # d/dx_i logsumexp(|ac|) ≈ sum_k |d|ac_k|/dx_i| * exp(|ac_k|-m) / sum(exp(...))
        # This is complex; use simple perturbation-based subgradient
        return obj, None
    
    for it in range(max_iter):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        obj, _ = smooth_obj(h)
        
        # Try flipping the element with largest contribution to peak
        ac = fftconvolve(h, h, mode='full')
        abs_ac = np.abs(ac)
        
        # Find which positions contribute most to the peak
        peak_idx = np.argmax(abs_ac)
        
        # The autoconv at lag k is sum_i h[i]*h[i+k]
        # Try flipping the element at the position that most affects the peak
        # Heuristic: flip the element with largest |h[i]| that's in a "hot" region
        
        # Simple approach: try all flips, pick best
        best_flip = -1
        best_flip_score = obj
        
        for i in range(len(h)):
            h[i] *= -1
            ac_new = fftconvolve(h, h, mode='full')
            s_new = np.sum(h)
            if abs(s_new) < 1e-10:
                h[i] *= -1
                continue
            ac_max = np.max(np.abs(ac_new))
            new_score = ac_max / (s_new * s_new)
            if new_score < best_flip_score:
                best_flip_score = new_score
                best_flip = i
            h[i] *= -1
        
        if best_flip >= 0 and best_flip_score < best_score:
            h[best_flip] *= -1
            best_score = best_flip_score
            best_h = h.copy()
        else:
            # No improvement from single flip; try pair flips on top contributors
            top_k = 10
            idx = np.argsort(-np.abs(h))[:top_k]
            found = False
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    h[i] *= -1
                    h[j] *= -1
                    ac_new = fftconvolve(h, h, mode='full')
                    s_new = np.sum(h)
                    if abs(s_new) < 1e-10:
                        h[i] *= -1
                        h[j] *= -1
                        continue
                    ac_max = np.max(np.abs(ac_new))
                    new_score = ac_max / (s_new * s_new)
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h.copy()
                        found = True
                    else:
                        h[i] *= -1
                        h[j] *= -1
            
            if not found:
                break
    
    return best_h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    # Strategy: Thue-Morse based deterministic sequence with local optimization
    
    np.random.seed(42)
    
    best_h = None
    best_score = np.inf
    
    # Generate Thue-Morse sequence and convert to ±1
    tm = _thue_morse(N)
    h_tm = 1.0 - 2.0 * tm  # Convert 0/1 to ±1
    
    # Try multiple Thue-Morse variants (shifts, sign flips)
    candidates = []
    
    # Original Thue-Morse
    candidates.append(h_tm.copy())
    
    # Negated Thue-Morse
    candidates.append(-h_tm.copy())
    
    # Shifted Thue-Morse (circular shift)
    for shift in [1, 2, 3]:
        candidates.append(np.roll(h_tm, shift).copy())
    
    # Thue-Morse with sparse zeros (to reduce integral magnitude issues)
    # Actually, keep all ±1 for maximum integral
    
    # Also try a few structured patterns for comparison
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    candidates.append(h_alt.copy())
    
    # Block pattern +1+1-1-1
    h_block = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    candidates.append(h_block.copy())
    
    # Random sparse 50%
    mask = np.random.rand(N) < 0.5
    h_rand = np.zeros(N)
    h_rand[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand.copy())
    
    # Random sparse 30%
    mask = np.random.rand(N) < 0.3
    h_rand2 = np.zeros(N)
    h_rand2[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    candidates.append(h_rand2.copy())
    
    # Evaluate and optimize each candidate
    for trial, h in enumerate(candidates):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # Quick score before optimization
        sc = _score(h)
        
        # Local sign flip optimization
        h_opt, sc_opt = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 5)
        
        if sc_opt < best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
    
    # Smooth surrogate refinement on the best candidate
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 15:
        best_h, best_score = _smooth_surrogate_refine(
            best_h, max_iter=100, time_budget=_TIME_LIMIT - 5
        )
    
    # Final local polish
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=50, time_budget=_TIME_LIMIT - 2
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
