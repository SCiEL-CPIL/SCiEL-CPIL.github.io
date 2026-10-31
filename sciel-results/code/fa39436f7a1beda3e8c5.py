"""Optimized program: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
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


def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0


def _generate_legendre_sequence(p, N):
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(np.float64)
    best_score = _score(best_h)
    n = len(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        for i in range(n):
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


def _pair_flip_optimize(h, max_rounds=10, time_budget=None):
    """Try flipping pairs of signs to escape local minima."""
    best_h = h.copy().astype(np.float64)
    best_score = _score(best_h)
    n = len(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        for i in range(n):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            for j in range(i+1, n):
                if time_budget is not None:
                    if time.time() - _START_TIME > time_budget:
                        return best_h, best_score
                
                best_h[i] *= -1
                best_h[j] *= -1
                new_score = _score(best_h)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_h[i] *= -1
                    best_h[j] *= -1
        
        if not improved:
            break
    
    return best_h, best_score


def _triple_flip_optimize(h, max_rounds=5, time_budget=None):
    """Try flipping triples of signs to escape deeper local minima."""
    best_h = h.copy().astype(np.float64)
    best_score = _score(best_h)
    n = len(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Sample random triples to keep it tractable
        num_triples = min(500, n * (n-1) * (n-2) // 6)
        for _ in range(num_triples):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            i, j, k = np.random.choice(n, 3, replace=False)
            
            best_h[i] *= -1
            best_h[j] *= -1
            best_h[k] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1
                best_h[j] *= -1
                best_h[k] *= -1
        
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    np.random.seed(42)
    
    best_h = None
    best_score = np.inf
    
    # Generate diverse initial patterns
    patterns = []
    
    # Legendre symbol sequences (algebraic structure)
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    for p in primes_to_try:
        h_leg = _generate_legendre_sequence(p, N)
        patterns.append(h_leg)
    
    # Pattern: Dense alternating
    h1 = np.ones(N)
    h1[1::2] = -1
    patterns.append(h1)
    
    # Pattern: Sparse even positions with alternating signs
    h2 = np.zeros(N)
    h2[0::2] = 1
    h2[2::4] = -1
    patterns.append(h2)
    
    # Pattern: Random sparse 50%
    mask3 = np.random.rand(N) < 0.5
    h3 = np.zeros(N)
    h3[mask3] = np.random.choice([-1, 1], size=np.sum(mask3))
    patterns.append(h3)
    
    # Pattern: Sparse 30% with random signs
    mask4 = np.random.rand(N) < 0.3
    h4 = np.zeros(N)
    h4[mask4] = np.random.choice([-1, 1], size=np.sum(mask4))
    patterns.append(h4)
    
    # Pattern: Block pattern +1+1-1-1
    h5 = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
    patterns.append(h5)
    
    # Pattern: Random sparse 40%
    mask6 = np.random.rand(N) < 0.4
    h6 = np.zeros(N)
    h6[mask6] = np.random.choice([-1, 1], size=np.sum(mask6))
    patterns.append(h6)
    
    # Pattern: Sparse 25%
    mask7 = np.random.rand(N) < 0.25
    h7 = np.zeros(N)
    h7[mask7] = np.random.choice([-1, 1], size=np.sum(mask7))
    patterns.append(h7)
    
    # Pattern: Alternating blocks of length 3
    h8 = np.tile(np.array([1, 1, -1]), N // 3 + 1)[:N]
    patterns.append(h8)
    
    # Pattern: Alternating blocks of length 5
    h9 = np.tile(np.array([1, 1, 1, -1, -1]), N // 5 + 1)[:N]
    patterns.append(h9)
    
    # Pattern: Random with bias toward zero (sparse)
    mask10 = np.random.rand(N) < 0.35
    h10 = np.zeros(N)
    h10[mask10] = np.random.choice([-1, 1], size=np.sum(mask10))
    patterns.append(h10)
    
    # Cosine patterns
    for k in range(1, 6):
        x = np.arange(N)
        h_cos = np.sign(np.cos(2 * np.pi * k * x / N))
        h_cos[h_cos == 0] = 1.0
        patterns.append(h_cos)
    
    # Optimize each pattern
    for idx, h in enumerate(patterns):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        # Single sign flip optimization
        h_opt, sc = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 15)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # If time allows, try pair flips on the best so far
        if time.time() - _START_TIME < _TIME_LIMIT - 30 and idx == 0:
            h_pair, sc_pair = _pair_flip_optimize(best_h, max_rounds=5, time_budget=_TIME_LIMIT - 25)
            if sc_pair < best_score:
                best_score = sc_pair
                best_h = h_pair.copy()
    
    # Final polish with more rounds
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=50, time_budget=_TIME_LIMIT - 5
        )
    
    # One more pair flip pass if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 5:
        best_h, best_score = _pair_flip_optimize(
            best_h, max_rounds=3, time_budget=_TIME_LIMIT - 2
        )
    
    # Triple flip pass if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 2:
        best_h, best_score = _triple_flip_optimize(
            best_h, max_rounds=2, time_budget=_TIME_LIMIT - 1
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
