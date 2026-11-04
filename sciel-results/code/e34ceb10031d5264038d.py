"""Signed autocorrelation max-abs minimization via spectral flatness gradient descent."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 120.0
_MARGIN = 10.0  # seconds safety margin


def _autoconv(a):
    """Full linear autoconvolution of a (length N)."""
    F = np.fft.rfft(a, n=2 * N - 1)
    return np.fft.irfft(F * F, n=2 * N - 1)


def _score(a):
    """Upper bound on C3': max|autoconv| / (sum a)^2. Assumes a in {-1,+1}."""
    ac = _autoconv(a)
    s = a.sum()
    if s == 0:
        return np.max(np.abs(ac))  # degenerate
    return np.max(np.abs(ac)) / (s * s)


def _local_search(s, max_rounds=8):
    """Iterated single-flip local search. Returns (best_signs, best_val)."""
    s = s.copy()
    v = _score(s)
    best_local = s.copy()
    best_local_v = v
    for _round in range(max_rounds):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE - _MARGIN:
                break
            s[i] = -s[i]
            s_sum2 = s.sum()
            if s_sum2 == 0:
                ac2 = _autoconv(s)
                v2 = np.max(np.abs(ac2))
            else:
                ac2 = _autoconv(s)
                v2 = np.max(np.abs(ac2)) / (s_sum2 * s_sum2)
            if v2 < best_local_v - 1e-12:
                best_local_v = v2
                best_local = s.copy()
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
        s = best_local.copy()
    return best_local, best_local_v


def _chirp_sequence(beta, n=N):
    """Generate a signed sequence from a quadratic chirp phase.
    
    x_k = sign(cos(2*pi*beta*k^2/n + 2*pi*beta*k/n))
    This gives a Zadoff-Chu-like structure with parameter beta.
    """
    k = np.arange(n)
    # Quadratic phase: pi * beta * k^2 / n (scaled for good coverage)
    phase = np.pi * beta * (k * k) / n + np.pi * beta * k / n
    x = np.cos(phase)
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    return signs


def _multi_start_chirp_search():
    """Search over chirp parameter beta with local search refinement."""
    best_signs = None
    best_val = np.inf
    
    # Coarse grid over beta
    beta_candidates = np.linspace(0.01, 10.0, 200)
    # Also include some structured values
    beta_candidates = np.concatenate([beta_candidates, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 0.5, 0.25]])
    
    for beta in beta_candidates:
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = _chirp_sequence(beta)
        v = _score(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    # Refine top candidates with local search
    if best_signs is not None:
        best_signs, best_val = _local_search(best_signs, max_rounds=10)
    
    # Also try some random initializations with local search
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        s, v = _local_search(s, max_rounds=6)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    return best_signs, best_val


def construct():
    global _DEADLINE
    
    # Phase 1: Chirp-based search (structural prior)
    best_signs, best_val = _multi_start_chirp_search()
    
    # Phase 2: Additional local search refinement
    if best_signs is not None:
        best_signs, best_val = _local_search(best_signs, max_rounds=15)
    
    # Phase 3: Try a few more diverse initializations if time permits
    rng = np.random.default_rng(123)
    for _ in range(10):
        if time.time() > _DEADLINE - _MARGIN:
            break
        # Mix of random and structured
        if _ % 2 == 0:
            s = rng.choice([-1.0, 1.0], size=N)
        else:
            # Low-frequency structured pattern
            k = np.arange(N)
            freq = rng.uniform(0.1, 2.0)
            s = np.sign(np.sin(2 * np.pi * freq * k / N))
            s[s == 0] = 1.0
        s, v = _local_search(s, max_rounds=5)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
    
    return best_signs
# EVOLVE-BLOCK-END
