"""Search for a low-autocorrelation signed step function (400 equal intervals).

Uses simulated annealing with single-flip moves and fast O(N) in-place
autocorrelation updates, plus a multi-start greedy polish phase.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    ac = _compute_acf(h)
    return float(np.max(np.abs(ac[1:])))


def _fast_flip_update(h, ac, i):
    """
    Flip h[i] and update ac in place.
    Returns the new peak sidelobe value.
    """
    val = h[i]
    
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # Flipping h[i] affects terms where j=i (if i+k<N) or j+k=i i.e. j=i-k (if i-k>=0)
    # delta_ac[k] = -2*val*(h[i+k]*1_{i+k<N} + h[i-k]*1_{i-k>=0})
    
    if i + 1 < N:
        ac[1:N-i] -= 2.0 * val * h[i+1:N]
    if i > 0:
        ac[1:i+1] -= 2.0 * val * h[i-1::-1]
        
    h[i] = -val
    
    return float(np.max(np.abs(ac[1:])))


def _anneal(seed, time_budget, T_init=8.0, T_min=0.01):
    """Simulated annealing with single-flip moves."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = T_init
    start = time.time()
    n_steps = 0
    
    while time.time() - start < time_budget:
        i = int(rng.integers(0, N))
        
        new_val = _fast_flip_update(h, ac, i)
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            _fast_flip_update(h, ac, i)
            
        n_steps += 1
        # Geometric cooling with floor
        T = max(T_min, T * 0.99995)
        
        # Periodically recompute ac to avoid drift
        if n_steps % 5000 == 0:
            ac[:] = _compute_acf(h)
            cur = float(np.max(np.abs(ac[1:])))
            if cur < best:
                best = cur
                best_h = h.copy()
        
    return best_h, best


def _greedy_polish(h, time_budget):
    """Greedy single-flip polish: repeatedly find and apply the best single flip."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        best_delta = 0.0
        
        for i in range(N):
            val = h[i]
            # Compute what the peak would be after flipping i
            # We need to find max |ac[k] + delta_ac[k]| for k>=1
            # delta_ac[k] = -2*val*(h[i+k]*1_{i+k<N} + h[i-k]*1_{i-k>=0})
            
            # Build the delta array for k=1..N-1
            # For efficiency, compute the new ac values directly
            new_ac = ac.copy()
            if i + 1 < N:
                new_ac[1:N-i] -= 2.0 * val * h[i+1:N]
            if i > 0:
                new_ac[1:i+1] -= 2.0 * val * h[i-1::-1]
            
            c = float(np.max(np.abs(new_ac[1:])))
            if c < best_c:
                best_c = c
                best_i = i
                best_delta = c - cur
        
        if best_i < 0:
            break
        
        # Apply the best flip
        _fast_flip_update(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
            
    return best_h, best


def construct():
    try:
        hard_deadline = 145.0
        start = time.time()
        
        # Start with all +1
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Phase 1: Long anneal from random start
        budget1 = min(40.0, hard_deadline - (time.time() - start) - 30.0)
        if budget1 > 2.0:
            h, c = _anneal(42, budget1, T_init=10.0, T_min=0.005)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 2: Multiple shorter restarts for diversity
        restart_id = 1
        while time.time() - start < hard_deadline - 25.0:
            sub = min(8.0, hard_deadline - (time.time() - start) - 25.0)
            if sub < 1.0:
                break
            h, c = _anneal(1000 + restart_id, sub, T_init=5.0, T_min=0.01)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1
        
        # Phase 3: Greedy polish with remaining time
        remaining = hard_deadline - (time.time() - start)
        if remaining > 5.0:
            h, c = _greedy_polish(global_best_h, remaining - 5.0)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Safety: ensure we return before hard deadline
        if time.time() - start > hard_deadline - 3.0:
            return global_best_h
        
        return global_best_h
    except Exception:
        return np.ones(N)
