"""Search for a low-autocorrelation signed step function (400 equal intervals).

The C3 autoconvolution upper bound is governed by the peak aperiodic
autocorrelation sidelobe of the step heights. We minimize it using
analytic single-bit flip updates to the autocorrelation array, which is
significantly faster than FFT-based recalculation, allowing for more search
iterations within the time budget.
"""
import numpy as np
import time

N = 400


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 0
    for i in range(1, n):
        tm[i] = tm[i // 2] if (i % 2 == 0) else 1 - tm[i // 2]
    # Map 0 -> -1, 1 -> 1
    return 1.0 - 2.0 * tm.astype(np.float64)


def _calc_ac(h):
    """Calculate aperiodic autocorrelation using FFT (for initialization)."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.fft(h, fft_size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe_from_ac(ac):
    return float(np.max(np.abs(ac[1:])))


def _flip_delta(h, i):
    """
    Compute the delta to add to ac[1:] when flipping h[i].
    Delta[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid k, else 0.
    Returns array of length N where delta[0]=0, delta[1..N-1] are the changes.
    """
    delta = np.zeros(N, dtype=np.float64)
    
    # Contribution from h[i+k] for k=1 to N-1-i
    if i + 1 < N:
        delta[1 : N - i] += -2.0 * h[i] * h[i + 1 : N]
    
    # Contribution from h[i-k] for k=1 to i
    if i > 0:
        delta[1 : i + 1] += -2.0 * h[i] * h[i - 1 : -1 : -1]
    
    return delta


def _anneal(seed, time_budget, start_h=None):
    rng = np.random.default_rng(seed)
    
    if start_h is not None:
        h = start_h.copy()
    else:
        h = _thue_morse(N)
    
    ac = _calc_ac(h)
    cur = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best = cur
    
    # Adaptive temperature
    T = 2.0
    T_min = 1e-8
    cooling = 0.9999
    
    start = time.time()
    
    while time.time() - start < time_budget:
        i = int(rng.integers(0, N))
        
        # Apply flip
        old_h_i = h[i]
        h[i] = -old_h_i
        d = _flip_delta(h, i)  # This uses new h[i], but we need delta based on old
        # Wait, _flip_delta uses h[i] which is now -old_h_i
        # The formula is Delta = -2 * h_old * S
        # So we need to use old_h_i
        # Let me recalculate: actually _flip_delta computes -2*h[i]*S where h[i] is current
        # But we flipped, so h[i] = -old_h_i
        # So _flip_delta gives -2*(-old_h_i)*S = 2*old_h_i*S
        # But the actual change should be -2*old_h_i*S
        # So we need to negate it, or compute differently.
        
        # Let me just compute the delta correctly:
        # When we flip h[i] from old to -old, the change in R[k] is:
        # R_new[k] - R_old[k] = -2 * old_h_i * (h[i+k] + h[i-k])
        # So delta = -2 * old_h_i * (shift_right + shift_left)
        
        # Recompute properly
        delta = np.zeros(N, dtype=np.float64)
        if i + 1 < N:
            delta[1 : N - i] = -2.0 * old_h_i * h[i + 1 : N]
        if i > 0:
            delta[1 : i + 1] += -2.0 * old_h_i * h[i - 1 : -1 : -1]
        
        ac[1:] += delta
        new = _peak_sidelobe_from_ac(ac)
        
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_h_i
            ac[1:] -= delta
        
        T = max(T_min, T * cooling)
        
        # Periodically refresh temperature to escape local minima
        if rng.random() < 0.001:
            T = max(T, 0.5)
    
    return best_h, best


def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _calc_ac(h)
    cur = _peak_sidelobe_from_ac(ac)
    best = cur
    best_h = h.copy()
    
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        
        for i in range(N):
            old_h_i = h[i]
            h[i] = -old_h_i
            
            # Compute delta for this flip
            delta = np.zeros(N, dtype=np.float64)
            if i + 1 < N:
                delta[1 : N - i] = -2.0 * old_h_i * h[i + 1 : N]
            if i > 0:
                delta[1 : i + 1] += -2.0 * old_h_i * h[i - 1 : -1 : -1]
            
            ac[1:] += delta
            new = _peak_sidelobe_from_ac(ac)
            
            if new < best_c:
                best_c = new
                best_i = i
            
            # Revert
            h[i] = old_h_i
            ac[1:] -= delta
        
        if best_i < 0:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
            # Commit the best flip
            i = best_i
            old_h_i = h[i]
            h[i] = -old_h_i
            delta = np.zeros(N, dtype=np.float64)
            if i + 1 < N:
                delta[1 : N - i] = -2.0 * old_h_i * h[i + 1 : N]
            if i > 0:
                delta[1 : i + 1] += -2.0 * old_h_i * h[i - 1 : -1 : -1]
            ac[1:] += delta
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
    
    return best_h, best


def construct():
    try:
        hard_deadline = 145.0  # Safety margin before 150s kill
        start = time.time()
        
        # Initialize with Thue-Morse
        global_best_h = _thue_morse(N)
        global_best = _peak_sidelobe_from_ac(_calc_ac(global_best_h))
        
        # Phase 1: Long simulated annealing from TM
        budget1 = min(40.0, hard_deadline - (time.time() - start) - 10.0)
        if budget1 > 2.0:
            h, c = _anneal(12345, budget1)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Phase 2: Multiple shorter restarts with random perturbations
        restart_id = 1
        while time.time() - start < hard_deadline - 10.0:
            sub = min(8.0, hard_deadline - (time.time() - start) - 10.0)
            if sub < 1.0:
                break
            
            # Start from current best with random perturbation
            h_init = global_best_h.copy()
            rng_perturb = np.random.default_rng(100000 + restart_id)
            n_flips = int(rng_perturb.integers(5, 20))
            flip_indices = rng_perturb.choice(N, size=n_flips, replace=False)
            h_init[flip_indices] *= -1.0
            
            h, c = _anneal(200000 + restart_id, sub, start_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1
        
        # Phase 3: Final greedy single-flip polish
        remaining = hard_deadline - (time.time() - start) - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
