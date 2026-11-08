import numpy as np
import time

N = 50

def _compute_score(h):
    """Compute the ratio (L2^2) / (L1 * sup_autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 <= 0 or l2_sq <= 0:
        return 0.0
    # Autoconvolution: for each lag k, sum h[i]*h[i+k]
    # Use FFT for efficiency
    n = 2 * N
    f = np.zeros(n, dtype=float)
    f[:N] = h
    F = np.fft.fft(f)
    conv = np.real(np.fft.ifft(F * F))
    # conv[k] for k=0..N-1 gives sum_i h[i]*h[i+k] (circular, but with zero-padding it's linear)
    # Actually with zero-padding to 2N, conv[0..N-1] are the linear autocorrelation lags 0..N-1
    # And conv[N..2N-1] are negative lags
    # The sup norm is the max over all lags
    sup_conv = np.max(conv)
    if sup_conv <= 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _seed_score():
    """Score of the all-ones seed."""
    h = np.ones(N)
    return _compute_score(h)

_SEED_SCORE = _seed_score()

def _single_spike(pos):
    """Create a single spike at position pos."""
    h = np.zeros(N)
    h[pos] = 1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_h = None
    best_val = 0.0
    
    # Try all single-spike positions
    for pos in range(N):
        h = _single_spike(pos)
        val = _compute_score(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        if time.time() > deadline:
            break
    
    if best_h is None:
        return [1.0] * N
    
    # Local search: try small perturbations around the best spike
    # Try moving the spike, or splitting into two close spikes
    for pos in range(N):
        if time.time() > deadline:
            break
        # Single spike at pos
        h = _single_spike(pos)
        val = _compute_score(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        
        # Two adjacent spikes with varying weights
        for j in range(pos + 1, min(pos + 5, N)):
            if time.time() > deadline:
                break
            for w in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
                h2 = np.zeros(N)
                h2[pos] = 1.0
                h2[j] = w
                val2 = _compute_score(h2)
                if val2 > best_val:
                    best_val = val2
                    best_h = h2.copy()
    
    # Try a few more structured patterns if time permits
    if time.time() < deadline - 10:
        # Try a few random sparse patterns
        np.random.seed(42)
        for trial in range(200):
            if time.time() > deadline:
                break
            k = np.random.randint(1, 6)  # 1 to 5 nonzero entries
            idx = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            for i in idx:
                h[i] = np.random.uniform(0.5, 2.0)
            val = _compute_score(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    # Return the best heights found
    return list(best_h)
