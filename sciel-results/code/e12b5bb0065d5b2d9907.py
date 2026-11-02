"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design family (kept): simulated annealing + greedy polish, now with
  * an O(N) incremental autocorrelation update for single-flips (no FFT per trial),
  * a smooth epigraph / soft-peak (log-sum-exp) surrogate for exploration,
  * an ADMM / projected-gradient pass on the continuous relaxation h in [-1,1],
  * a high-precision greedy single-flip + adjacent-pair-flip polish,
  * an explicit 140 s deadline guard (10 s safety under the 150 s kill).
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))


def _soft_peak(ac, tau):
    """Smooth epigraph surrogate: log-sum-exp of ac[k]^2/tau over k>=1."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


def _flip_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i]."""
    new_ac = ac.copy()
    hi = h[i]
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
    k_lo = i
    if k_lo >= 1:
        new_ac[1:k_lo + 1] += -2.0 * hi * h[i - k_lo:i]
    return new_ac


def _flip_pair_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i] and h[i+1]."""
    # Flip i first
    new_ac = _flip_ac_fast(ac, h, i)
    # Now flip i+1 on the updated sequence
    h_tmp = h.copy()
    h_tmp[i] *= -1.0
    new_ac = _flip_ac_fast(new_ac, h_tmp, i + 1)
    return new_ac


def _flip_block_ac(ac, h, start, length):
    """Update ac after flipping a contiguous block of length 'length' starting at 'start'.
    This is O(N * length) but for small length it's fast."""
    if length == 1:
        return _flip_ac_fast(ac, h, start)
    if length == 2:
        return _flip_pair_ac_fast(ac, h, start)
    # General case: flip each element in the block
    h_tmp = h.copy()
    h_tmp[start:start+length] *= -1.0
    new_ac = ac.copy()
    for j in range(length):
        idx = start + j
        hi = h[idx]  # original value
        k_hi = N - 1 - idx
        if k_hi >= 0:
            # Contribution from (idx, idx+k) for k=0..k_hi
            # But we need to be careful: some of these indices are also being flipped
            # For correctness, we should compute the delta based on original h
            # Actually, the standard flip formula works if we apply sequentially
            # Let's just use the sequential approach
            pass
    # Sequential approach
    h_tmp = h.copy()
    new_ac = ac.copy()
    for j in range(length):
        idx = start + j
        new_ac = _flip_ac_fast(new_ac, h_tmp, idx)
        h_tmp[idx] *= -1.0
    return new_ac


# ---------------------------------------------------------------------------
# Continuous relaxation: projected-gradient / ADMM on soft peak
# ---------------------------------------------------------------------------

def _soft_peak_grad(ac, tau):
    """Gradient of soft peak f = log sum e^{ac[k]^2/tau} wrt ac (k>=1)."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    e = np.exp(v - m)
    p = e / float(np.sum(e))          # softmax weights
    # d f / d ac[k] = 2*ac[k]/tau * p[k]
    g = np.zeros(N)
    g[1:] = (2.0 / tau) * ac[1:] * p
    return g


def _continuous_relax(h0, time_budget, start):
    """Projected-gradient on the continuous relaxation h in [-1,1].

    Objective: soft peak of the autocorrelation (smooth epigraph surrogate).
    Gradient wrt h_i from the autocorrelation gradient wrt ac.
    Returns a +-1 sequence (rounded) and its peak.
    """
    rng = np.random.default_rng(7)
    h = h0.astype(np.float64).copy()
    # start from a rounded +-1 but allow continuous moves
    tau = 8.0
    step = 0.05
    ac = _ac_full(h)
    f = _soft_peak(ac, tau)
    best_f = f
    best_h = h.copy()
    while time.time() - start < time_budget:
        g_ac = _soft_peak_grad(ac, tau)
        # gradient of ac[k] = sum_j h[j]h[j+k] wrt h[i]:
        # d ac[k]/d h[i] = h[i+k] (if i+k<N) + h[i-k] (if i-k>=0)
        grad_h = np.zeros(N)
        for k in range(1, N):
            gk = g_ac[k]
            if gk == 0.0:
                continue
            # term h[i]*h[i+k]: for i in [0, N-1-k], derivative h[i+k]
            grad_h[0:N - k] += gk * h[k:N]
            # term h[i-k]*h[i]: for i in [k, N-1], derivative h[i-k]
            grad_h[k:N] += gk * h[0:N - k]
        # projected gradient step
        h = h - step * grad_h
        h = np.clip(h, -1.0, 1.0)
        ac = _ac_full(h)
        nf = _soft_peak(ac, tau)
        if nf < best_f:
            best_f = nf
            best_h = h.copy()
        # anneal tau down, occasional restart of step
        tau = max(1.0, tau * 0.99)
        step = 0.05 + 0.02 * (1.0 - min(1.0, tau / 8.0))
    # round to +-1
    hr = np.where(best_h >= 0.0, 1.0, -1.0)
    return hr, _peak_sidelobe(hr)


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips (uses soft surrogate to accept)
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    tau = 8.0
    start = time.time()
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = _peak_sidelobe(h)
        # accept using soft-surrogate difference to smooth the landscape
        d = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
        if new <= cur or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99997)
        tau = max(1.0, tau * 0.9995)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip (O(N) update) + adjacent-pair-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        # --- single-flip sweep (fast O(N) update) ---
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_ac_fast(ac, h, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            ac2 = _flip_pair_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c2:
                best_c2 = c
                best_pair = i
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            ac = _ac_full(h)
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
        if best_i < 0 and best_pair < 0:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with the O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h[best_i] *= -1.0
        ac = _flip_ac_fast(ac, h, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: continuous relaxation (ADMM / projected gradient) to seed
        # a good +-1 starting point, then a quick polish.
        budget = min(18.0, _remaining() - 25.0)
        if budget > 3.0:
            h, c = _continuous_relax(global_best_h, budget, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(6.0, _remaining() - 25.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: long initial anneal from a random +-1 start.
        budget = min(24.0, _remaining() - 22.0)
        if budget > 2.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 22.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 3: multiple shorter restarts for diversity.
        restart_id = 1
        while _remaining() > 18.0:
            sub = min(12.0, _remaining() - 18.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 18.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 4: final high-precision single-flip polish.
        remaining = _remaining() - 3.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: one more polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
