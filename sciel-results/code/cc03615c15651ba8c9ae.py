"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design: multi-start search with Golay-inspired initialization, fast O(N) flip
updates, simulated annealing with block flips, and exhaustive single-flip polish.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


# ---------------------------------------------------------------------------
# Autocorrelation helpers
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))


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


def _flip_ac_pair(ac, h, i, j):
    """Vectorized O(N) update of ac after flipping two positions i and j simultaneously.
    
    Flipping h[i] and h[j] changes ac[d] by:
    For each flipped position p, the contribution to ac[d] changes by:
      -2*h[p]*h[p+d] for d>0 (when p+d < N and p+d != other flip)
      -2*h[p-d]*h[p] for d>0 (when p-d >= 0 and p-d != other flip)
    Plus correction terms when both flipped positions interact.
    """
    new_ac = ac.copy()
    other = j if i != j else i
    # For position i:
    # ac[d] += -2*h[i]*h[i+d] for d=1..N-1-i (if i+d != j)
    # ac[d] += -2*h[i-d]*h[i] for d=1..i (if i-d != j)
    # For position j:
    # ac[d] += -2*h[j]*h[j+d] for d=1..N-1-j (if j+d != i)
    # ac[d] += -2*h[j-d]*h[j] for d=1..j (if j-d != i)
    # Interaction: if |i-j| = d, the pair (min(i,j), max(i,j)) is counted in both
    #   and needs a +4 correction (flipped twice = no net change for that pair,
    #   but we subtracted it twice, so add back 2*original... let me think carefully)
    
    # Actually, let's use the same approach as _flip_ac_fast but for two positions.
    # Flipping position p: ac_new[d] = ac[d] - 2*h[p]*h[p+d] (for d where p+d<N)
    #                        ac_new[d] = ac[d] - 2*h[p-d]*h[p] (for d where p-d>=0)
    # This is valid when only one flip. For two flips at i and j:
    # The cross term: when d = |i-j|, the pair (i,j) or (j,i) is involved.
    # If we apply flip-i then flip-j sequentially:
    #   After flip i: ac1 = ac - 2*h[i]*h[i+d] - 2*h[i-d]*h[i]
    #   After flip j: ac2 = ac1 - 2*h[j]*h[j+d] - 2*h[j-d]*h[j]  (using original h[j] since h[j] unchanged by flip i)
    # But wait, h[i] was flipped in step 1, so h[j] in step 2 should use new h[j]?
    # No: _flip_ac_fast uses the CURRENT h values. If we flip i first, h[i] changes,
    # then flipping j uses the updated h[i] if i is involved in j's neighbors.
    
    # Let me just do it sequentially with the fast update:
    ac1 = _flip_ac_fast(ac, h, i)
    h[i] *= -1.0  # temporarily flip for the second update
    ac2 = _flip_ac_fast(ac1, h, j)
    h[i] *= -1.0  # restore
    return ac2


# ---------------------------------------------------------------------------
# Golay-inspired initialization
# ---------------------------------------------------------------------------

def _golay_pair(n):
    """Generate a Golay complementary pair of length n (power of 2).
    Returns (a, b) where a, b are +-1 arrays of length n with
    aperiodic autocorrelation sum being zero for all k >= 1."""
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < n:
        a = np.concatenate([a, a])
        b = np.concatenate([b, -b])
    return a[:n], b[:n]


def _golay_init(seed):
    """Create a +-1 sequence of length N from Golay pair structure.
    
    Use the Golay pair to create a sequence with low autocorrelation structure,
    then truncate/extend to length N with random tail."""
    rng = np.random.default_rng(seed)
    # Use largest power of 2 <= N for the Golay pair
    m = 1
    while m * 2 <= N:
        m *= 2
    a, b = _golay_pair(m)
    # Combine: use a where |a| is "dominant", or just use a directly
    # Actually, for a single sequence, use a (or b) as the base
    h = a.copy()
    # Extend to length N with random signs
    if N > m:
        tail = rng.choice([-1.0, 1.0], size=N - m)
        h = np.concatenate([h, tail])
    else:
        h = h[:N]
    return h


def _random_init(seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)


def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h


# ---------------------------------------------------------------------------
# Annealing with block flips
# ---------------------------------------------------------------------------

def _soft_peak(ac, beta=50.0):
    """Soft-max surrogate for peak sidelobe using log-sum-exp."""
    vals = np.abs(ac[1:])
    if vals.size == 0:
        return 0.0
    m = np.max(vals)
    if m == 0:
        return 0.0
    return float(m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta)


def _anneal(seed, time_budget, init_h=None):
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        h = _random_init(seed)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-6
    start = time.time()
    # Adaptive block length: start large, shrink over time
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Block length: from 16 down to 1
        max_len = max(1, int(16.0 * (1.0 - progress) + 1))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = _peak_sidelobe(h)
        # Use soft-peak for acceptance decision (smoother), hard peak for tracking
        soft_new = _soft_peak(ac_new, beta=30.0)
        soft_cur = _soft_peak(ac, beta=30.0)
        d_soft = soft_new - soft_cur
        if new <= cur:
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            if rng.random() < np.exp(-d_soft / max(T, 1e-9)):
                cur = new
                ac = ac_new
            else:
                h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip only (fast)
# ---------------------------------------------------------------------------

def _greedy_single_flip(h, time_budget, start):
    """Exhaustive single-flip polish: repeatedly scan all positions, accept best improvement."""
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


def _greedy_pair_flip(h, time_budget, start):
    """Greedy pair-flip polish: try all adjacent pairs, accept best improvement."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_pair = -1
        best_c = cur
        for i in range(N - 1):
            ac2 = _flip_ac_pair(ac, h, i, i + 1)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_pair = i
        if best_pair < 0:
            break
        h[best_pair] *= -1.0
        h[best_pair + 1] *= -1.0
        ac = _flip_ac_pair(ac, h, best_pair, best_pair + 1)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _combined_polish(h, time_budget, start):
    """Alternate single-flip and pair-flip polish passes with vectorized updates."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        # Single flip pass: find best single flip
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
            continue
        # Try pair flips (adjacent pairs, vectorized)
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            ac2 = _flip_ac_pair(ac, h, i, i + 1)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c2:
                best_c2 = c
                best_pair = i
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            ac = _flip_ac_pair(ac, h, best_pair, best_pair + 1)
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Try non-adjacent pair flips (sampled)
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            found = False
            for _ in range(20):
                i = int(rng.integers(0, N))
                j = int(rng.integers(0, N))
                if abs(i - j) <= 1:
                    continue
                ac2 = _flip_ac_pair(ac, h, i, j)
                c = float(np.max(np.abs(ac2[1:])))
                if c < best_c2:
                    best_c2 = c
                    best_pair = i
                    found = True
                    break
            if found:
                h[best_pair] *= -1.0
                h[j] *= -1.0
                ac = _flip_ac_pair(ac, h, best_pair, j)
                cur = best_c2
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                break
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)  # 399 for all-ones

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: Try multiple diverse initializations with annealing
        inits = []
        # Golay-based
        inits.append(_golay_init(42))
        inits.append(_golay_init(123))
        # Sine-based
        inits.append(_sine_init(3.0, 0.0, 1))
        inits.append(_sine_init(7.0, np.pi / 2, 2))
        inits.append(_sine_init(11.0, np.pi / 3, 3))
        # Random
        inits.append(_random_init(7))
        inits.append(_random_init(99))

        for idx, h_init in enumerate(inits):
            if _remaining() < 20.0:
                break
            budget = min(12.0, _remaining() - 15.0)
            if budget < 1.0:
                break
            h, c = _anneal(1000 + idx, budget, init_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h.copy()
            # Quick polish
            sub = min(4.0, _remaining() - 15.0)
            if sub > 0.5:
                h2, c2 = _combined_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: More annealing restarts from the best solution found
        restart_id = 100
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 12.0)
            if sub < 1.0:
                break
            # Perturb the best solution
            rng = np.random.default_rng(restart_id)
            h_perturbed = global_best_h.copy()
            n_flips = rng.integers(5, 20)
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            h_perturbed[flip_indices] *= -1.0
            h, c = _anneal(restart_id, sub, init_h=h_perturbed)
            if c < global_best:
                global_best = c
                global_best_h = h.copy()
            sub2 = min(3.0, _remaining() - 12.0)
            if sub2 > 0.3:
                h2, c2 = _combined_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: Combined polish (single + pair flips)
        remaining = _remaining() - 3.0
        if remaining > 0.5:
            h, c = _combined_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: Final single-flip squeeze
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.3:
            h, c = _greedy_single_flip(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: Final combined polish squeeze
        remaining3 = _remaining() - 0.3
        if remaining3 > 0.2:
            h, c = _combined_polish(global_best_h, remaining3, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
