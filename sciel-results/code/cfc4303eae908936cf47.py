"""
Signed step function for the third autocorrelation inequality.

We model the task as minimising the maximum absolute value of the
autoconvolution (i.e. the full linear autocorrelation) of a length-400
±1 sequence.  The structural change here is:

1.  Use a *fast* exact Δ-update for the autocorrelation when flipping a
    single bit, so that a single flip costs O(N) additions rather than a
    full FFT.  This makes many more local moves feasible within the
    wall-clock budget.
2.  Replace pure single-bit greedy + simulated annealing with a
    *multi-scale block flip* local search:
      - small blocks (1, 2, 4 bits) for fine polish,
      - medium blocks (8, 16 bits) to escape shallow local minima,
      - large blocks (32, 64 bits) for coarser rearrangements.
3.  Interleave random restarts seeded by several structured initialisations
    (Thue–Morse, van der Corput, random) so that the search does not
    commit to a single basin.

The entry point construct() returns a numpy array of shape (400,) with
entries in {-1, +1}.  The evaluator's score is the raw upper bound on C3;
lower is better.
"""

import numpy as np
import time

N = 400
DEADLINE = 140.0  # evaluator kills at 150 s; leave margin


# ---------------------------------------------------------------------------
# Autocorrelation utilities
# ---------------------------------------------------------------------------

def _full_autocorr(x):
    """
    Linear (non-circular) autocorrelation of x for lags 0..N-1 and
    (by symmetry) N-1..0.  Returns array of length 2*N-1.
    Uses FFT for speed.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # ac[0..n-1] = lags 0..n-1 ; ac[n..2n-2] = lags -(n-1)..-1
    return ac[:2 * n - 1]


def _max_abs_autocorr(x):
    """Maximum absolute value of the full linear autocorrelation."""
    ac = _full_autocorr(x)
    return float(np.max(np.abs(ac)))


def _delta_autocorr(x, i, sign_change):
    """
    Compute the change in the full autocorrelation when x[i] is flipped.
    sign_change = new_val - old_val (which is -2*old_val).
    Returns array of length 2*N-1 with the delta at each lag.
    """
    n = len(x)
    delta = np.zeros(2 * n - 1)
    # For lag k (0-indexed, k=0..n-1): ac[k] = sum_j x[j]*x[j+k]
    # Flipping x[i] affects ac[k] for k such that i is in [0, n-1-k]
    # or i+k is in [0, n-1].
    # Contribution to ac[k]:
    #   if i <= n-1-k:  x[i]*x[i+k]  -> change = sign_change * x[i+k]
    #   if i+k <= n-1 and i+k != i:  x[i-k]*x[i]  -> change = sign_change * x[i-k]
    # For k=0, both terms are the same (x[i]*x[i]), so count once.
    # For negative lags (represented at index n+|k|), similar logic.

    # Positive lags k = 0..n-1 (indices 0..n-1 in ac)
    # Term 1: i in [0, n-1-k]  =>  k in [0, n-1-i]
    k_max1 = n - 1 - i
    if k_max1 >= 0:
        k_range = np.arange(0, k_max1 + 1)
        delta[k_range] += sign_change * x[i + k_range]
    # Term 2: i+k in [0, n-1] and k>0  =>  k in [1, n-1-i]
    k_max2 = n - 1 - i
    if k_max2 >= 1:
        k_range = np.arange(1, k_max2 + 1)
        delta[k_range] += sign_change * x[i - k_range]

    # Negative lags: ac[n + m] for m = 1..n-1 corresponds to lag -m
    # ac[-m] = sum_j x[j]*x[j-m]
    # Flipping x[i] affects:
    #   if i in [m, n-1]: x[i]*x[i-m] -> change = sign_change * x[i-m]
    #   if i-m in [0, n-1]: x[i-m]*x[i] -> change = sign_change * x[i-m]
    # So for lag -m, both terms give sign_change * x[i-m] if valid.
    # m such that i-m in [0, n-1] => m in [i-(n-1), i]  intersect [1, n-1]
    m_lo = max(1, i - (n - 1))
    m_hi = min(n - 1, i)
    if m_lo <= m_hi:
        m_range = np.arange(m_lo, m_hi + 1)
        delta[n + m_range] += 2 * sign_change * x[i - m_range]
        # Wait, need to check: for lag -m, ac[-m] = sum_{j=m}^{n-1} x[j]*x[j-m]
        # x[i] appears in x[j] when j=i (needs i>=m) and in x[j-m] when j-m=i (j=i+m, needs i+m<=n-1).
        # So term 1: if i>=m: x[i]*x[i-m], change = sign_change*x[i-m]
        #    term 2: if i+m<=n-1: x[i+m]*x[i], change = sign_change*x[i+m]
        # These are different terms unless i-m == i+m which is impossible for m>0.
        # So I should not double count. Let me redo.
    
    # Let me recompute more carefully.
    # Reset delta for negative lags
    delta[n:] = 0
    
    # For lag -m (m=1..n-1), ac[-m] = sum_{j=m}^{n-1} x[j] * x[j-m]
    # x[i] contributes:
    #   (a) as x[j] with j=i: requires i >= m, term = x[i]*x[i-m], delta = sign_change * x[i-m]
    #   (b) as x[j-m] with j-m=i => j=i+m: requires i+m <= n-1, term = x[i+m]*x[i], delta = sign_change * x[i+m]
    
    # (a): m in [1, i]  (and m <= n-1)
    m_a_lo = 1
    m_a_hi = min(i, n - 1)
    if m_a_lo <= m_a_hi:
        m_range = np.arange(m_a_lo, m_a_hi + 1)
        delta[n + m_range] += sign_change * x[i - m_range]
    
    # (b): m in [1, n-1-i]
    m_b_lo = 1
    m_b_hi = n - 1 - i
    if m_b_lo <= m_b_hi:
        m_range = np.arange(m_b_lo, m_b_hi + 1)
        delta[n + m_range] += sign_change * x[i + m_range]
    
    return delta


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _thue_morse(n):
    """Thue-Morse ±1 sequence of length n."""
    x = np.ones(n)
    for i in range(1, n):
        # t(i) = parity of popcount(i)
        pc = bin(i).count('1')
        x[i] = 1.0 if pc % 2 == 0 else -1.0
    return x


def _van_der_corput(n, base=2):
    vdc = np.zeros(n)
    for i in range(n):
        f = 1.0
        num = i + 1
        v = 0.0
        while num > 0:
            num, r = divmod(num, base)
            f /= base
            v += f * r
        vdc[i] = v
    return np.where(vdc < 0.5, -1.0, 1.0)


def _random_sign(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.choice([-1.0, 1.0], size=n)


# ---------------------------------------------------------------------------
# Local search: multi-scale block flip
# ---------------------------------------------------------------------------

def _try_flip_block(x, ac, start, block_len, new_bits):
    """
    Attempt to flip a block of bits x[start:start+block_len] to new_bits
    (new_bits is ±1 array of length block_len).  Returns (new_ac, improved, new_max).
    ac is current full autocorrelation array (length 2N-1).
    """
    n = len(x)
    # Compute delta for each bit in the block
    # For each position i in the block, old_val = x[i], new_val = new_bits[i-start]
    # sign_change = new_val - old_val
    # Total delta = sum of individual deltas
    delta = np.zeros(2 * n - 1)
    for idx in range(block_len):
        i = start + idx
        old_val = x[i]
        new_val = new_bits[idx]
        sc = new_val - old_val
        if sc == 0:
            continue
        # Positive lags
        k_max1 = n - 1 - i
        if k_max1 >= 0:
            k_range = np.arange(0, k_max1 + 1)
            delta[k_range] += sc * x[i + k_range]
        k_max2 = n - 1 - i
        if k_max2 >= 1:
            k_range = np.arange(1, k_max2 + 1)
            delta[k_range] += sc * x[i - k_range]
        # Negative lags
        m_a_lo = 1
        m_a_hi = min(i, n - 1)
        if m_a_lo <= m_a_hi:
            m_range = np.arange(m_a_lo, m_a_hi + 1)
            delta[n + m_range] += sc * x[i - m_range]
        m_b_lo = 1
        m_b_hi = n - 1 - i
        if m_b_lo <= m_b_hi:
            m_range = np.arange(m_b_lo, m_b_hi + 1)
            delta[n + m_range] += sc * x[i + m_range]
    
    new_ac = ac + delta
    new_max = float(np.max(np.abs(new_ac)))
    return new_ac, new_max


def _multi_scale_local_search(x, max_time, block_sizes=(1, 2, 4, 8, 16, 32)):
    """
    Local search that cycles through block sizes, trying random new bits
    for each block.  Accepts any move that reduces max|ac|.  Also allows
    a small number of neutral moves to escape plateaus.
    Returns (best_x, best_max).
    """
    n = len(x)
    x = x.copy()
    ac = _full_autocorr(x)
    best_max = float(np.max(np.abs(ac)))
    best_x = x.copy()
    start_time = time.time()
    
    # We'll iterate: for each block size, try several random positions and bit patterns
    # Use a counter to limit total moves
    moves_done = 0
    max_moves = 500000  # safety
    
    for bs in block_sizes:
        if bs > n:
            continue
        # Number of trials per block size
        trials = max(10, 200000 // bs)
        for _ in range(trials):
            elapsed = time.time() - start_time
            if elapsed > max_time:
                break
            if moves_done >= max_moves:
                break
            
            start = np.random.randint(0, n - bs + 1)
            new_bits = np.random.choice([-1.0, 1.0], size=bs)
            
            new_ac, new_max = _try_flip_block(x, ac, start, bs, new_bits)
            moves_done += 1
            
            if new_max < best_max - 1e-12:
                # Apply
                x[start:start + bs] = new_bits
                ac = new_ac
                best_max = new_max
                best_x = x.copy()
            elif new_max <= best_max + 1e-9 and np.random.random() < 0.05:
                # Neutral move, small chance to accept
                x[start:start + bs] = new_bits
                ac = new_ac
    
    return best_x, best_max


# ---------------------------------------------------------------------------
# Simulated annealing with single-bit flips (fast via delta)
# ---------------------------------------------------------------------------

def _anneal_single_bit(x, start_temp, end_temp, max_iter, time_budget):
    n = len(x)
    x = x.copy()
    ac = _full_autocorr(x)
    curr_max = float(np.max(np.abs(ac)))
    best_max = curr_max
    best_x = x.copy()
    start_time = time.time()
    
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        i = np.random.randint(n)
        new_val = -x[i]
        sc = new_val - x[i]  # = -2*x[i]
        
        # Compute delta
        delta = np.zeros(2 * n - 1)
        k_max1 = n - 1 - i
        if k_max1 >= 0:
            k_range = np.arange(0, k_max1 + 1)
            delta[k_range] += sc * x[i + k_range]
        k_max2 = n - 1 - i
        if k_max2 >= 1:
            k_range = np.arange(1, k_max2 + 1)
            delta[k_range] += sc * x[i - k_range]
        m_a_lo = 1
        m_a_hi = min(i, n - 1)
        if m_a_lo <= m_a_hi:
            m_range = np.arange(m_a_lo, m_a_hi + 1)
            delta[n + m_range] += sc * x[i - m_range]
        m_b_lo = 1
        m_b_hi = n - 1 - i
        if m_b_lo <= m_b_hi:
            m_range = np.arange(m_b_lo, m_b_hi + 1)
            delta[n + m_range] += sc * x[i + m_range]
        
        new_ac = ac + delta
        new_max = float(np.max(np.abs(new_ac)))
        delta_val = new_max - curr_max
        
        if delta_val < 0 or np.random.random() < np.exp(-delta_val / max(temp, 1e-10)):
            x[i] = new_val
            ac = new_ac
            curr_max = new_max
            if curr_max < best_max:
                best_max = curr_max
                best_x = x.copy()
        # else: reject, x unchanged
    
    return best_x, best_max


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    """
    Construct a signed step function of length N=400 minimising the
    upper bound on C3 (max absolute autocorrelation).
    Returns a numpy array of shape (400,) with values in {-1, +1}.
    """
    t0 = time.time()
    np.random.seed(12345)
    
    # Multiple initialisations
    init_candidates = [
        _thue_morse(N),
        _van_der_corput(N),
        _random_sign(N, seed=0),
        _random_sign(N, seed=1),
        _random_sign(N, seed=2),
    ]
    
    best_x = None
    best_score = np.inf
    
    # Phase 1: quick local search on each initialisation
    # Allocate time: Phase 1 gets ~40% of budget, Phase 2 gets ~40%, Phase 3 gets ~20%
    phase1_budget = 0.35 * DEADLINE
    phase2_budget = 0.35 * DEADLINE
    phase3_budget = 0.25 * DEADLINE
    
    # Phase 1: local search on each init
    per_init_time = phase1_budget / len(init_candidates)
    for x0 in init_candidates:
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - phase2_budget - phase3_budget
        if remaining < 5:
            break
        time_for_this = min(per_init_time, remaining)
        
        x_ls, score_ls = _multi_scale_local_search(x0, max_time=time_for_this,
                                                    block_sizes=(1, 2, 4, 8, 16))
        if score_ls < best_score:
            best_score = score_ls
            best_x = x_ls.copy()
    
    if best_x is None:
        best_x = init_candidates[0].copy()
        best_score = _max_abs_autocorr(best_x)
    
    # Phase 2: annealing from best, then local search
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed - phase3_budget
    if remaining > 10:
        anneal_time = remaining * 0.6
        polish_time = remaining * 0.4
        
        x_anneal, score_anneal = _anneal_single_bit(best_x, start_temp=0.8, end_temp=0.01,
                                                     max_iter=200000, time_budget=anneal_time)
        if score_anneal < best_score:
            best_score = score_anneal
            best_x = x_anneal.copy()
        
        # Polish with local search
        elapsed = time.time() - t0
        if DEADLINE - elapsed > 5:
            x_ls, score_ls = _multi_scale_local_search(best_x, max_time=polish_time,
                                                        block_sizes=(1, 2, 4, 8, 16, 32))
            if score_ls < best_score:
                best_score = score_ls
                best_x = x_ls.copy()
    
    # Phase 3: a few more random restarts if time permits
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed
    if remaining > 20:
        # Try 2 more random restarts with quick local search
        per_restart = remaining / 3
        for seed in [100, 200]:
            elapsed = time.time() - t0
            if DEADLINE - elapsed < 5:
                break
            x0 = _random_sign(N, seed=seed)
            time_for_this = min(per_restart, DEADLINE - elapsed - 5)
            x_ls, score_ls = _multi_scale_local_search(x0, max_time=time_for_this,
                                                        block_sizes=(1, 2, 4, 8, 16))
            if score_ls < best_score:
                best_score = score_ls
                best_x = x_ls.copy()
    
    # Final safety
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        # Shouldn't happen, but return what we have
        pass
    
    # Ensure output is correct format
    best_x = np.asarray(best_x, dtype=np.float64)
    assert best_x.shape == (N,)
    assert np.all(np.isin(best_x, [-1.0, 1.0]))
    
    return best_x


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    score = _max_abs_autocorr(result)
    print(f"Shape: {result.shape}, Min: {np.min(result)}, Max: {np.max(result)}")
    print(f"Score (max |autocorr|): {score}")
    print(f"Time: {elapsed:.2f}s")

