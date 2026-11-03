"""Autoconvolution minimization via multi-restart simulated annealing with smooth surrogate and high-precision polish.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.  Evaluation is done with a zero-padded FFT so each candidate costs
a single O(L log L) transform (L=1024).

Strategy:
1. Diverse initializations (random, quadratic phases, sines)
2. Multi-restart simulated annealing with variable-length block flips,
   using a smooth log-sum-exp surrogate for the peak sidelobe to guide
   the annealing more effectively than the raw max.
3. Greedy multi-scale local polish (block flips of sizes 1,2,4,8,16,32)
4. Final high-precision single-bit polish with tighter tolerance
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _hard_score_from_ac(ac):
    """Compute hard score from precomputed autocorrelation."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate from precomputed autocorrelation."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score(h):
    """The actual metric: peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    return _hard_score_from_ac(ac)


def _smooth_score(h, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    ac = _autocorr_fast(h)
    return _smooth_score_from_ac(ac, beta)


def _anneal_round(h_init, seed, time_budget, beta_start=30.0, beta_end=80.0):
    """One round of simulated annealing with variable-length block flips.
    
    Uses smooth surrogate for acceptance decisions.
    Returns (best_h, best_hard).
    """
    rng = np.random.default_rng(seed)
    h = h_init.copy()
    ac = _autocorr_fast(h)
    cur_hard = _hard_score_from_ac(ac)
    cur_smooth = _smooth_score_from_ac(ac, beta_start)
    best_h = h.copy()
    best_hard = cur_hard

    T = 0.02
    T_min = 1e-5
    decay = 0.9997
    sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
    
    start_time = time.time()
    while time.time() - start_time < time_budget:
        # Adaptive beta: increase over time to sharpen the surrogate
        progress = min(1.0, (time.time() - start_time) / max(time_budget, 1e-9))
        beta = beta_start + (beta_end - beta_start) * progress

        if rng.random() < 0.70:
            size = int(rng.choice(sizes))
            start = int(rng.integers(0, N - size + 1))
            sl = slice(start, start + size)
            old = h[sl].copy()
            h[sl] *= -1.0
            ac = _autocorr_fast(h)
            new_hard = _hard_score_from_ac(ac)
            new_smooth = _smooth_score_from_ac(ac, beta)
            if new_hard < cur_hard:
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            else:
                dS = new_smooth - cur_smooth
                if dS <= 0 or rng.random() < np.exp(-dS / T):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                else:
                    h[sl] = old
        else:
            i = int(rng.integers(0, N))
            old = h[i]
            h[i] *= -1.0
            ac = _autocorr_fast(h)
            new_hard = _hard_score_from_ac(ac)
            new_smooth = _smooth_score_from_ac(ac, beta)
            if new_hard < cur_hard:
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            else:
                dS = new_smooth - cur_smooth
                if dS <= 0 or rng.random() < np.exp(-dS / T):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                else:
                    h[i] = old
        T = max(T_min, T * decay)

    return best_h, best_hard


def _greedy_polish(h, time_budget, start_time):
    """Greedy multi-scale local polish with block flips."""
    best_h = h.copy()
    best_hard = _hard_score(h)
    # Try larger blocks first to escape local minima, then smaller for refinement
    sizes_polish = [32, 16, 8, 4, 2, 1]
    
    for _pass in range(100):
        if time.time() > time_budget:
            break
        any_improve = False
        for size in sizes_polish:
            for start in range(0, N - size + 1):
                if time.time() > time_budget:
                    return best_h, best_hard
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                ac = _autocorr_fast(h)
                new_hard = _hard_score_from_ac(ac)
                if new_hard < best_hard - 1e-12:
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
                else:
                    h[sl] = old
        if not any_improve:
            break

    return best_h, best_hard


def _single_bit_polish(h, time_budget):
    """Final high-precision single-bit polish."""
    best_h = h.copy()
    best_hard = _hard_score(h)
    
    for _pass in range(25):
        if time.time() > time_budget:
            break
        any_improve = False
        for i in range(N):
            if time.time() > time_budget:
                break
            old = h[i]
            h[i] *= -1.0
            ac = _autocorr_fast(h)
            new_hard = _hard_score_from_ac(ac)
            if new_hard < best_hard - 1e-15:
                best_hard = new_hard
                best_h = h.copy()
                any_improve = True
            else:
                h[i] = old
        if not any_improve:
            break

    return best_h, best_hard


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (5 * N)),
            np.cos(2 * np.pi * idx ** 2 / (7 * N)),
        ]
        
        # Ensure signs are ±1
        inits = [np.sign(h) for h in inits]
        # Replace any zeros with +1
        for i in range(len(inits)):
            inits[i] = np.where(inits[i] == 0, 1.0, inits[i])

        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        # ---------- multi-restart simulated annealing ----------
        # Use remaining time for multiple anneal rounds
        time_elapsed = time.time() - t0
        remaining = deadline - 8.0 - time_elapsed  # leave 8s for polish
        
        # Number of anneal rounds: at least 2, up to 5
        num_rounds = max(2, min(5, int(remaining / 20.0)))
        time_per_round = (remaining - 8.0) / num_rounds if num_rounds > 0 else 0
        
        current_h = best_h.copy()
        
        for r in range(num_rounds):
            if time.time() > deadline - 8.0:
                break
            round_budget = time_per_round - 1.0
            if round_budget < 1.0:
                break
            
            # For subsequent rounds, perturb the current best to create diversity
            if r == 0:
                h_init = current_h.copy()
            else:
                h_init = current_h.copy()
                # Randomly flip 5-15% of signs to create a new starting point
                n_flip = int(rng.integers(N // 20, N // 5 + 1))
                flip_idx = rng.choice(N, n_flip, replace=False)
                h_init[flip_idx] *= -1.0
            
            seed = _SEED + 1000 + r * 7919
            h_round, hard_round = _anneal_round(h_init, seed, round_budget)
            
            if hard_round < best_hard:
                best_hard = hard_round
                best_h = h_round.copy()
                current_h = h_round.copy()
            else:
                # Keep the better of current best and round result
                current_h = h_round.copy() if hard_round < _hard_score(current_h) else current_h

        # ---------- greedy multi-scale local polish ----------
        polish_deadline = deadline - 5.0
        h = best_h.copy()
        best_h, best_hard = _greedy_polish(h, polish_deadline, t0)

        # ---------- final high-precision single-bit polish ----------
        h = best_h.copy()
        best_h, best_hard = _single_bit_polish(h, deadline - 1.0)

        # Ensure output is strictly ±1
        best_h = np.sign(best_h).astype(np.float64)
        best_h = np.where(best_h == 0, 1.0, best_h)

        return best_h
    except Exception:
        return np.ones(N)
