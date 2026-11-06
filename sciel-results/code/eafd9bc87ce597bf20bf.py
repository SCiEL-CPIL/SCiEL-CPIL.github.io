"""Additive bases with few representations - directed peak-reduction search.

Structural change: instead of random simulated annealing, use a directed search
that specifically targets the peak target and finds swaps that reduce it.
Precompute all swap deltas for O(1) evaluation per candidate swap.
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets
_KS = np.arange(_T0, _T1 + 1, dtype=np.int64)

_START = time.time()
_DEADLINE = 85.0


def _elapsed():
    return time.time() - _START


def _unordered_counts(x):
    """Compute unordered pair counts for targets [128, 256]."""
    y = np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")
    w = y[_T0:_T1 + 1].copy()
    is_even = (np.arange(_T0, _T1 + 1) % 2 == 0)
    mids = np.arange(_T0, _T1 + 1) // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _precompute_deltas(x):
    """Precompute delta for all (remove_i, add_j) swaps.
    Returns a 2D array delta[i,j] = change in u when removing i and adding j.
    Only valid where x[i]=1 and x[j]=0.
    """
    # For each target k, delta_ordered[k] = x_old[k-j] - x_old[k-i]
    # We can vectorize: for each (i,j), delta_ordered at position p (target T0+p) is:
    #   x[T0+p-j] (if 0<=T0+p-j<129) - x[T0+p-i] (if 0<=T0+p-i<129)
    
    # Build shift matrices
    # shift_add[j, p] = x[T0+p-j] if in range else 0
    # shift_rem[i, p] = x[T0+p-i] if in range else 0
    
    # T0+p ranges from 128 to 256, p from 0 to 128
    # T0+p-j ranges from 128-j to 256-j
    # Valid when 0 <= 128+p-j <= 128, i.e., j-128 <= p <= 128+j-128 = j
    # So p in [max(0, j-128), min(128, j)]
    
    shift_add = np.zeros((_LEN, _WIN), dtype=np.int64)
    shift_rem = np.zeros((_LEN, _WIN), dtype=np.int64)
    
    for j in range(_LEN):
        p_lo = max(0, j - 128)
        p_hi = min(128, j)
        if p_lo <= p_hi:
            # target = 128+p, index = 128+p-j
            idxs = 128 + np.arange(p_lo, p_hi + 1) - j
            shift_add[j, p_lo:p_hi+1] = x[idxs]
    
    for i in range(_LEN):
        p_lo = max(0, i - 128)
        p_hi = min(128, i)
        if p_lo <= p_hi:
            idxs = 128 + np.arange(p_lo, p_hi + 1) - i
            shift_rem[i, p_lo:p_hi+1] = x[idxs]
    
    # delta_ordered[i,j,p] = shift_add[j,p] - shift_rem[i,p]
    # = shift_add[j] - shift_rem[i] (broadcast)
    # shape: (_LEN, _LEN, _WIN)
    delta_ordered = shift_add[None, :, :] - shift_rem[:, None, :]
    
    # delta_diag: +1 at p=2j-128 if 0<=2j-128<=128, -1 at p=2i-128 if 0<=2i-128<=128
    delta_diag = np.zeros((_LEN, _LEN, _WIN), dtype=np.int64)
    for j in range(_LEN):
        p = 2 * j - _T0
        if 0 <= p < _WIN:
            delta_diag[:, j, p] += 1
    for i in range(_LEN):
        p = 2 * i - _T0
        if 0 <= p < _WIN:
            delta_diag[i, :, p] -= 1
    
    delta_u = (delta_ordered + delta_diag) // 2
    return delta_u


def _make_seed():
    """Build a valid seed using structured approach."""
    # Start with all evens (covers all even targets)
    x = np.zeros(_LEN, dtype=np.int64)
    for c in range(0, _LEN, 2):
        x[c] = 1
    
    u = _unordered_counts(x)
    
    # Add odds to cover odd targets
    rng = random.Random(12345)
    odds = list(range(1, _LEN, 2))
    rng.shuffle(odds)
    
    for c in odds:
        if _valid(u):
            break
        x[c] = 1
        u = _unordered_counts(x)
    
    # If still not valid, add more
    for c in range(_LEN):
        if _valid(u):
            break
        if x[c] == 0:
            x[c] = 1
            u = _unordered_counts(x)
    
    return x


def _directed_search(x, rng, max_time):
    """Directed search: find swaps that reduce the peak."""
    u = _unordered_counts(x)
    if not _valid(u):
        return x
    
    best_x = x.copy()
    best_peak = _peak(u)
    best_var = float(np.var(u))
    
    while _elapsed() < max_time:
        current_peak = _peak(u)
        if current_peak < best_peak:
            best_peak = current_peak
            best_var = float(np.var(u))
            best_x = x.copy()
            break
        
        # Find peak targets
        peak_targets = np.where(u == current_peak)[0]  # indices into u (0..128)
        
        # For each peak target, find swaps that reduce it
        # A swap (remove i, add j) changes u[p] by delta_u[i,j,p]
        # We want delta_u[i,j,p] < 0 for some peak p, and u_new all > 0
        
        in_A = np.where(x == 1)[0]
        not_in_A = np.where(x == 0)[0]
        
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
        
        # Precompute deltas for current x
        deltas = _precompute_deltas(x)  # shape (129, 129, 129)
        
        found_improvement = False
        best_swap = None
        best_new_peak = current_peak
        best_new_var = float(np.var(u))
        
        # For each peak target, check swaps that reduce it
        for pt in peak_targets:
            # We need delta_u[i, j, pt] < 0
            # And u_new = u + delta_u[i,j] must be all > 0
            # And new peak < current_peak (or same peak but lower variance)
            
            # Vectorized: for each (i,j), check if delta at pt < 0
            delta_at_pt = deltas[:, :, pt]  # shape (129, 129)
            
            # Only consider valid swaps: x[i]=1, x[j]=0
            valid_mask = (x[in_A][:, None] == 1) & (x[not_in_A][None, :] == 0)
            # Actually in_A and not_in_A are already filtered, so all are valid
            # But we need the full (129,129) mask
            full_mask = np.zeros((_LEN, _LEN), dtype=bool)
            full_mask[in_A[:, None], not_in_A[None, :]] = True
            
            candidates = np.where(full_mask & (delta_at_pt < 0))
            
            if len(candidates[0]) == 0:
                continue
            
            # Check each candidate (limit to top few for speed)
            n_cand = len(candidates[0])
            if n_cand > 200:
                # Sample
                idx = rng.sample(range(n_cand), 200)
                cand_i = candidates[0][idx]
                cand_j = candidates[1][idx]
            else:
                cand_i = candidates[0]
                cand_j = candidates[1]
            
            for i, j in zip(cand_i, cand_j):
                delta = deltas[i, j]
                u_new = u + delta
                if not np.all(u_new > 0):
                    continue
                new_peak = int(u_new.max())
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_new_var = float(np.var(u_new))
                    best_swap = (i, j, u_new)
                    found_improvement = True
                elif new_peak == best_new_peak:
                    new_var = float(np.var(u_new))
                    if new_var < best_new_var:
                        best_new_var = new_var
                        best_swap = (i, j, u_new)
                        found_improvement = True
        
        if found_improvement and best_swap is not None:
            i, j, u_new = best_swap
            x[i] = 0
            x[j] = 1
            u = u_new
        else:
            # No improvement found; try random swaps to escape
            i = rng.choice(in_A)
            j = rng.choice(not_in_A)
            delta = deltas[i, j]
            u_new = u + delta
            if not np.all(u_new > 0):
                # Try a few more
                found = False
                for _ in range(20):
                    i2 = rng.choice(in_A)
                    j2 = rng.choice(not_in_A)
                    delta2 = deltas[i2, j2]
                    u_new2 = u + delta2
                    if np.all(u_new2 > 0):
                        x[i2] = 0
                        x[j2] = 1
                        u = u_new2
                        found = True
                        break
                if not found:
                    break
            else:
                x[i] = 0
                x[j] = 1
                u = u_new
    
    return best_x


def _multi_restart_search(x, rng, max_time):
    """Multiple restarts with different perturbations."""
    best_x = x.copy()
    best_peak = _peak(_unordered_counts(x))
    
    while _elapsed() < max_time:
        # Perturb
        x_pert = best_x.copy()
        in_A = np.where(x_pert == 1)[0]
        not_in_A = np.where(x_pert == 0)[0]
        
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
        
        n_swaps = rng.randint(3, 10)
        for _ in range(n_swaps):
            i = rng.choice(in_A)
            j = rng.choice(not_in_A)
            x_pert[i] = 0
            x_pert[j] = 1
            in_A = np.where(x_pert == 1)[0]
            not_in_A = np.where(x_pert == 0)[0]
        
        # Check validity
        u_pert = _unordered_counts(x_pert)
        if not _valid(u_pert):
            continue
        
        # Run directed search with limited time
        remaining = max_time - _elapsed()
        if remaining < 3:
            break
        sa_time = min(remaining * 0.3, 15)
        x_result = _directed_search(x_pert, rng, max_time=_elapsed() + sa_time)
        
        u_result = _unordered_counts(x_result)
        if _valid(u_result):
            p_result = _peak(u_result)
            if p_result < best_peak:
                best_peak = p_result
                best_x = x_result.copy()
    
    return best_x


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Generate seed
    x = _make_seed()
    u = _unordered_counts(x)
    
    if not _valid(u):
        # Fallback: add elements until valid
        for c in range(_LEN):
            if _valid(_unordered_counts(x)):
                break
            x[c] = 1
    
    # Phase 2: Directed search
    remaining = _DEADLINE - _elapsed() - 5
    if remaining > 2:
        x = _directed_search(x, rng, max_time=_elapsed() + min(remaining * 0.4, 20))
    
    # Phase 3: Multi-restart
    remaining = _DEADLINE - _elapsed() - 3
    if remaining > 3:
        x = _multi_restart_search(x, rng, max_time=_elapsed() + min(remaining * 0.8, 40))
    
    # Phase 4: Final directed search
    remaining = _DEADLINE - _elapsed() - 1
    if remaining > 1:
        x = _directed_search(x, rng, max_time=_elapsed() + remaining)
    
    # Ensure validity
    u_final = _unordered_counts(x)
    if not _valid(u_final):
        for c in range(_LEN):
            if _valid(_unordered_counts(x)):
                break
            x[c] = 1
    
    result = [int(i) for i in np.where(x == 1)[0]]
    return result
