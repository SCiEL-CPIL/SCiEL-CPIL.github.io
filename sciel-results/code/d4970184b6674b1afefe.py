"""Additive bases with few representations - structural rewrite.

Key structural change: replace the ILP (which is too slow for the 90s budget to be useful)
with a fast simulated-annealing / multi-restart local search that operates on a compact
representation. The key insight from the cell findings is that a variance-aware objective
flattens the distribution and helps escape local optima. We combine:
  1. A greedy balanced seed that minimizes peak early.
  2. Fast delta-based evaluation (no full convolve per step).
  3. Simulated annealing with variance-aware acceptance.
  4. Multiple restarts with different random seeds.
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets

_START = time.time()
_DEADLINE = 85.0


def _elapsed():
    return time.time() - _START


def _unordered_counts(x):
    """Compute unordered pair counts for targets [128, 256]."""
    y = np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")
    w = y[_T0:_T1 + 1].copy()
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _delta_counts(x, add_idx, remove_idx):
    """
    Compute the change in unordered counts when we add add_idx to A and remove remove_idx from A.
    Returns (delta_u, new_valid) where delta_u is the change vector for targets [128,256].
    """
    # If add_idx == remove_idx, no change
    if add_idx == remove_idx:
        return np.zeros(_WIN, dtype=np.int64)
    
    # The change in ordered convolution when we toggle x[add_idx] += 1, x[remove_idx] -= 1:
    # delta_ordered[k] = x[k - add_idx] + x[k - remove_idx] - 1 (if add and remove differ)
    # More precisely:
    # New ordered count = old + (x[k-add] - x[k-remove]) - (if add==remove then 0, else 0)
    # Wait, let's think carefully.
    # ordered[k] = sum_{i+j=k} x_i x_j
    # When we set x[a] from 0->1 and x[r] from 1->0:
    # delta_ordered[k] = sum_{i+j=k} (delta_i x_j + x_i delta_j + delta_i delta_j)
    # where delta_a = +1, delta_r = -1, others 0.
    # = x[k-a] - x[k-r] + (if a+r==k then -1 else 0)  ... wait no
    # Actually: delta_ordered[k] = sum_j delta_x[k-j] * x_j + sum_i x_i * delta_x[k-i] + sum_{i,j} delta_x[k-i-j]... 
    # This is getting complicated. Let me just use the direct formula.
    
    # Simpler: the change in ordered count at position k is:
    # For adding a: contributes x[k-a] (for k-a in [0,128])
    # For removing r: contributes -x[k-r] (for k-r in [0,128])
    # Cross term: if a + r == k, then we lose the x_a * x_r term... but x_a was 0 and x_r was 1,
    # so originally x_a*x_r = 0, after: x_a=1, x_r=0, so x_a*x_r = 0. No change from cross.
    # Wait, actually the cross term in the bilinear form: when both change, the product term
    # x_a * x_r changes from 0*1=0 to 1*0=0. So no cross effect.
    # But we also have the self terms: x_a^2 changes from 0 to 1, x_r^2 changes from 1 to 0.
    
    # So delta_ordered[k] = x[k-a] - x[k-r] + (1 if k == 2*a else 0) - (1 if k == 2*r else 0)
    # where x[k-a] is the OLD value of x at position k-a.
    
    delta = np.zeros(_WIN, dtype=np.int64)
    ks = np.arange(_T0, _T1 + 1)
    
    # Contribution from adding a
    for idx, k in enumerate(ks):
        j = k - add_idx
        if 0 <= j < _LEN:
            delta[idx] += x[j]
    # Self term for a
    if 2 * add_idx >= _T0 and 2 * add_idx <= _T1:
        idx = 2 * add_idx - _T0
        delta[idx] += 1
    
    # Contribution from removing r
    for idx, k in enumerate(ks):
        j = k - remove_idx
        if 0 <= j < _LEN:
            delta[idx] -= x[j]
    # Self term for r
    if 2 * remove_idx >= _T0 and 2 * remove_idx <= _T1:
        idx = 2 * remove_idx - _T0
        delta[idx] -= 1
    
    # Now convert ordered delta to unordered delta
    # U_k = (ordered_k + diag_k) / 2
    # delta_U_k = (delta_ordered_k + delta_diag_k) / 2
    # delta_diag_k = 1 if k even and x[k/2] changes from 0 to 1 (add) or 1 to 0 (remove)
    # For add: if k == 2*a, diag increases by 1
    # For remove: if k == 2*r, diag decreases by 1
    
    # But wait, we already included the self terms in delta_ordered. Let me reconsider.
    # Actually, ordered[k] includes the term x_{k/2}^2 when k is even.
    # So the self term IS part of ordered. Let me redo.
    
    # ordered[k] = sum_{i=0}^{128} x_i * x_{k-i}
    # When x_a goes 0->1: delta from this is x_{k-a} (old value)
    # When x_r goes 1->0: delta from this is -x_{k-r} (old value)
    # The cross term x_a * x_r: old = 0*1 = 0, new = 1*0 = 0. No change.
    # So delta_ordered[k] = x_old[k-a] - x_old[k-r]
    
    # For unordered: U_k = (ordered_k + (1 if k even and x_{k/2}=1 else 0)) / 2
    # delta_U_k = (delta_ordered_k + delta_diag_k) / 2
    # delta_diag_k = +1 if k==2*a (x_a was 0, now 1), -1 if k==2*r (x_r was 1, now 0)
    
    # So:
    delta_ordered = np.zeros(_WIN, dtype=np.int64)
    for idx, k in enumerate(ks):
        j = k - add_idx
        if 0 <= j < _LEN:
            delta_ordered[idx] += x[j]
        j2 = k - remove_idx
        if 0 <= j2 < _LEN:
            delta_ordered[idx] -= x[j2]
    
    delta_diag = np.zeros(_WIN, dtype=np.int64)
    if 2 * add_idx >= _T0 and 2 * add_idx <= _T1:
        delta_diag[2 * add_idx - _T0] += 1
    if 2 * remove_idx >= _T0 and 2 * remove_idx <= _T1:
        delta_diag[2 * remove_idx - _T0] -= 1
    
    delta_u = (delta_ordered + delta_diag) // 2
    # Note: integer division should be exact since both terms have same parity
    
    return delta_u


def _make_greedy_seed(rng):
    """Build a valid seed using greedy coverage with load balancing."""
    x = np.zeros(_LEN, dtype=np.int64)
    u = _unordered_counts(x)
    
    # Greedy: repeatedly add the element that covers the most uncovered targets
    # while minimizing the increase in peak
    for _ in range(200):
        if _valid(u):
            break
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_score = -1
        current_peak = _peak(u)
        
        for c in range(_LEN):
            if x[c] == 1:
                continue
            # Compute what u would look like
            delta = _delta_counts(x, c, -1)  # add c, remove nothing
            # Actually, let's just compute directly for simplicity in seed
            x[c] = 1
            u_new = _unordered_counts(x)
            x[c] = 0
            
            if not _valid(u_new) and not np.all(u_new > 0):
                # Check if it helps coverage
                pass
            
            # Score: prefer covering more uncovered, penalize peak increase
            new_peak = _peak(u_new)
            covered = np.sum(u_new > 0) - np.sum(u > 0)
            score = covered * 10 - max(0, new_peak - current_peak) * 2
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c is None:
            break
        x[best_c] = 1
        u = _unordered_counts(x)
    
    return x


def _make_balanced_seed(rng):
    """Build a seed that tries to balance representation counts."""
    x = np.zeros(_LEN, dtype=np.int64)
    u = _unordered_counts(x)
    
    # Start with a reasonable base set
    # Use a structured approach: take every other element plus some extras
    base = list(range(0, _LEN, 2))
    rng.shuffle(base)
    for c in base:
        x[c] = 1
    u = _unordered_counts(x)
    
    # Now greedily fix uncovered targets
    for _ in range(300):
        if _valid(u):
            break
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_gain = -1
        current_peak = _peak(u)
        
        for c in range(_LEN):
            if x[c] == 1:
                continue
            x[c] = 1
            u_new = _unordered_counts(x)
            x[c] = 0
            
            gain = np.sum((u_new > 0) & (u == 0))
            new_peak = _peak(u_new)
            penalty = max(0, new_peak - current_peak)
            score = gain * 5 - penalty
            if score > best_gain:
                best_gain = score
                best_c = c
        
        if best_c is None:
            break
        x[best_c] = 1
        u = _unordered_counts(x)
    
    return x


def _simulated_annealing(x, rng, max_time, seed_offset=0):
    """Simulated annealing with variance-aware acceptance."""
    u = _unordered_counts(x)
    if not _valid(u):
        return x
    
    current_peak = _peak(u)
    current_var = float(np.var(u))
    best_x = x.copy()
    best_peak = current_peak
    best_var = current_var
    
    T0 = 5.0
    T1 = 0.1
    steps_per_iter = 50
    
    # Precompute in_A and not_in_A
    in_A = np.where(x == 1)[0]
    not_in_A = np.where(x == 0)[0]
    
    step = 0
    total_steps = 500000
    T = T0
    
    while _elapsed() < max_time and step < total_steps:
        step += 1
        # Anneal temperature
        T = T0 * (T1 / T0) ** (step / total_steps)
        
        # Choose a move: swap one in, one out
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
        
        i = rng.choice(in_A)
        j = rng.choice(not_in_A)
        
        if i == j:
            continue
        
        # Compute delta
        delta = _delta_counts(x, j, i)  # add j, remove i
        u_new = u + delta
        
        if not np.all(u_new > 0):
            continue
        
        new_peak = int(u_new.max())
        new_var = float(np.var(u_new))
        
        # Objective: minimize peak, then variance
        # Accept if better, or with probability exp(-delta_obj / T)
        old_obj = current_peak * 1000 + current_var
        new_obj = new_peak * 1000 + new_var
        delta_obj = new_obj - old_obj
        
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
            # Accept
            x[i] = 0
            x[j] = 1
            u = u_new
            current_peak = new_peak
            current_var = new_var
            
            # Update in_A and not_in_A
            in_A = np.where(x == 1)[0]
            not_in_A = np.where(x == 0)[0]
            
            if current_peak < best_peak or (current_peak == best_peak and current_var < best_var):
                best_peak = current_peak
                best_var = current_var
                best_x = x.copy()
    
    return best_x


def _fast_local_search(x, rng, max_time):
    """Fast local search: only accept improvements, no temperature."""
    u = _unordered_counts(x)
    if not _valid(u):
        return x
    
    current_peak = _peak(u)
    current_var = float(np.var(u))
    best_x = x.copy()
    best_peak = current_peak
    best_var = current_var
    
    in_A = np.where(x == 1)[0]
    not_in_A = np.where(x == 0)[0]
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while _elapsed() < max_time and no_improve_count < max_no_improve:
        if len(in_A) == 0 or len(not_in_A) == 0:
            break
        
        i = rng.choice(in_A)
        j = rng.choice(not_in_A)
        
        if i == j:
            no_improve_count += 1
            continue
        
        delta = _delta_counts(x, j, i)
        u_new = u + delta
        
        if not np.all(u_new > 0):
            no_improve_count += 1
            continue
        
        new_peak = int(u_new.max())
        new_var = float(np.var(u_new))
        
        old_obj = current_peak * 1000 + current_var
        new_obj = new_peak * 1000 + new_var
        
        if new_obj < old_obj:
            x[i] = 0
            x[j] = 1
            u = u_new
            current_peak = new_peak
            current_var = new_var
            in_A = np.where(x == 1)[0]
            not_in_A = np.where(x == 0)[0]
            no_improve_count = 0
            
            if current_peak < best_peak or (current_peak == best_peak and current_var < best_var):
                best_peak = current_peak
                best_var = current_var
                best_x = x.copy()
        else:
            no_improve_count += 1
    
    return best_x


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Generate multiple seeds and pick the best
    seeds = []
    seeds.append(_make_greedy_seed(rng))
    seeds.append(_make_balanced_seed(rng))
    
    # Also try a few random structured seeds
    for _ in range(3):
        x = np.zeros(_LEN, dtype=np.int64)
        # Random subset of size ~40-60
        size = rng.randint(40, 60)
        chosen = rng.sample(range(_LEN), size)
        for c in chosen:
            x[c] = 1
        # Fix uncovered
        u = _unordered_counts(x)
        for _ in range(200):
            if _valid(u):
                break
            uncovered = np.where(u == 0)[0] + _T0
            if len(uncovered) == 0:
                break
            best_c = None
            best_gain = -1
            for c in range(_LEN):
                if x[c] == 1:
                    continue
                x[c] = 1
                u_new = _unordered_counts(x)
                x[c] = 0
                gain = np.sum((u_new > 0) & (u == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is None:
                break
            x[best_c] = 1
            u = _unordered_counts(x)
        if _valid(_unordered_counts(x)):
            seeds.append(x)
    
    # Pick the best seed
    best_seed = None
    best_seed_peak = float('inf')
    for s in seeds:
        u_s = _unordered_counts(s)
        if _valid(u_s):
            p = _peak(u_s)
            if p < best_seed_peak:
                best_seed_peak = p
                best_seed = s
    
    if best_seed is None:
        # Fallback
        best_seed = np.ones(_LEN, dtype=np.int64)
    
    best_x = best_seed.copy()
    best_peak = best_seed_peak
    best_var = float(np.var(_unordered_counts(best_x)))
    
    # Phase 2: Local search on the best seed
    remaining = _DEADLINE - _elapsed() - 5
    if remaining > 2:
        x_ls = _fast_local_search(best_x.copy(), rng, max_time=min(remaining * 0.3, 15))
        u_ls = _unordered_counts(x_ls)
        if _valid(u_ls):
            p_ls = _peak(u_ls)
            v_ls = float(np.var(u_ls))
            if p_ls < best_peak or (p_ls == best_peak and v_ls < best_var):
                best_peak = p_ls
                best_var = v_ls
                best_x = x_ls.copy()
    
    # Phase 3: Simulated annealing with multiple restarts
    remaining = _DEADLINE - _elapsed() - 3
    if remaining > 3:
        n_restarts = max(1, int(remaining / 10))
        for r in range(n_restarts):
            if _elapsed() > _DEADLINE - 3:
                break
            # Perturb the current best
            x_pert = best_x.copy()
            # Make a few random swaps
            in_A = np.where(x_pert == 1)[0]
            not_in_A = np.where(x_pert == 0)[0]
            if len(in_A) > 0 and len(not_in_A) > 0:
                for _ in range(rng.randint(3, 8)):
                    i = rng.choice(in_A)
                    j = rng.choice(not_in_A)
                    x_pert[i] = 0
                    x_pert[j] = 1
                    in_A = np.where(x_pert == 1)[0]
                    not_in_A = np.where(x_pert == 0)[0]
            
            # Run SA on the perturbed solution
            sa_time = min(remaining - (r + 1) * 5, 15)
            if sa_time < 2:
                break
            x_sa = _simulated_annealing(x_pert, rng, max_time=sa_time, seed_offset=r)
            u_sa = _unordered_counts(x_sa)
            if _valid(u_sa):
                p_sa = _peak(u_sa)
                v_sa = float(np.var(u_sa))
                if p_sa < best_peak or (p_sa == best_peak and v_sa < best_var):
                    best_peak = p_sa
                    best_var = v_sa
                    best_x = x_sa.copy()
    
    # Phase 4: Final local search polish
    remaining = _DEADLINE - _elapsed() - 1
    if remaining > 1:
        x_final = _fast_local_search(best_x.copy(), rng, max_time=remaining)
        u_final = _unordered_counts(x_final)
        if _valid(u_final):
            p_final = _peak(u_final)
            v_final = float(np.var(u_final))
            if p_final < best_peak or (p_final == best_peak and v_final < best_var):
                best_peak = p_final
                best_var = v_final
                best_x = x_final.copy()
    
    # Convert to sorted list
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
