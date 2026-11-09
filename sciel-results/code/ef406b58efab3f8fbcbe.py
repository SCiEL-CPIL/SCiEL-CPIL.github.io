# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0
_DEADLINE_CHECK_EVERY = 10000


def _sim_score(item, bins, params):
    """Score each bin for the given item. Higher score = more preferred.
    
    params: (w_worst, w_best, w_next, w_frag)
      w_worst: weight for worst-fit (prefer most remaining capacity)
      w_best: weight for best-fit (prefer least remaining capacity)
      w_next: weight for 'next item fit' (prefer bins where next item also fits)
      w_frag: weight for fragmentation penalty (prefer bins that won't create tiny leftovers)
    """
    w_worst, w_best, w_next, w_frag = params
    rem = bins - item
    cap = bins.max() + 1e-9
    
    # Normalize remaining capacity
    fr = np.clip(rem / cap, 0.0, 1.0)
    
    # Worst-fit component: prefer bins with most remaining
    worst_score = fr
    
    # Best-fit component: prefer bins with least remaining (but still fit)
    best_score = 1.0 - fr
    
    # Next-item fit: estimate if a typical next item (median of current items ~ cap/4) fits
    # This is a proxy: prefer bins where remaining is either very small or can fit another item
    typical_next = cap * 0.25
    next_fits = (rem >= typical_next).astype(np.float64)
    next_score = next_fits
    
    # Fragmentation: penalize bins where remaining is a small fraction (hard to use)
    # Sweet spot: remaining is either near 0 (bin almost full) or large enough for another item
    frag_penalty = np.where((fr > 0.05) & (fr < 0.3), 1.0, 0.0)
    frag_score = 1.0 - frag_penalty
    
    score = (w_worst * worst_score + w_best * best_score + 
             w_next * next_score + w_frag * frag_score)
    return score


def _simulate_fast(items, cap, params):
    """Fast online bin-packing simulation. Returns num_bins_used."""
    n = len(items)
    max_bins = n + 1
    bins = np.full((max_bins,), cap, dtype=np.float64)
    used = 0
    
    for i in range(n):
        if i % _DEADLINE_CHECK_EVERY == 0 and time.time() > _DEADLINE:
            return used
        item = items[i]
        if item > cap:
            continue
        if used == 0:
            used = 1
            bins[0] = cap - item
            continue
        rem = bins[:used] - item
        mask = rem >= -1e-9
        if not mask.any():
            used += 1
            bins[used - 1] = cap - item
            continue
        feasible_bins = bins[:used][mask]
        scores = _sim_score(item, feasible_bins, params)
        j = np.argmax(scores)
        feasible_indices = np.where(mask)[0]
        bins[feasible_indices[j]] -= item
    return used


def _l1_lb(items, cap):
    """L1 lower bound: ceil(sum(items) / cap)."""
    return int(np.ceil(np.sum(items) / cap))


def _generate_weibull(n, cap, seed):
    """Generate a Weibull-distributed item sequence."""
    rng = np.random.RandomState(seed)
    shape, scale = 0.8, 15.0
    items = rng.weibull(shape, n) * scale
    items = np.clip(items, 0.05 * cap, cap * 0.95)
    return items


def _evaluate_params(params, items, cap, lb):
    """Evaluate a parameter set on a single item sequence."""
    used = _simulate_fast(items, cap, params)
    if used == 0:
        return np.inf
    return (used - lb) / lb


def _optimize_params():
    """Multi-stage parameter optimization using random search + local refinement."""
    n = 500
    cap = 100.0
    
    # Generate multiple test sequences for robustness
    test_seeds = [1234, 5678, 9012, 3456, 7890]
    test_items = []
    test_lbs = []
    for seed in test_seeds:
        items = _generate_weibull(n, cap, seed)
        test_items.append(items)
        test_lbs.append(_l1_lb(items, cap))
    
    # Use first sequence for coarse search (faster), then validate on all
    primary_items = test_items[0]
    primary_lb = test_lbs[0]
    
    best_score = np.inf
    best_params = (1.0, 0.5, 0.3, 0.2)
    
    # Stage 1: Random search over a broad space
    # Parameters: (w_worst, w_best, w_next, w_frag)
    # Each in [0, 3], with sum normalized implicitly by relative weights
    
    rng = np.random.RandomState(42)
    n_random = 300
    
    for it in range(n_random):
        if time.time() > _DEADLINE - 40.0:
            break
        # Sample from a distribution biased toward interesting regions
        w_worst = rng.uniform(0.0, 3.0)
        w_best = rng.uniform(0.0, 2.0)
        w_next = rng.uniform(0.0, 1.5)
        w_frag = rng.uniform(0.0, 1.0)
        
        params = (w_worst, w_best, w_next, w_frag)
        score = _evaluate_params(params, primary_items, cap, primary_lb)
        
        if score < best_score:
            best_score = score
            best_params = params
    
    # Stage 2: Local refinement around best point
    if time.time() < _DEADLINE - 25.0:
        w0_worst, w0_best, w0_next, w0_frag = best_params
        
        # Local grid around best
        steps_worst = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        steps_best = [0.5, 0.75, 1.0, 1.25, 1.5]
        steps_next = [0.5, 0.75, 1.0, 1.25, 1.5]
        steps_frag = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for sw in steps_worst:
            if time.time() > _DEADLINE - 20.0:
                break
            for sb in steps_best:
                if time.time() > _DEADLINE - 20.0:
                    break
                for sn in steps_next:
                    if time.time() > _DEADLINE - 20.0:
                        break
                    for sf in steps_frag:
                        if time.time() > _DEADLINE - 20.0:
                            break
                        w_worst = w0_worst * sw
                        w_best = w0_best * sb
                        w_next = w0_next * sn
                        w_frag = w0_frag * sf
                        
                        if w_worst < 0.01:
                            w_worst = 0.01
                        if w_best < 0.01:
                            w_best = 0.01
                        
                        params = (w_worst, w_best, w_next, w_frag)
                        score = _evaluate_params(params, primary_items, cap, primary_lb)
                        
                        if score < best_score:
                            best_score = score
                            best_params = params
    
    # Stage 3: Validate on all sequences and fine-tune
    if time.time() < _DEADLINE - 15.0:
        w0_worst, w0_best, w0_next, w0_frag = best_params
        
        # Evaluate on all sequences to get a more robust score
        def _full_score(params):
            total = 0.0
            for items, lb in zip(test_items, test_lbs):
                total += _evaluate_params(params, items, cap, lb)
            return total / len(test_items)
        
        best_full_score = _full_score(best_params)
        
        # Fine refinement on full score
        fine_steps = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        for sw in fine_steps:
            if time.time() > _DEADLINE - 10.0:
                break
            for sb in fine_steps:
                if time.time() > _DEADLINE - 10.0:
                    break
                for sn in fine_steps:
                    if time.time() > _DEADLINE - 10.0:
                        break
                    for sf in fine_steps:
                        if time.time() > _DEADLINE - 10.0:
                            break
                        w_worst = max(w0_worst * sw, 0.01)
                        w_best = max(w0_best * sb, 0.01)
                        w_next = max(w0_next * sn, 0.0)
                        w_frag = max(w0_frag * sf, 0.0)
                        
                        params = (w_worst, w_best, w_next, w_frag)
                        score = _full_score(params)
                        
                        if score < best_full_score:
                            best_full_score = score
                            best_params = params
    
    return best_params


# ---- Startup: fit parameters once ----
_PARAMS = _optimize_params()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return one score per bin with enough remaining capacity (higher = chosen)."""
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    rem = bins - item
    scores = _sim_score(item, bins, _PARAMS)
    scores = np.where(rem >= -1e-9, scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
