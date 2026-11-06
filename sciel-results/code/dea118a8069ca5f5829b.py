"""Bounded lookahead 0/1 knapsack heuristic with dense candidate scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard: total wall-clock budget for the whole evaluation.
_START = time.monotonic()
_TOTAL_BUDGET = 115.0  # leave margin before the 120s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _greedy_complete(weights, values, capacity, policy='density'):
    """Simulate greedy completion from a state. Returns total added value."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    safe_w = np.maximum(w, 1e-12)
    if policy == 'density':
        order = np.argsort(-v / safe_w)
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / safe_w + 0.15 * v))
    else:
        order = np.argsort(-v / safe_w)
    
    cap = float(capacity)
    # Vectorized greedy: iterate in sorted order, pick items that fit
    # Use a loop but with early exit; for n<=100 this is fast
    total = 0.0
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 1e-9:
                break
    return float(total)

def _score_candidates(w, v, cap, n):
    """Generate diverse candidate indices using multiple scoring families."""
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    
    # Multiple scoring families with tuned coefficients
    scores = np.vstack([
        v,                                      # pure value
        density,                                # pure density
        density + 0.12 * v,                     # blend
        v - 0.015 * w,                          # value minus weight penalty
        density * (1.0 + 0.25 * (w / cap)),     # density boosted by relative size
        v / (w + 0.08 * cap),                   # smoothed density
        v * (1.0 - w / (cap + 1e-9)),           # value discounted by weight fraction
    ])
    
    cands = set()
    k = min(n, 12)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            cands.add(int(t))
    
    # Ensure density-argmax and value-argmax are included
    cands.add(int(np.argmax(density)))
    cands.add(int(np.argmax(v)))
    
    return list(cands), density

def _evaluate_candidate(ci, w, v, cap, n, density):
    """Estimate total value if we pick candidate ci now, using multi-policy greedy completion.
    Uses soft-max over policy estimates for a smoother objective."""
    used_w = float(w[ci])
    used_v = float(v[ci])
    new_cap = cap - used_w
    if new_cap < 0:
        return -np.inf
    
    # Build mask excluding candidate
    mask = np.ones(n, dtype=bool)
    mask[ci] = False
    rw = w[mask]
    rv = v[mask]
    
    # Try multiple completion policies
    estimates = []
    for pol in ('density', 'value', 'blend'):
        estimates.append(used_v + _greedy_complete(rw, rv, new_cap, pol))
    
    # Soft-max surrogate: smooth approximation of max
    # temperature controls smoothness; lower = closer to hard max
    temp = 0.1
    est_arr = np.array(estimates, dtype=np.float64)
    # log-sum-exp for numerical stability
    m = est_arr.max()
    soft_max = m + temp * np.log(np.sum(np.exp((est_arr - m) / temp)))
    return float(soft_max)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Only consider feasible items
    feasible = w <= cap
    if not np.any(feasible):
        return -1
    
    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible[idx] else -1
    
    # Generate diverse candidates
    cands, density = _score_candidates(w, v, cap, n)
    
    # Filter to feasible candidates only
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1
    
    # Limit candidates to keep per-call cost bounded
    sim_limit = 60
    if len(cands) > sim_limit:
        cands.sort(key=lambda i: -density[i])
        cands = cands[:sim_limit]
    
    # Evaluate each candidate with bounded lookahead
    best_idx = cands[0]
    best_est = -np.inf
    
    for ci in cands:
        est = _evaluate_candidate(ci, w, v, cap, n, density)
        if est > best_est:
            best_est = est
            best_idx = ci
        
        # Deadline check
        if _remaining_budget() < 0.5:
            break
    
    # Second-level refinement: for top-5 candidates, do a 2-step lookahead
    # to break ties or improve selection when soft-max is close
    if len(cands) >= 3 and _remaining_budget() > 1.0:
        # Get top-5 by first-level estimate
        first_level_estimates = []
        for ci in cands[:10]:
            first_level_estimates.append((ci, _evaluate_candidate(ci, w, v, cap, n, density)))
        first_level_estimates.sort(key=lambda x: -x[1])
        top_k = first_level_estimates[:min(5, len(first_level_estimates))]
        
        for ci, fl_est in top_k:
            # 2-step: pick ci, then pick best second item, then greedy complete
            used_w = float(w[ci])
            used_v = float(v[ci])
            new_cap = cap - used_w
            if new_cap < 0:
                continue
            
            mask1 = np.ones(n, dtype=bool)
            mask1[ci] = False
            rw1 = w[mask1]
            rv1 = v[mask1]
            n1 = len(rw1)
            
            if n1 == 0 or new_cap <= 0:
                continue
            
            # Find best second item by density among feasible
            safe_w1 = np.maximum(rw1, 1e-12)
            dens1 = rv1 / safe_w1
            feasible1 = rw1 <= new_cap
            if not np.any(feasible1):
                continue
            
            # Try top-3 second items
            best_2step = -np.inf
            top_second = np.argsort(-dens1)[:min(3, n1)]
            for si in top_second:
                if not feasible1[si]:
                    continue
                sw = float(rw1[si])
                sv = float(rv1[si])
                cap2 = new_cap - sw
                if cap2 < 0:
                    continue
                mask2 = np.ones(n1, dtype=bool)
                mask2[si] = False
                rw2 = rw1[mask2]
                rv2 = rv1[mask2]
                total2 = used_v + sv + _greedy_complete(rw2, rv2, cap2, 'density')
                total2b = used_v + sv + _greedy_complete(rw2, rv2, cap2, 'value')
                total2c = used_v + sv + _greedy_complete(rw2, rv2, cap2, 'blend')
                best_2step = max(best_2step, total2, total2b, total2c)
            
            if best_2step > best_est:
                best_est = best_2step
                best_idx = ci
        
        if _remaining_budget() < 0.5:
            pass  # just proceed with current best
    
    return int(best_idx)
# EVOLVE-BLOCK-END
