"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 130.0  # safety margin under 150s limit


def _greedy_complete(capacity, weights, values, policy="density"):
    """
    Estimate total value achievable by greedy completion.
    policy: 'density', 'value', 'blend'
    Returns estimated total value (float).
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Only consider feasible items
    mask = weights <= capacity
    if not np.any(mask):
        return 0.0
    
    w = weights[mask]
    v = values[mask]
    cap = capacity
    total = 0.0
    
    # Work on copies
    w = w.copy()
    v = v.copy()
    
    while cap > 0:
        if policy == "density":
            scores = v / w
        elif policy == "value":
            scores = v
        elif policy == "blend":
            # Blend: value density weighted by value, plus small preference for lighter items
            scores = (v / w) * 0.5 + v * 0.3 + (1.0 / (w + 1e-9)) * 0.2
        else:
            scores = v / w
        
        idx = int(np.argmax(scores))
        if w[idx] > cap:
            # Find best item that fits
            fit_mask = w <= cap
            if not np.any(fit_mask):
                break
            fit_scores = scores.copy()
            fit_scores[~fit_mask] = -np.inf
            idx = int(np.argmax(fit_scores))
        
        total += v[idx]
        cap -= w[idx]
        w[idx] = -1  # mark as taken
        v[idx] = -1
    
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Time budget guard
    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1
    
    if remaining_capacity <= 0:
        return -1
    
    # Only consider items that actually fit
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_mask]
    fv = remaining_values[feasible_mask]
    nf = len(fw)
    
    if nf == 0:
        return -1
    
    # For each feasible item, compute lookahead score:
    # score_i = value_i + max_over_policies(greedy_complete(cap - w_i, remaining without i, policy))
    
    # To keep this fast, we'll sample a subset of items if nf is large,
    # but with capacity 12.5 and 100 items, nf is likely small.
    
    # If nf is very small, evaluate all. Otherwise, sample top candidates by density.
    if nf <= 50:
        candidates = feasible_indices
        cf = fw
        cv = fv
    else:
        # Take top 50 by value density as candidates
        densities = fv / cf
        top_k = 50
        top_local = np.argsort(densities)[-top_k:]
        candidates = feasible_indices[top_local]
        cf = fw[top_local]
        cv = fv[top_local]
    
    nc = len(candidates)
    lookahead_scores = np.zeros(nc)
    
    for i in range(nc):
        wi = cf[i]
        vi = cv[i]
        new_cap = remaining_capacity - wi
        
        if new_cap <= 0:
            lookahead_scores[i] = vi
            continue
        
        # Build remaining arrays without item i
        # We need to exclude candidate i from the remaining set
        # For simplicity, we work with the full feasible set minus item i
        
        # Create masks for the full feasible arrays
        # Remove item i from consideration
        if nc == nf:
            # All candidates are the full feasible set
            # Create boolean mask for all feasible items, then exclude i
            all_mask = np.ones(n, dtype=bool)
            all_mask[feasible_mask] = True
            # Exclude candidate i
            all_mask[candidates[i]] = False
            rw = remaining_weights[all_mask]
            rv = remaining_values[all_mask]
        else:
            # We only have a subset; for lookahead, use the full feasible set minus i
            all_mask = np.ones(n, dtype=bool)
            all_mask[feasible_mask] = True
            all_mask[candidates[i]] = False
            rw = remaining_weights[all_mask]
            rv = remaining_values[all_mask]
        
        # Estimate completion value using multiple policies
        est_density = _greedy_complete(new_cap, rw, rv, "density")
        est_value = _greedy_complete(new_cap, rw, rv, "value")
        est_blend = _greedy_complete(new_cap, rw, rv, "blend")
        
        best_est = max(est_density, est_value, est_blend)
        lookahead_scores[i] = vi + best_est
    
    best_local = int(np.argmax(lookahead_scores))
    return int(candidates[best_local])
# EVOLVE-BLOCK-END
