"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 3.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)


def _lp_relaxation_value(capacity, weights, values, order_indices):
    """
    Compute the fractional knapsack (LP relaxation) value for the given items
    in the given order. Returns (total_value, fractional_item_index_or_-1,
    fraction_of_that_item).
    """
    sim_cap = capacity
    sim_val = 0.0
    frac_idx = -1
    frac_amt = 0.0
    for idx in order_indices:
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        else:
            frac_idx = idx
            frac_amt = sim_cap / w if w > 0 else 0.0
            sim_val += values[idx] * frac_amt
            break
    return sim_val, frac_idx, frac_amt


def _simulate_with_lp_fill(capacity, weights, values, order_indices, exclude_idx):
    """
    Simulate: pick exclude_idx first, then fill remaining capacity using
    fractional knapsack (LP relaxation) on the remaining items in order.
    Returns total LP value (not integer, but a good upper bound surrogate).
    """
    sim_cap = capacity
    sim_val = 0.0
    
    w = weights[exclude_idx]
    if w <= sim_cap:
        sim_cap -= w
        sim_val += values[exclude_idx]
    else:
        # Infeasible, shouldn't happen if we filtered, but guard
        return sim_val
    
    # Now fill with LP relaxation over remaining items in order
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        wi = weights[idx]
        if wi <= sim_cap:
            sim_cap -= wi
            sim_val += values[idx]
        else:
            if wi > 0:
                sim_val += values[idx] * (sim_cap / wi)
            break
    return sim_val


def _greedy_fill_integer(capacity, weights, values, order_indices, exclude_idx):
    """
    Pick exclude_idx first, then greedily fill with integer items in order.
    Returns total value.
    """
    sim_cap = capacity
    sim_val = 0.0
    w = weights[exclude_idx]
    if w <= sim_cap:
        sim_cap -= w
        sim_val += values[exclude_idx]
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        wi = weights[idx]
        if wi <= sim_cap:
            sim_cap -= wi
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val


def _local_search_polish(capacity, weights, values, picked_set, m, max_iters=200):
    """
    Try to improve the picked set by swapping one item out for one item in.
    Returns improved picked_set and total value.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in picked_set:
        sim_cap -= weights[idx]
        sim_val += values[idx]
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _elapsed_ok():
        improved = False
        iters += 1
        # Try removing each picked item and adding the best available
        for out_idx in list(picked_set):
            w_out = weights[out_idx]
            v_out = values[out_idx]
            temp_cap = sim_cap + w_out
            best_in = -1
            best_gain = 0.0
            for in_idx in range(m):
                if in_idx in picked_set:
                    continue
                w_in = weights[in_idx]
                if w_in <= temp_cap:
                    v_in = values[in_idx]
                    gain = v_in - v_out
                    if gain > best_gain:
                        best_gain = gain
                        best_in = in_idx
            if best_in >= 0:
                picked_set.discard(out_idx)
                picked_set.add(best_in)
                sim_cap = temp_cap - weights[best_in]
                sim_val += best_gain
                improved = True
                break
    return picked_set, sim_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    if not _elapsed_ok():
        # Fallback: highest density feasible item
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        idxs = np.where(feasible)[0]
        densities = remaining_values[idxs] / remaining_weights[idxs]
        return int(idxs[np.argmax(densities)])

    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_idx].astype(np.float64)
    f_v = remaining_values[feasible_idx].astype(np.float64)
    m = len(f_w)
    
    if m == 1:
        return int(feasible_idx[0])

    # Compute densities
    densities = f_v / f_w
    order = np.argsort(-densities)  # local indices, best density first
    full_order = order.tolist()

    # Strategy: Use LP relaxation as surrogate.
    # The LP relaxation value is an upper bound. We want to pick the item
    # that, when committed, leaves the best LP relaxation value for the rest.
    # This is equivalent to minimizing the "gap" we create by committing.
    #
    # Actually, a better heuristic: pick the item that maximizes the
    # LP relaxation value of the remaining problem after committing to it.
    # This guides us toward the LP-optimal solution.
    
    # Limit candidates to top K by density to keep cost manageable
    K = min(40, m)
    candidates = full_order[:K]

    # For each candidate, compute LP relaxation value of remaining
    best_lp_val = -1.0
    best_local = candidates[0]
    candidate_scores = []
    
    for cand in candidates:
        # LP relaxation value if we pick cand first
        lp_val = _simulate_with_lp_fill(remaining_capacity, f_w, f_v, full_order, cand)
        candidate_scores.append((lp_val, cand))
        if lp_val > best_lp_val:
            best_lp_val = lp_val
            best_local = cand
        if not _elapsed_ok():
            break

    # Sort by LP surrogate score, take top L for more detailed evaluation
    candidate_scores.sort(key=lambda x: -x[0])
    L = min(10, len(candidate_scores))
    top_candidates = [c for _, c in candidate_scores[:L]]

    # Phase 2: For top candidates, do integer greedy fill + local search
    best_int_val = -1.0
    best_int_local = best_local
    best_picked = None
    
    for cand in top_candidates:
        # Integer greedy fill
        int_val = _greedy_fill_integer(remaining_capacity, f_w, f_v, full_order, cand)
        
        # Build picked set for local search
        picked = set()
        sim_cap = remaining_capacity
        if f_w[cand] <= sim_cap:
            sim_cap -= f_w[cand]
            picked.add(cand)
        
        for idx in full_order:
            if idx == cand:
                continue
            if f_w[idx] <= sim_cap:
                sim_cap -= f_w[idx]
                picked.add(idx)
            if sim_cap <= 0:
                break
        
        # Local search polish
        if _elapsed_ok():
            picked, int_val = _local_search_polish(
                remaining_capacity, f_w, f_v, picked, m, max_iters=50
            )
        
        if int_val > best_int_val:
            best_int_val = int_val
            best_int_local = cand
            best_picked = picked
        if not _elapsed_ok():
            break

    if best_int_local < 0:
        best_int_local = best_local

    return int(feasible_idx[best_int_local])
# EVOLVE-BLOCK-END
