"""Bounded lookahead 0/1 knapsack heuristic with multi-policy greedy completion."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _greedy_complete(w, v, cap, policy='density'):
    """Simulate greedy completion. Returns total added value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    elif policy == 'lagrangian':
        # Find lambda via bisection for Lagrangian relaxation
        # Maximize sum(v_i - lambda * w_i) for items with w_i <= cap
        feasible_mask = w <= cap
        if not np.any(feasible_mask):
            return 0.0
        fw = w[feasible_mask]
        fv = v[feasible_mask]
        
        # Bisection for lambda
        lo, hi = 0.0, max(fv) / max(min(fw), 1e-12) + 1.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            scores = fv - mid * fw
            # Pick items with positive score in order of score
            pos_mask = scores > 0
            if not np.any(pos_mask):
                hi = mid
                continue
            pos_idx = np.where(pos_mask)[0]
            pos_scores = scores[pos_idx]
            order = pos_idx[np.argsort(-pos_scores)]
            rem = float(cap)
            total_w = 0.0
            for idx in order:
                if fw[idx] <= rem:
                    rem -= fw[idx]
                    total_w += fw[idx]
            if total_w < cap * 0.95:
                lo = mid
            elif total_w > cap * 1.05:
                hi = mid
            else:
                break
        
        # Use the found lambda to order items
        lambda_val = (lo + hi) / 2.0
        scores = v - lambda_val * w
        order = np.argsort(-scores)
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))

    rem = float(cap)
    total = 0.0
    for i in order:
        wi = float(w[i])
        if wi <= rem:
            rem -= wi
            total += float(v[i])
            if rem <= 0:
                break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add using lookahead greedy completion.
    For each candidate item, simulate adding it plus a density-greedy fill
    of the remaining capacity, and choose the item that maximizes total
    estimated value.
    """
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Only consider feasible items (weight <= capacity)
    feasible = w <= cap
    if not np.any(feasible):
        return -1

    feasible_indices = np.where(feasible)[0]
    fw = w[feasible]
    fv = v[feasible]
    n_feasible = len(fw)

    # Check time budget
    if _remaining_budget() < 0.5:
        # Fallback to simple density
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    # Evaluate each candidate item with lookahead
    # For small n, evaluate all; for larger n, sample top candidates by multiple criteria
    if n_feasible <= 40:
        candidates = feasible_indices
    else:
        # Take top candidates by density, value, and Lagrangian score
        density = fv / np.maximum(fw, 1e-12)
        top_by_density = np.argsort(-density)[:min(25, n_feasible)]
        top_by_value = np.argsort(-fv)[:min(15, n_feasible)]
        
        # Combine unique candidates
        candidate_set = set(top_by_density.tolist())
        candidate_set.update(top_by_value.tolist())
        top_local = list(candidate_set)
        top_local.sort(key=lambda x: -density[x])
        top_local = top_local[:40]
        candidates = feasible_indices[top_local]

    best_score = -1.0
    best_candidate = None
    best_policy = 'density'

    # Check time budget more frequently
    time_check_interval = max(1, len(candidates) // 5)
    
    for idx, ci in enumerate(candidates):
        # Periodic time check
        if idx > 0 and idx % time_check_interval == 0 and _remaining_budget() < 1.0:
            break
            
        wi = float(w[ci])
        vi = float(v[ci])
        new_cap = cap - wi
        if new_cap < 0:
            continue

        # Create remaining arrays after taking this item
        mask = np.ones(n, dtype=bool)
        mask[ci] = False
        rem_w = w[mask]
        rem_v = v[mask]

        # Try multiple policies for completion
        for policy in ('density', 'lagrangian', 'blend'):
            completion_value = _greedy_complete(rem_w, rem_v, new_cap, policy)
            total_value = vi + completion_value
            if total_value > best_score:
                best_score = total_value
                best_candidate = ci
                best_policy = policy

    if best_candidate is None:
        # Fallback to highest density feasible item
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    return int(best_candidate)
# EVOLVE-BLOCK-END
