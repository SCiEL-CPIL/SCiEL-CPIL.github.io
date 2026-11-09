"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 150s. We want to stay well under.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin


def _lp_order_and_pivot(capacity, weights, values):
    """
    Compute the LP-relaxation optimal order for the fractional knapsack.
    Returns:
        order: array of indices sorted by density descending
        pivot_idx: index in the full array of the break item (fractionally filled), or -1 if none
        lp_value: the LP optimal value
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return np.array([], dtype=int), -1, 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)

    order = np.argsort(-densities, kind='stable')

    lp_value = 0.0
    remaining_cap = capacity
    pivot_idx = -1

    for pos in range(len(order)):
        idx = order[pos]
        w = weights[idx]
        v = values[idx]
        if w <= remaining_cap:
            lp_value += v
            remaining_cap -= w
        else:
            # This is the break item
            if w > 0:
                lp_value += v * (remaining_cap / w)
            pivot_idx = idx
            remaining_cap = 0.0
            break

    return order, pivot_idx, lp_value


def _greedy_fill(capacity, weights, values, exclude_idx=None):
    """
    Greedily fill capacity with items sorted by density (value/weight).
    Returns the total value achieved.
    exclude_idx: if provided, skip this index (in the local array).
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)

    order = np.argsort(-densities, kind='stable')

    total_value = 0.0
    remaining_cap = capacity
    for idx in order:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        if weights[idx] <= remaining_cap:
            total_value += values[idx]
            remaining_cap -= weights[idx]
        if remaining_cap <= 0:
            break

    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item index (into the remaining arrays) to add, or -1 to stop.
    
    Strategy: LP-relaxation-guided heuristic.
    1. Compute the fractional knapsack optimal order and identify the pivot (break) item.
    2. Score each feasible item by how well it aligns with the LP-optimal structure:
       - Items before the pivot in LP order get high priority (they should be taken fully).
       - The pivot item gets medium priority (it's the fractional boundary).
       - Items after the pivot get lower priority.
    3. Use a 1-step lookahead to break ties and refine the choice.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]

    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Compute LP relaxation order and pivot
    order, pivot_idx, lp_value = _lp_order_and_pivot(
        remaining_capacity, weights, values
    )

    # Build a rank map: for each global index, its position in the LP order
    rank_map = np.full(n, -1, dtype=np.int64)
    for pos, idx in enumerate(order):
        rank_map[idx] = pos

    # Score each feasible item based on LP alignment
    # Items with rank < pivot_rank are "should be taken fully"
    # The pivot item is the boundary
    # Items with rank > pivot_rank are "should not be taken"
    
    if pivot_idx >= 0:
        pivot_rank = rank_map[pivot_idx]
    else:
        pivot_rank = len(order)  # no pivot, all items fit

    best_score = -1.0
    best_idx = -1

    for local_i in range(len(feasible_indices)):
        # Skip if deadline approaching
        if time.time() > _DEADLINE - 0.5:
            break

        global_i = feasible_indices[local_i]
        w_i = feasible_weights[local_i]
        v_i = feasible_values[local_i]
        cap_after = remaining_capacity - w_i

        if cap_after < 0:
            continue

        rank_i = rank_map[global_i]

        # LP alignment score:
        # - If rank_i < pivot_rank: this item should be fully taken in LP solution. High score.
        # - If rank_i == pivot_rank: this is the pivot item. Medium score.
        # - If rank_i > pivot_rank: this item should not be taken in LP solution. Low score.
        if rank_i < 0:
            lp_alignment = 0.0
        elif rank_i < pivot_rank:
            # Higher density items before pivot are more important
            # Normalize by position: closer to pivot = slightly less critical
            lp_alignment = 1.0 + (pivot_rank - rank_i) * 0.1
        elif rank_i == pivot_rank:
            lp_alignment = 0.5
        else:
            # Beyond pivot: penalize based on distance
            lp_alignment = max(0.0, 0.3 - (rank_i - pivot_rank) * 0.05)

        # 1-step lookahead: greedy fill after taking this item
        future_value = _greedy_fill(
            cap_after, feasible_weights, feasible_values, exclude_idx=local_i
        )

        # Combined score: LP alignment weighted + future value
        # Scale future_value to be comparable with lp_alignment
        # Use a blend: prioritize LP structure but use lookahead for tie-breaking
        total_score = lp_alignment * 1000.0 + future_value

        if total_score > best_score:
            best_score = total_score
            best_idx = int(global_i)

    if best_idx == -1:
        # Fallback: pick highest density feasible item
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(feasible_weights > 0, feasible_values / feasible_weights, 0.0)
        best_density = -1
        best_idx = -1
        for local_i in range(len(feasible_indices)):
            if densities[local_i] > best_density:
                best_density = densities[local_i]
                best_idx = int(feasible_indices[local_i])

    return best_idx
# EVOLVE-BLOCK-END
