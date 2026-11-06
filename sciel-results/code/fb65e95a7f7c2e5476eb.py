"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies + 1-swap + 2-opt refinement, returns best total value."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    m = len(w)

    if m == 0:
        return 0.0

    best_total = 0.0
    best_selected = None  # track indices (into w/v) for swap refinement

    def _run_policy(order):
        total = 0.0
        cap = capacity
        selected = []
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                selected.append(i)
                if cap <= 0:
                    break
        return total, selected

    # Policy 0: density
    order = np.argsort(-(v / w))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 1: value
    order = np.argsort(-v)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 2: blend (density + 0.5 * value)
    order = np.argsort(-(v / w + 0.5 * v))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 3: small-weight first
    order = np.argsort(w)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 4: snug-fit density (prefer items close to remaining capacity fraction)
    scores4 = (v / w) * (1 + 0.1 * (w / max(capacity, 1e-12)))
    order = np.argsort(-scores4)
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 5: value / sqrt(weight) — favors lighter high-value items
    order = np.argsort(-(v / np.sqrt(w)))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 6: value / weight^1.5 — strong light preference
    order = np.argsort(-(v / (w * np.sqrt(w))))
    t, sel = _run_policy(order)
    if t > best_total:
        best_total = t
        best_selected = sel

    # Policy 7: light-first then density
    light_mask = w <= 0.3 * capacity
    if np.any(light_mask):
        light_order = np.where(light_mask)[0]
        light_order = light_order[np.argsort(-(v[light_order] / w[light_order]))]
        heavy_order = np.where(~light_mask)[0]
        heavy_order = heavy_order[np.argsort(-(v[heavy_order] / w[heavy_order]))]
        combined = np.concatenate([light_order, heavy_order])
        t, sel = _run_policy(combined)
        if t > best_total:
            best_total = t
            best_selected = sel

    # 1-swap refinement on the best solution found
    if best_selected is not None and len(best_selected) > 0 and m > len(best_selected):
        sel_arr = np.array(best_selected)
        unselected = np.setdiff1d(np.arange(m), sel_arr)
        if len(unselected) > 0:
            sel_w = w[sel_arr]
            sel_v = v[sel_arr]
            un_w = w[unselected]
            un_v = v[unselected]
            for si in range(len(sel_arr)):
                freed_w = sel_w[si]
                freed_v = sel_v[si]
                cand_mask = (un_w <= freed_w + 1e-12) & (un_v > freed_v)
                if np.any(cand_mask):
                    cand_vals = un_v[cand_mask]
                    best_cand_local = int(np.argmax(cand_vals))
                    best_cand_val = cand_vals[best_cand_local]
                    new_val = best_total - freed_v + best_cand_val
                    if new_val > best_total:
                        best_total = new_val

    # 2-opt refinement: try removing 2 selected items and adding 1 unselected that fits
    if best_selected is not None and len(best_selected) >= 2 and m > len(best_selected):
        sel_arr = np.array(best_selected)
        unselected = np.setdiff1d(np.arange(m), sel_arr)
        if len(unselected) > 0:
            sel_w = w[sel_arr]
            sel_v = v[sel_arr]
            un_w = w[unselected]
            un_v = v[unselected]
            # For each pair of selected items, freed weight = sel_w[i] + sel_w[j]
            # freed value = sel_v[i] + sel_v[j]
            # Find best unselected with weight <= freed weight and value > freed value
            # Limit to top pairs by freed weight to keep it fast
            k = len(sel_arr)
            if k <= 30:
                for i in range(k):
                    for j in range(i + 1, k):
                        freed_w = sel_w[i] + sel_w[j]
                        freed_v = sel_v[i] + sel_v[j]
                        cand_mask = (un_w <= freed_w + 1e-12) & (un_v > freed_v)
                        if np.any(cand_mask):
                            cand_vals = un_v[cand_mask]
                            best_cand_val = cand_vals.max()
                            new_val = best_total - freed_v + best_cand_val
                            if new_val > best_total:
                                best_total = new_val

    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    fn = len(fw)

    if fn == 1:
        return int(feasible_indices[0])

    # Fast fallback if time is tight
    if elapsed > _TOTAL_BUDGET * 0.7:
        density = fv / fw
        d_min, d_max = density.min(), density.max()
        v_min, v_max = fv.min(), fv.max()
        d_range = d_max - d_min + 1e-12
        v_range = v_max - v_min + 1e-12
        d_norm = (density - d_min) / d_range
        v_norm = (fv - v_min) / v_range
        scores = 0.6 * d_norm + 0.4 * v_norm
        return int(feasible_indices[int(np.argmax(scores))])

    # Medium fallback: simple lookahead with fewer items if time is moderate
    if elapsed > _TOTAL_BUDGET * 0.5:
        best_score = -1.0
        best_local_idx = 0
        for li in range(fn):
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                    feasible_indices[li])
            total_est = item_v + est
            if total_est > best_score:
                best_score = total_est
                best_local_idx = li
        return int(feasible_indices[best_local_idx])

    # Full lookahead: for each feasible item, estimate total via greedy completion
    best_score = -1.0
    best_local_idx = 0
    scores_arr = np.zeros(fn)

    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        scores_arr[li] = total_est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

    # Pair-lookahead: for the top-3 candidates, also try picking a second item
    # that fills the remaining gap well, to break ties or find better combos
    if fn >= 2 and elapsed < _TOTAL_BUDGET * 0.85:
        top3 = np.argsort(-scores_arr)[:min(3, fn)]
        for li in top3:
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            if remaining_cap <= 0:
                continue
            # Find best second item that fits in remaining_cap
            second_mask = np.ones(n, dtype=bool)
            second_mask[feasible_indices[li]] = False
            second_mask &= remaining_weights <= remaining_cap
            if not np.any(second_mask):
                continue
            second_idxs = np.where(second_mask)[0]
            sw = remaining_weights[second_idxs]
            sv = remaining_values[second_idxs]
            if len(sw) > 0:
                # Pick the second item by density
                best_second_local = int(np.argmax(sv / sw))
                best_second = second_idxs[best_second_local]
                pair_remaining_cap = remaining_cap - sw[best_second_local]
                est2 = _greedy_fill_fast(pair_remaining_cap, remaining_weights,
                                         remaining_values, best_second)
                total_pair = item_v + sv[best_second_local] + est2
                if total_pair > best_score:
                    best_score = total_pair
                    best_local_idx = li

    # DP polish: if remaining capacity is small enough for a quick DP, use it
    # to get an exact optimum for the remaining subproblem, which can beat greedy.
    # We apply this only when the number of feasible items is small and capacity
    # is bounded, to keep runtime under control.
    if elapsed < _TOTAL_BUDGET * 0.9:
        # Check if we can afford a DP on the remaining feasible items
        # Capacity is in continuous units; DP works best with small integer-like capacity.
        # Since weights/values are floats, we skip DP unless capacity is very small.
        if remaining_capacity <= 5.0 and fn <= 60:
            # Scale weights to integers for DP (only if they are "nice")
            # Try to use a simple DP with capacity discretized
            # This is heuristic: only attempt if weights are multiples of some step
            min_w = fw.min()
            if min_w > 1e-6:
                # Estimate number of distinct weight buckets
                # Use a simple DP with capacity discretized to ~200 buckets
                num_buckets = min(200, int(remaining_capacity / min_w) + 1)
                if num_buckets <= 200 and num_buckets > 0:
                    dp = np.zeros(num_buckets + 1, dtype=np.float64)
                    for i in range(fn):
                        wi = fw[i]
                        vi = fv[i]
                        wi_bucket = min(int(wi / min_w), num_buckets)
                        if wi_bucket > 0:
                            for b in range(num_buckets, wi_bucket - 1, -1):
                                new_val = dp[b - wi_bucket] + vi
                                if new_val > dp[b]:
                                    dp[b] = new_val
                    dp_best = dp.max()
                    # Compare with current best_score: dp_best is an estimate
                    # for the remaining items (not including the first pick),
                    # so we need to add the value of the first item.
                    # Actually scores_arr[li] already includes item value + greedy est.
                    # dp_best is a better estimate of the "est" part.
                    # So for the best candidate, total = item_v + dp_best
                    # But dp_best is for ALL remaining items, not just after picking one.
                    # This is tricky: dp gives optimum for all feasible items with capacity.
                    # That's actually what we want as a bound.
                    # If dp_best > scores_arr[best_local_idx] - fv[best_local_idx] + fv[best_local_idx],
                    # i.e., dp_best > scores_arr[best_local_idx], we can't directly use it
                    # because we must pick exactly one item first.
                    # Skip this DP polish as it's not directly applicable to the
                    # "pick one then fill rest" structure. Leave as is.
                    pass

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
