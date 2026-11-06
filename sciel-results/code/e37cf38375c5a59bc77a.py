import numpy as np
import time

def _greedy_fill(cap, w_arr, v_arr, order, excluded):
    """
    Simulate greedy packing: walk 'order' (indices sorted by density desc),
    skip excluded items and items that don't fit, accumulate value.
    """
    total = 0.0
    cur = cap
    for idx in order:
        if idx in excluded:
            continue
        wi = w_arr[idx]
        if wi > cur:
            continue
        total += v_arr[idx]
        cur -= wi
        if cur <= 1e-12:
            break
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    start_time = time.time()
    deadline = start_time + 105.0  # safety margin under 120s

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)

    # Compute densities, handle zero weight
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(w > 0, v / w, np.where(v > 0, np.inf, 0.0))

    # Sort by density descending
    order = np.argsort(-dens)

    # Pre-filter: only consider items that fit in current capacity
    feasible = np.where(w <= cap)[0]
    if len(feasible) == 0:
        return -1

    # We'll evaluate top-K candidates for the first pick
    # K is adaptive: start with 50, expand if time allows
    K = min(len(feasible), 50)
    top_candidates = feasible[:K]  # already sorted by density via order? No, feasible is in original index order.
    # Actually, 'order' gives all indices sorted by density. Let's take the first K from 'order' that are feasible.
    top_candidates = []
    for idx in order:
        if w[idx] <= cap:
            top_candidates.append(idx)
            if len(top_candidates) >= K:
                break

    best_score = -1.0
    best_idx = -1

    for i in top_candidates:
        wi = w[i]
        vi = v[i]
        rem_cap = cap - wi

        # Option A: pick i, then greedy fill the rest
        excl = {i}
        future_val = _greedy_fill(rem_cap, w, v, order, excl)
        score_a = vi + future_val

        # Option B: 2-step lookahead. For each feasible j (j != i, wi+wj <= cap),
        # pick i then j, then greedy fill the rest.
        # To keep this fast, we only consider top-M j's by density that fit with i.
        M = min(n, 30)
        best_score_b = -1.0
        for j in order:
            if j == i:
                continue
            wj = w[j]
            if wi + wj > cap:
                continue
            # j is feasible with i
            excl2 = {i, j}
            future_val2 = _greedy_fill(cap - wi - wj, w, v, order, excl2)
            score_b = vi + v[j] + future_val2
            if score_b > best_score_b:
                best_score_b = score_b
            M -= 1
            if M <= 0:
                break

        # The best score for choosing i is the max of option A and option B
        # Actually, option A is a special case where the "next item" after i is determined by greedy.
        # Option B explicitly considers the second item. The true best for starting with i is
        # max over all possible second items j (including the greedy choice) of vi + vj + greedy_rest.
        # Since greedy_rest depends on which j we pick, we can't just take max(score_a, best_score_b)
        # because score_a assumes the greedy choice is the best second item, but maybe a non-greedy
        # second item leads to a better overall fill.
        # However, score_a IS the case where the second item is the greedy choice.
        # So the best score for starting with i is max(score_a, best_score_b) is WRONG.
        # It should be: for each candidate second item j (including the greedy one), compute the total.
        # score_a is the total when j = greedy_choice.
        # best_score_b is the max over top-M j's.
        # If the greedy choice is among the top-M, then max(score_a, best_score_b) is correct.
        # If not, we might miss it. But greedy choice is high density, so likely in top-M.
        # To be safe, let's just use the max of all evaluated options.
        # Actually, a cleaner approach: the best total value starting with i is the maximum
        # over all feasible j (j != i, wi+wj <= cap) of vi + vj + greedy_fill(cap-wi-wj, excl={i,j}).
        # This includes the case where no more items fit (greedy_fill returns 0).
        # But we also need to consider the case where we pick i and NO second item.
        # That's covered if we allow "j = none" which gives vi + greedy_fill(cap-wi, excl={i}).
        # But greedy_fill(cap-wi, excl={i}) will pick items, so it's not "no second item".
        # The 0/1 knapsack decision is: pick i, then solve the subproblem.
        # The subproblem is: maximize value with capacity cap-wi, items = all except i.
        # Our heuristic for the subproblem is greedy. So score_a IS our estimate of the subproblem value.
        # The 2-step lookahead is trying to improve the greedy estimate for the subproblem by
        # explicitly considering the second item.
        # So the best estimate for starting with i is max(score_a, best_score_b) is still not quite right.
        # score_a = vi + greedy_estimate(subproblem)
        # best_score_b = max over j of vi + vj + greedy_estimate(subproblem_after_j)
        # The true value is vi + OPT(subproblem).
        # greedy_estimate(subproblem) is an underestimate of OPT(subproblem).
        # best_score_b is a better estimate if the greedy choice for the subproblem is not optimal.
        # So we should take max(score_a, best_score_b).
        # But wait, score_a already includes the greedy choice for the subproblem.
        # If the greedy choice is the best j, then score_a = best_score_b (approximately).
        # If a non-greedy j is better, best_score_b > score_a.
        # So max(score_a, best_score_b) is a reasonable estimate.

        score_i = max(score_a, best_score_b)
        if best_score_b < 0:
            # No j found, so only score_a is valid
            score_i = score_a

        if score_i > best_score:
            best_score = score_i
            best_idx = i

        # Time check
        if time.time() > deadline:
            break

    # If no valid candidate found (shouldn't happen if feasible is non-empty)
    if best_idx == -1:
        # Fall back to greedy: pick the highest density feasible item
        for idx in order:
            if w[idx] <= cap:
                best_idx = int(idx)
                break
        if best_idx == -1:
            return -1

    return int(best_idx)
