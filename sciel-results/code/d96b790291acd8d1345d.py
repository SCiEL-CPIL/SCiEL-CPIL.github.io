# EVOLVE-BLOCK-START
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_customers = len(demands)
    n_facilities = len(capacities)

    # Start with a strong greedy assignment: sort by demand descending,
    # choose the facility minimizing assign_cost + marginal open cost.
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    count = np.zeros(n_facilities, dtype=int)

    for i in order:
        best_f = -1
        best_score = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                continue
            if count[f] == 0:
                score = assign_cost[i, f] + open_cost[f]
            else:
                score = assign_cost[i, f] + open_cost[f] / (count[f] + 1)
            if score < best_score:
                best_score = score
                best_f = f
        if best_f == -1:
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                best_f = int(np.argmax(capacities - load))
        result[i] = best_f
        load[best_f] += demands[i]
        count[best_f] += 1

    # Compute initial total cost
    open_mask = (count > 0)
    total_cost = float(np.sum(open_cost[open_mask])) + float(np.sum(assign_cost[np.arange(n_customers), result]))

    # Local search: single moves with 2-opt (customer swaps)
    max_iterations = 80
    for iteration in range(max_iterations):
        improved = False

        # --- Single moves ---
        for i in range(n_customers):
            old_f = result[i]
            best_new_f = -1
            best_new_cost = total_cost

            for f in range(n_facilities):
                if f == old_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue

                assign_change = assign_cost[i, f] - assign_cost[i, old_f]
                open_change = 0.0
                if count[old_f] == 1:
                    open_change -= open_cost[old_f]
                if count[f] == 0:
                    open_change += open_cost[f]

                new_total = total_cost + assign_change + open_change
                if new_total < best_new_cost - 1e-9:
                    best_new_cost = new_total
                    best_new_f = f

            if best_new_f != -1:
                result[i] = best_new_f
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                count[old_f] -= 1
                count[best_new_f] += 1
                total_cost = best_new_cost
                improved = True

        # --- Swap moves (2-opt): exchange two customers between facilities ---
        for i in range(n_customers):
            fi = result[i]
            for j in range(i + 1, n_customers):
                fj = result[j]
                if fi == fj:
                    continue
                # Check capacity feasibility after swap
                # After swap: i goes to fj, j goes to fi
                # fi loses demands[i], gains demands[j]
                # fj loses demands[j], gains demands[i]
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue

                # Compute cost change
                assign_change = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                                (assign_cost[j, fi] - assign_cost[j, fj])

                # Open cost change
                open_change = 0.0
                # fi: count changes from count[fi] to count[fi] (same count, since one leaves one enters)
                # Actually count stays the same for both facilities in a swap
                # So no open cost change unless a facility becomes empty or was empty
                # But in a swap, both facilities retain at least the same number of customers
                # Wait: if count[fi] == 1 and count[fj] == 1, after swap both still have 1 customer
                # So open cost doesn't change in a swap between two non-empty facilities
                # But what if one facility has only this customer? After swap it still has one (the other customer)
                # So open cost is unchanged for swaps between facilities with count >= 1
                # Actually this is correct: swaps don't change the set of open facilities

                new_total = total_cost + assign_change + open_change
                if new_total < total_cost - 1e-9:
                    # Perform the swap
                    result[i] = fj
                    result[j] = fi
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    # counts unchanged
                    total_cost = new_total
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return result
# EVOLVE-BLOCK-END
