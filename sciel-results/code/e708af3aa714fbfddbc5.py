# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location via Lagrangian relaxation with subgradient
    optimization and capacity repair, followed by local search polish.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    deadline = time.time() + 135.0

    # ---------- Helper: build a capacity-feasible solution from a raw assignment ----------
    def repair(res):
        """
        Given a raw assignment (possibly infeasible), repair it to be capacity-feasible
        by moving customers from overloaded facilities to the cheapest feasible
        alternative. Uses a priority queue approach: move the customer with the
        largest marginal cost increase first? No — move the one with smallest
        increase to minimize damage.
        """
        res = res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]

        # Iteratively fix overloaded facilities
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            overloaded = np.where(load > capacities + 1e-9)[0]
            if len(overloaded) == 0:
                break

            f = overloaded[0]
            # Find customers at f, pick the one to move that causes least cost increase
            custs_at_f = np.where(res == f)[0]
            if len(custs_at_f) == 0:
                break

            best_i = -1
            best_g = -1
            best_inc = np.inf

            for i in custs_at_f:
                # Try all other facilities
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g] + 1e-9:
                        inc = assign_cost[i, g] - assign_cost[i, f]
                        if inc < best_inc:
                            best_inc = inc
                            best_i = i
                            best_g = g

            if best_i == -1:
                # No feasible move found for any customer at f — try to move to any
                # facility with most remaining capacity, ignoring cost
                remaining = capacities - load
                g = np.argmax(remaining)
                if remaining[g] <= 0:
                    break
                # Pick the customer at f with smallest demand to fit
                idx = np.argmin(demands[custs_at_f])
                best_i = custs_at_f[idx]
                best_g = g

            res[best_i] = best_g
            load[f] -= demands[best_i]
            load[best_g] += demands[best_i]

        return res

    # ---------- Helper: compute objective ----------
    def obj(res):
        used = np.unique(res)
        return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res]))

    # ---------- Lagrangian relaxation ----------
    # Dual variables for capacity constraints: lambda[f] >= 0
    # Lagrangian for customer i assigned to f:
    #   assign_cost[i,f] + lambda[f]*demands[i]
    # Plus open cost if f is the only customer assigned to it (handled in subproblem)
    #
    # Subproblem: for each customer, pick f minimizing (assign_cost[i,f] + lambda[f]*demands[i])
    # But we also need to account for open costs. We handle this by:
    #   1. Solve the relaxed problem ignoring open costs (pure assignment with modified costs)
    #   2. Compute the actual objective including open costs
    #   3. Use subgradient to update lambda
    #
    # Actually, a cleaner approach: include open costs in the subproblem by
    # noting that if facility f gets at least one customer, we pay open_cost[f].
    # This is a fixed-charge cost. For the Lagrangian, we can't easily decompose
    # this. Instead, we use the standard approach:
    #
    # The Lagrangian dual of the CFLP:
    #   min over x: sum_i sum_f c_if x_if + sum_f F_f y_f
    #   s.t. sum_f x_if = 1 for all i
    #        sum_i d_i x_if <= C_f y_f for all f
    #        x_if, y_f in {0,1}
    #
    # Relax the capacity constraints with multipliers lambda_f >= 0:
    #   L(lambda) = min over x (sum_i sum_f (c_if + lambda_f d_i) x_if) + sum_f F_f y_f
    #   But y_f is coupled to x. For the subproblem, if we fix which facilities are "open",
    #   it's easy. But the open decision is part of the optimization.
    #
    # Standard trick: the subproblem decomposes by customer if we ignore the open cost,
    # then we evaluate the true objective. The subgradient for lambda_f is:
    #   (sum_i d_i x_if) - C_f  (when y_f=1, i.e., facility is used)
    #   or  (sum_i d_i x_if) - C_f  in general (since if no customer is assigned, the
    #   constraint is trivially satisfied and the gradient contribution is 0 from the
    #   perspective of the relaxed problem, but for subgradient we use the actual
    #   load minus capacity).
    #
    # Let's implement:
    #   1. Initialize lambda = 0
    #   2. For each iteration:
    #      a. Solve subproblem: for each customer i, assign to f minimizing
    #         assign_cost[i,f] + lambda[f] * demands[i]
    #      b. Compute actual load and check feasibility
    #      c. Compute subgradient: g_f = load[f] - capacities[f] if load[f] > 0 else 0
    #         Actually, the subgradient of the Lagrangian w.r.t. lambda_f is:
    #         (sum_i demands[i] * x_if) - capacities[f]  (this is the violation)
    #         But we only care about positive violations for the ascent direction.
    #      d. Update: lambda_f = max(0, lambda_f + step * g_f)
    #      e. Track the best feasible solution found (after repair)
    #
    # Step size: use the standard diminishing step: step = alpha * (UB - L(lambda)) / ||g||^2
    # where UB is the best feasible objective found so far, L(lambda) is the Lagrangian value.
    # The Lagrangian value is: sum_i min_f (assign_cost[i,f] + lambda[f]*demands[i])
    # (this is a lower bound on the true objective, but doesn't include open costs properly)
    #
    # Actually, the Lagrangian lower bound is:
    #   L(lambda) = sum_i min_f (c_if + lambda_f d_i)
    # This is a valid lower bound on the assignment part, but the true objective also has
    # open costs. So the gap UB - L(lambda) may not be the right measure.
    #
    # Alternative: use a fixed diminishing step size, which is simpler and works well in practice.

    best_result = None
    best_score = np.inf

    # Initial greedy solution
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = np.argmax(capacities - load)
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]

    if np.all(load <= capacities):
        best_result = result_greedy.copy()
        best_score = obj(best_result)

    # Lagrangian relaxation
    lam = np.zeros(n_fac)
    n_iter_max = 3000
    # Use a fixed initial step and diminish it
    step_init = 1.0
    alpha = 2.0  # multiplier for step size

    # Precompute for speed
    # modified_cost[i,f] = assign_cost[i,f] + lam[f] * demands[i]
    # We update this incrementally

    for it in range(n_iter_max):
        if time.time() > deadline - 15.0:
            break

        # Solve subproblem: for each customer, argmin over f of (assign_cost[i,f] + lam[f]*demands[i])
        # Vectorized: modified_cost = assign_cost + np.outer(demands, lam)
        modified_cost = assign_cost + np.outer(demands, lam)
        raw_assign = np.argmin(modified_cost, axis=1)

        # Compute load
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[raw_assign[i]] += demands[i]

        # Check feasibility
        if np.all(load <= capacities + 1e-9):
            score = obj(raw_assign)
            if score < best_score:
                best_score = score
                best_result = raw_assign.copy()
        else:
            # Repair
            repaired = repair(raw_assign)
            if np.all(np.array([np.sum(demands[repaired == f]) for f in range(n_fac)]) <= capacities + 1e-9):
                score = obj(repaired)
                if score < best_score:
                    best_score = score
                    best_result = repaired.copy()

        # Compute Lagrangian lower bound
        min_costs = np.min(modified_cost, axis=1)
        L_val = np.sum(min_costs)

        # Subgradient: g_f = load[f] - capacities[f]
        g = load - capacities
        # For subgradient ascent on the dual (maximization of L(lambda)),
        # we want to increase lambda where there's a violation (g > 0)
        # and decrease where there's slack (g < 0), but lambda >= 0.
        # The dual function is concave, so we do subgradient ascent.
        # Step size: alpha * (UB - L_val) / (||g||^2 + eps)
        g_norm_sq = np.sum(g ** 2) + 1e-12
        if best_score < np.inf:
            step = alpha * (best_score - L_val) / g_norm_sq
        else:
            step = step_init
        step = max(step, 1e-6)
        step = min(step, 10.0)  # cap step size

        # Update: lam = max(0, lam + step * g)
        lam = np.maximum(0.0, lam + step * g)

    # ---------- Local search polish on the best result ----------
    if best_result is None:
        best_result = result_greedy
        best_score = obj(best_result)

    res = best_result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[res[i]] += demands[i]

    def local_search(res, load):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 3.0:
                break

            # Single customer moves (vectorized inner loop)
            for i in range(n_cust):
                if time.time() > deadline - 3.0:
                    break
                f_old = res[i]
                # Compute delta for all f
                # delta_assign[i,f] = assign_cost[i,f] - assign_cost[i,f_old]
                # Capacity check: load[f] + demands[i] <= capacities[f]
                feasible_mask = (load + demands[i] <= capacities)
                feasible_mask[f_old] = False
                if not np.any(feasible_mask):
                    continue

                delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
                # Open cost delta
                delta_open = np.zeros(n_fac)
                if load[f_old] == demands[i]:
                    delta_open[f_old] = -open_cost[f_old]
                zero_load = (load == 0)
                delta_open[zero_load] = open_cost[zero_load]
                # But we only consider feasible moves
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf

                f_new = np.argmin(total_delta)
                if total_delta[f_new] < -1e-9:
                    res[i] = f_new
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    improved = True

            # Swap moves (sampled)
            if time.time() > deadline - 5.0:
                break
            n_swaps = 500
            for _ in range(n_swaps):
                if time.time() > deadline - 3.0:
                    break
                i, j = np.random.choice(n_cust, 2, replace=False)
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j:
                    continue
                if load[f_j] - demands[j] + demands[i] > capacities[f_j] + 1e-9:
                    continue
                if load[f_i] - demands[i] + demands[j] > capacities[f_i] + 1e-9:
                    continue
                delta = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                        (assign_cost[j, f_i] - assign_cost[j, f_j])
                if delta < -1e-9:
                    res[i], res[j] = res[j], res[i]
                    load[f_i] -= demands[i]
                    load[f_i] += demands[j]
                    load[f_j] -= demands[j]
                    load[f_j] += demands[i]
                    improved = True

            # Close-facility move: try to close a facility by moving all its customers
            if time.time() > deadline - 5.0:
                break
            for f in range(n_fac):
                if time.time() > deadline - 3.0:
                    break
                if load[f] == 0:
                    continue
                # Check if we can move all customers from f to other facilities
                custs_at_f = np.where(res == f)[0]
                if len(custs_at_f) == 0:
                    continue
                # Try to move each customer to the best alternative
                total_delta = 0.0
                new_assignments = {}
                feasible = True
                temp_load = load.copy()
                for i in custs_at_f:
                    # Find best g != f with capacity
                    best_g = -1
                    best_cost = np.inf
                    for g in range(n_fac):
                        if g == f:
                            continue
                        if temp_load[g] + demands[i] <= capacities[g] + 1e-9:
                            cost = assign_cost[i, g] - assign_cost[i, f]
                            if cost < best_cost:
                                best_cost = cost
                                best_g = g
                    if best_g == -1:
                        feasible = False
                        break
                    new_assignments[i] = best_g
                    temp_load[best_g] += demands[i]
                    total_delta += best_cost
                # Open cost savings
                total_delta -= open_cost[f]
                if feasible and total_delta < -1e-9:
                    for i in custs_at_f:
                        g = new_assignments[i]
                        res[i] = g
                        load[f] -= demands[i]
                        load[g] += demands[i]
                    improved = True

        return res, load

    res_final, load_final = local_search(res, load)
    if np.all(load_final <= capacities + 1e-9):
        score_final = obj(res_final)
        if score_final < best_score:
            best_result = res_final
            best_score = score_final

    return best_result.astype(int)
# EVOLVE-BLOCK-END
