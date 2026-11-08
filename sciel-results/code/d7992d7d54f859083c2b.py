# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def calc_cost(assignment):
        total_assign = np.sum(assign_cost[np.arange(n_cust), assignment])
        used = np.unique(assignment)
        total_open = np.sum(open_cost[used])
        return total_assign + total_open

    def is_valid(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.bincount(assignment, weights=demands, minlength=n_fac)
        return bool(np.all(loads <= capacities))

    def greedy_assign(order, load_offset=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        result = np.full(n_cust, -1, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        if load_offset is not None:
            load += load_offset
        for i in order:
            costs = assign_cost[i]
            fac_order = np.argsort(costs)
            placed = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    placed = True
                    break
            if not placed:
                # Fallback: facility with most remaining capacity
                remaining = capacities - load
                f = int(np.argmax(remaining))
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    placed = True
            if not placed:
                return None
        return result

    def lagrangian_guided_init():
        """
        Use Lagrangian relaxation with subgradient to get guided initial solutions.
        Minimizes sum_i c[i, f] + sum_f open_cost[f]*y_f + sum_f lambda_f * max(0, load_f - cap_f)
        where y_f = 1 if facility is used.
        We iterate: for given multipliers, solve a modified assignment (cheapest effective cost),
        then update multipliers via subgradient.
        """
        rng = np.random.default_rng(123)
        best_sol = None
        best_cost = np.inf

        # Initialize multipliers
        lambdas = np.zeros(n_fac)
        step_size = 1.0
        best_mult = lambdas.copy()
        best_mult_cost = np.inf

        for it in range(300):
            if time.time() > deadline - 60.0:
                break

            # Solve: for each customer, pick facility minimizing assign_cost[i,f] + lambda_f
            # (ignoring open cost for simplicity in the relaxation)
            eff_cost = assign_cost + lambdas[np.newaxis, :]
            assignment = np.argmin(eff_cost, axis=1).astype(np.int64)

            # Compute loads
            loads = np.bincount(assignment, weights=demands, minlength=n_fac)
            violations = np.maximum(0, loads - capacities)
            total_violation = np.sum(violations)

            # Objective of Lagrangian (lower bound-ish)
            obj = np.sum(assign_cost[np.arange(n_cust), assignment]) + np.sum(lambdas * loads)
            # Actually the Lagrangian value:
            # L = sum_i min_f (c[i,f] + lambda_f) + sum_f lambda_f * cap_f
            min_per_cust = np.min(eff_cost, axis=1)
            L_val = np.sum(min_per_cust) + np.sum(lambdas * capacities)

            if L_val < best_mult_cost:
                best_mult_cost = L_val
                best_mult = lambdas.copy()

            # Subgradient update
            # g_f = load_f - cap_f
            g = loads - capacities
            # Step size: standard Polyak-style
            gap = obj - (-np.inf)  # We don't have a good lower bound, use heuristic
            # Use diminishing step
            step = step_size / (1.0 + 0.01 * it)
            lambdas = np.maximum(0, lambdas + step * g)

            # If feasible, evaluate
            if total_violation == 0:
                cost = calc_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = assignment.copy()

        # Round the best multiplier solution: use best_mult to guide greedy
        if best_sol is None:
            # Use best_mult to create a guided greedy
            eff_cost = assign_cost + best_mult[np.newaxis, :]
            # Sort customers by "difficulty" (fewest feasible options under capacity)
            # Just use demand descending
            order = np.argsort(-demands)
            best_sol = greedy_assign(order)
            if best_sol is None:
                order = np.arange(n_cust)
                best_sol = greedy_assign(order)

        return best_sol

    def local_search(initial_assignment, max_time):
        """
        Fast vectorized local search with Relocate moves.
        Uses numpy to evaluate all potential moves at once where possible.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calc_cost(current_assignment)

        load = np.zeros(n_fac, dtype=np.int64)
        cust_count = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            f = current_assignment[i]
            load[f] += demands[i]
            cust_count[f] += 1

        local_deadline = time.time() + max_time
        improved = True
        no_improve_count = 0
        max_no_improve = 3

        while improved and time.time() < local_deadline:
            improved = False
            # Try relocate moves for all customers in random order
            order = np.random.permutation(n_cust)

            for i in order:
                if time.time() > local_deadline:
                    break

                f_from = current_assignment[i]
                d_i = demands[i]

                # Compute cost delta for moving i to each facility
                # delta_assign = assign_cost[i, f] - assign_cost[i, f_from]
                # delta_open: -open_cost[f_from] if cust_count[f_from]==1, +open_cost[f] if cust_count[f]==0
                delta_assign = assign_cost[i] - assign_cost[i, f_from]

                # Open cost adjustment
                delta_open = np.zeros(n_fac)
                if cust_count[f_from] == 1:
                    delta_open -= open_cost[f_from]
                # For f_to that are empty
                empty_mask = (cust_count == 0)
                delta_open[empty_mask] += open_cost[empty_mask]

                total_delta = delta_assign + delta_open

                # Feasibility: load[f] + d_i <= capacities[f], f != f_from
                feasible = (load + d_i <= capacities)
                feasible[f_from] = False

                # Find best feasible move
                if np.any(feasible):
                    # Among feasible, find min total_delta
                    costs_feasible = np.where(feasible, total_delta, np.inf)
                    best_f = int(np.argmin(costs_feasible))
                    best_delta = costs_feasible[best_f]

                    if best_delta < -1e-9:
                        # Apply move
                        current_assignment[i] = best_f
                        load[f_from] -= d_i
                        load[best_f] += d_i
                        cust_count[f_from] -= 1
                        cust_count[best_f] += 1
                        current_cost += best_delta
                        improved = True
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            if not improved:
                break

        return current_assignment, current_cost

    def swap_local_search(initial_assignment, max_time):
        """
        Local search with swap moves between pairs of customers in different facilities.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calc_cost(current_assignment)

        load = np.zeros(n_fac, dtype=np.int64)
        cust_count = np.zeros(n_fac, dtype=np.int64)
        for i in range(n_cust):
            f = current_assignment[i]
            load[f] += demands[i]
            cust_count[f] += 1

        local_deadline = time.time() + max_time
        improved = True
        rng = np.random.default_rng(777)

        while improved and time.time() < local_deadline:
            improved = False

            # Randomly sample pairs of customers
            for _ in range(2000):
                if time.time() > local_deadline:
                    break

                i, j = rng.integers(0, n_cust, size=2)
                if i == j:
                    continue
                f1 = current_assignment[i]
                f2 = current_assignment[j]
                if f1 == f2:
                    continue

                d_i = demands[i]
                d_j = demands[j]

                # Check capacity after swap
                new_load_f1 = load[f1] - d_i + d_j
                new_load_f2 = load[f2] - d_j + d_i
                if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                    continue

                # Cost delta
                delta = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])

                # Open cost: both facilities remain non-empty (since i goes to f2, j goes to f1)
                # So delta_open = 0

                if delta < -1e-9:
                    current_assignment[i] = f2
                    current_assignment[j] = f1
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    current_cost += delta
                    improved = True

        return current_assignment, current_cost

    # Phase 1: Generate diverse initial solutions
    candidates = []

    # Lagrangian-guided solution
    lag_sol = lagrangian_guided_init()
    if lag_sol is not None and is_valid(lag_sol):
        candidates.append(lag_sol)

    # Greedy: largest demand first
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)

    # Greedy: smallest demand first
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)

    # Greedy: by minimum assign cost
    min_costs = np.min(assign_cost, axis=1)
    order3 = np.argsort(min_costs)
    cand3 = greedy_assign(order3)
    if cand3 is not None and is_valid(cand3):
        candidates.append(cand3)

    # Random greedy starts
    rng_init = np.random.default_rng(42)
    for _ in range(8):
        order_r = rng_init.permutation(n_cust)
        cand_r = greedy_assign(order_r)
        if cand_r is not None and is_valid(cand_r):
            candidates.append(cand_r)

    if not candidates:
        # Emergency: assign all to facility 0 (may be invalid but better than crash)
        return np.zeros(n_cust, dtype=np.int64)

    # Sort by cost
    candidate_costs = [calc_cost(c) for c in candidates]
    sorted_idx = np.argsort(candidate_costs)
    candidates = [candidates[i] for i in sorted_idx]
    candidate_costs = [candidate_costs[i] for i in sorted_idx]

    best_assignment = candidates[0]
    best_cost = candidate_costs[0]

    # Phase 2: Local search on top candidates
    # Allocate time: give more time to better candidates
    remaining = deadline - 5.0 - time.time()
    if remaining <= 0:
        return best_assignment

    # Run LS on top 3 candidates
    num_ls = min(3, len(candidates))
    # Time allocation: 50%, 30%, 20% of remaining
    time_fracs = [0.5, 0.3, 0.2][:num_ls]

    for idx in range(num_ls):
        if time.time() > deadline - 5.0:
            break

        ls_time = remaining * time_fracs[idx]
        if ls_time < 1.0:
            ls_time = min(remaining, 5.0)

        result, cost = local_search(candidates[idx], ls_time)
        if is_valid(result) and cost < best_cost:
            best_assignment = result
            best_cost = cost

        # Also try swap LS on the best result so far
        remaining2 = deadline - 5.0 - time.time()
        if remaining2 > 3.0 and best_cost < candidate_costs[0]:
            swap_time = min(remaining2 * 0.3, 15.0)
            result2, cost2 = swap_local_search(best_assignment, swap_time)
            if is_valid(result2) and cost2 < best_cost:
                best_assignment = result2
                best_cost = cost2

    # Phase 3: Final polish with another round of relocate LS if time permits
    remaining3 = deadline - 3.0 - time.time()
    if remaining3 > 2.0:
        result3, cost3 = local_search(best_assignment, remaining3)
        if is_valid(result3) and cost3 < best_cost:
            best_assignment = result3
            best_cost = cost3

    return best_assignment.astype(np.int64)

# EVOLVE-BLOCK-END
