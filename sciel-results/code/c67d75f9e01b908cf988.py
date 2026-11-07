"""Lagrangian relaxation solver for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: Lagrangian relaxation of capacity constraints with subgradient
    optimization, followed by a feasibility repair and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost. Returns inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def solve_uncapacitated(lam):
        """
        Solve the uncapacitated facility location problem with Lagrangian
        multipliers lam added to assignment costs.
        Returns (assignment, lagrangian_value).
        """
        # For each facility, the per-unit penalty is lam[f].
        # Customer c assigned to f pays: assign_cost[c,f] + lam[f]
        # Plus open_cost[f] if f is used.
        #
        # For a given subset S of open facilities, customer c goes to
        # argmin_{f in S} (assign_cost[c,f] + lam[f]).
        #
        # We use a greedy approach: start with all facilities "available",
        # but only open those that are used. Since the problem is small
        # (20 facilities, 100 customers), we can enumerate or use a
        # smart greedy.
        #
        # Actually, for uncapacitated FLP, the problem is still NP-hard in
        # general, but with 20 facilities we can use a simple approach:
        # For each customer, the "effective cost" of facility f is
        # assign_cost[c,f] + lam[f] + open_cost[f] * (1/expected_usage).
        #
        # Better: use the fact that with 20 facilities, we can do a
        # "best response" iteration:
        # 1. Start with all facilities open.
        # 2. Assign each customer to cheapest (assign_cost[c,f] + lam[f]).
        # 3. Close any facility with no customers.
        # 4. Re-assign customers who were assigned to closed facilities.
        # 5. Repeat until stable.

        adj_cost = assign_cost + lam[np.newaxis, :]  # (n_customers, n_facilities)

        # Start with all facilities "open"
        open_mask = np.ones(n_facilities, dtype=bool)

        for _ in range(50):
            # Assign each customer to cheapest open facility
            masked = adj_cost.copy()
            masked[:, ~open_mask] = np.inf
            assignment = np.argmin(masked, axis=1)

            # Check which facilities are actually used
            used = np.zeros(n_facilities, dtype=bool)
            for c in range(n_customers):
                used[assignment[c]] = True

            # Close unused facilities
            new_open = used.copy()

            if np.array_equal(new_open, open_mask):
                break
            open_mask = new_open

        # Final assignment
        masked = adj_cost.copy()
        masked[:, ~open_mask] = np.inf
        assignment = np.argmin(masked, axis=1)

        # Compute Lagrangian value
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[assignment[c]] += demands[c]

        # Lagrangian value = sum of (assign_cost + lam) for each assignment
        # + sum of open_cost for used facilities
        # - sum of lam * capacity (the dual term)
        lag_value = 0.0
        for c in range(n_customers):
            lag_value += adj_cost[c, assignment[c]]
        for f in range(n_facilities):
            if used[f]:
                lag_value += open_cost[f]
        lag_value -= np.dot(lam, capacities)

        return assignment, lag_value, load

    def repair(assignment):
        """
        Repair an infeasible assignment by moving customers from
        overloaded facilities to feasible ones.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[assignment[c]] += demands[c]

        # Iteratively fix overloaded facilities
        for _ in range(100):
            overloaded = load > capacities + 1e-9
            if not np.any(overloaded):
                break

            for f in np.where(overloaded)[0]:
                excess = load[f] - capacities[f]
                if excess <= 0:
                    continue

                # Find customers at f, sorted by "regret" (cost of moving)
                customers_at_f = [c for c in range(n_customers) if assignment[c] == f]
                # Sort by demand descending (move big ones first to reduce moves)
                customers_at_f.sort(key=lambda c: -demands[c])

                for c in customers_at_f:
                    if load[f] <= capacities[f] + 1e-9:
                        break
                    # Find cheapest feasible facility to move to
                    load[f] -= demands[c]
                    feasible = load + demands[c] <= capacities + 1e-9
                    feasible[f] = False
                    if np.any(feasible):
                        costs = assign_cost[c].copy()
                        costs[~feasible] = np.inf
                        new_f = int(np.argmin(costs))
                        assignment[c] = new_f
                        load[new_f] += demands[c]
                    else:
                        # No feasible facility; put back
                        assignment[c] = f
                        load[f] += demands[c]

        return assignment

    def local_search(assignment, max_passes=30):
        """Fast local search: single reassignments + pairwise swaps."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_cost = current_cost
        best_assignment = assignment.copy()

        for _pass in range(max_passes):
            if time.time() > deadline - 3:
                break

            improved = False

            # Single reassignment
            for c in range(n_customers):
                if time.time() > deadline - 3:
                    break
                old_f = int(assignment[c])
                load[old_f] -= demands[c]

                deltas = assign_cost[c] - assign_cost[c, old_f]
                newly_used = (load == 0)
                deltas[newly_used] += open_cost[newly_used]
                if load[old_f] == 0:
                    deltas -= open_cost[old_f]

                feasible = load + demands[c] <= capacities + 1e-9
                feasible[old_f] = False
                deltas[~feasible] = np.inf

                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]

                if best_delta < -1e-9 and feasible[best_idx]:
                    assignment[c] = best_idx
                    load[best_idx] += demands[c]
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = assignment.copy()
                else:
                    load[old_f] += demands[c]

            # Pairwise swaps
            if time.time() > deadline - 3:
                break

            for c1 in range(n_customers):
                if time.time() > deadline - 3:
                    break
                f1 = int(assignment[c1])
                for c2 in range(c1 + 1, n_customers):
                    if time.time() > deadline - 3:
                        break
                    f2 = int(assignment[c2])
                    if f1 == f2:
                        continue

                    load_f1_new = load[f1] - demands[c1] + demands[c2]
                    load_f2_new = load[f2] - demands[c2] + demands[c1]
                    if load_f1_new > capacities[f1] + 1e-9 or load_f2_new > capacities[f2] + 1e-9:
                        continue

                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (assign_cost[c1, f1] + assign_cost[c2, f2])

                    if load[f1] > 0 and load_f1_new == 0:
                        delta -= open_cost[f1]
                    elif load[f1] == 0 and load_f1_new > 0:
                        delta += open_cost[f1]

                    if load[f2] > 0 and load_f2_new == 0:
                        delta -= open_cost[f2]
                    elif load[f2] == 0 and load_f2_new > 0:
                        delta += open_cost[f2]

                    if delta < -1e-9:
                        assignment[c1] = f2
                        assignment[c2] = f1
                        load[f1] = load_f1_new
                        load[f2] = load_f2_new
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()

            if not improved:
                break

        return best_assignment

    # Phase 0: Get a feasible initial solution via greedy
    def greedy_assignment():
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)  # Largest first
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            if np.any(feasible):
                f = int(np.argmin(costs))
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    best_solution = greedy_assignment()
    best_cost = compute_cost(best_solution) if best_solution is not None else np.inf

    # Phase 1: Lagrangian relaxation with subgradient optimization
    lam = np.zeros(n_facilities)
    best_lagrangian_solution = None
    best_lagrangian_cost = np.inf

    # Subgradient parameters
    max_subgrad_iters = 2000
    step_size_factor = 1.0
    best_lagrangian_value = np.inf
    patience = 0
    max_patience = 100

    for it in range(max_subgrad_iters):
        if time.time() > deadline - 30:
            break

        assignment, lag_value, load = solve_uncapacitated(lam)

        # Track best Lagrangian lower bound
        if lag_value < best_lagrangian_value:
            best_lagrangian_value = lag_value

        # Check if this solution is feasible
        if np.all(load <= capacities + 1e-9):
            cost = compute_cost(assignment)
            if cost < best_cost:
                best_cost = cost
                best_solution = assignment.copy()
            if cost < best_lagrangian_cost:
                best_lagrangian_cost = cost
                best_lagrangian_solution = assignment.copy()
            patience = 0
        else:
            patience += 1

        if patience > max_patience and it > 50:
            break

        # Subgradient update
        # Subgradient = load - capacities (positive if overloaded)
        subgrad = load - capacities

        # Step size: alpha * (best_upper_bound - lag_value) / ||subgrad||^2
        if best_lagrangian_cost < np.inf:
            gap = best_lagrangian_cost - lag_value
            if gap <= 0:
                gap = 1.0
            norm_sq = np.dot(subgrad, subgrad)
            if norm_sq < 1e-12:
                alpha = 0.01
            else:
                alpha = step_size_factor * gap / norm_sq
        else:
            alpha = 0.01

        # Decay step size
        step_size_factor *= 0.995

        lam = np.maximum(0, lam + alpha * subgrad)

    # Phase 2: If we have a feasible solution from Lagrangian, polish it
    if best_lagrangian_solution is not None and time.time() < deadline - 10:
        polished = local_search(best_lagrangian_solution, max_passes=50)
        cost = compute_cost(polished)
        if cost < best_cost:
            best_cost = cost
            best_solution = polished

    # Phase 3: If no feasible solution yet, use repair on best Lagrangian solution
    if best_solution is None:
        # Try to get any feasible solution
        assignment, _, _ = solve_uncapacitated(lam)
        repaired = repair(assignment)
        cost = compute_cost(repaired)
        if cost < best_cost:
            best_cost = cost
            best_solution = repaired

    # Phase 4: Final local search polish on the best solution
    if best_solution is not None and time.time() < deadline - 2:
        final = local_search(best_solution, max_passes=100)
        cost = compute_cost(final)
        if cost < best_cost:
            best_cost = cost
            best_solution = final

    # Fallback
    if best_solution is None:
        best_solution = greedy_assignment()
        if best_solution is None:
            # Last resort: round-robin
            best_solution = np.zeros(n_customers, dtype=int)
            load = np.zeros(n_facilities)
            for c in range(n_customers):
                f = c % n_facilities
                while load[f] + demands[c] > capacities[f]:
                    f = (f + 1) % n_facilities
                best_solution[c] = f
                load[f] += demands[c]

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
