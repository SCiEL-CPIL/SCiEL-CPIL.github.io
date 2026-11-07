"""Stronger solver for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: diverse initial solutions + fast vectorized local search with swaps.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    total_demand = demands.sum()

    def compute_cost(assignment):
        """Compute total cost for an assignment, return inf if infeasible."""
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

    def greedy_assignment(order=None, random_seed=None):
        """Greedy: assign in given order (or random) to cheapest feasible facility."""
        if order is None:
            order = np.arange(n_customers)

        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)

        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf

            if random_seed is not None:
                rng = np.random.RandomState(random_seed)
                mean_feas = np.mean(costs[feasible]) if np.any(feasible) else 1.0
                costs += rng.uniform(0, 0.1 * mean_feas, size=n_facilities)

            f = int(np.argmin(costs))
            if f >= 0 and feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None

        return result

    def local_search(assignment, max_iter=500):
        """Improve assignment via single reassignments and swaps."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_cost = current_cost

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            # Try reassigning each customer to a different facility
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]

                best_new_f = old_f
                best_new_cost = np.inf

                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue

                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]

                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]

                    if delta < best_new_cost - 1e-9:
                        best_new_cost = delta
                        best_new_f = new_f

                if best_new_f != old_f and best_new_cost < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_new_cost
                    improved = True
                else:
                    load[old_f] += demands[c]

            if not improved:
                break

        return assignment

    def simulated_annealing(assignment, max_iter=2000, T_start=10.0, T_end=0.01):
        """SA-based improvement."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_assignment = assignment.copy()
        best_cost = current_cost

        n_iter = max_iter
        for it in range(n_iter):
            if time.time() > deadline:
                break

            T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))

            c = np.random.randint(n_customers)
            old_f = int(assignment[c])

            load[old_f] -= demands[c]
            feasible = np.where(load + demands[c] <= capacities)[0]

            if len(feasible) == 0:
                load[old_f] += demands[c]
                continue

            new_f = int(feasible[np.random.randint(len(feasible))])

            if new_f == old_f:
                load[old_f] += demands[c]
                continue

            delta = assign_cost[c, new_f] - assign_cost[c, old_f]
            if load[new_f] == 0:
                delta += open_cost[new_f]
            if load[old_f] == 0:
                delta -= open_cost[old_f]

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                assignment[c] = new_f
                load[new_f] += demands[c]
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
            else:
                load[old_f] += demands[c]

        return best_assignment

    # Phase 1: Generate diverse candidate solutions
    best_solution = None
    best_cost = np.inf

    # Deterministic greedy variants with different orderings
    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
        np.argsort(assign_cost.sum(axis=1) / demands),
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Randomized greedy variants
    rng = np.random.RandomState(42)
    n_random_greedy = 30
    for i in range(n_random_greedy):
        if time.time() > deadline:
            break

        order = rng.permutation(n_customers)
        sol = greedy_assignment(order, random_seed=rng.randint(100000))
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

        bias = rng.random()
        score = bias * (-demands) + (1 - bias) * assign_cost.sum(axis=1)
        order = np.argsort(score)
        sol = greedy_assignment(order, random_seed=rng.randint(100000))
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Phase 2: Local search on best solutions
    if best_solution is not None and time.time() < deadline:
        improved = local_search(best_solution)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved

    # Phase 3: More diverse candidates + SA
    n_sa_tries = 10
    for i in range(n_sa_tries):
        if time.time() > deadline - 20:
            break

        if i % 2 == 0:
            order = rng.permutation(n_customers)
            sol = greedy_assignment(order, random_seed=rng.randint(100000))
        else:
            if best_solution is None:
                sol = greedy_assignment(np.argsort(-demands))
            else:
                sol = best_solution.copy()
                for _ in range(rng.randint(3, 10)):
                    c = rng.randint(n_customers)
                    old_f = int(sol[c])
                    load_tmp = np.zeros(n_facilities)
                    for cc in range(n_customers):
                        load_tmp[int(sol[cc])] += demands[cc]
                    load_tmp[old_f] -= demands[c]
                    feasible = np.where(load_tmp + demands[c] <= capacities)[0]
                    if len(feasible) > 1:
                        new_f = int(feasible[rng.randint(len(feasible))])
                        sol[c] = new_f

        if sol is None:
            continue

        sa_sol = simulated_annealing(sol, max_iter=1000)
        cost = compute_cost(sa_sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = sa_sol.copy()

        if time.time() < deadline - 10:
            ls_sol = local_search(sa_sol, max_iter=300)
            cost = compute_cost(ls_sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = ls_sol.copy()

    # Final local search on best
    if best_solution is not None and time.time() < deadline:
        final = local_search(best_solution, max_iter=1000)
        cost = compute_cost(final)
        if cost < best_cost:
            best_cost = cost
            best_solution = final

    if best_solution is None:
        best_solution = greedy_assignment(np.argsort(-demands))
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
