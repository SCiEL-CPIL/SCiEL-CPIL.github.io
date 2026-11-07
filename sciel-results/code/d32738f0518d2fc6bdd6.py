"""Lagrangian relaxation + repair + swap-based local search for CFLP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using Lagrangian relaxation,
    subgradient optimization, repair, and swap-based local search.
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

    def solve_relaxed(lambdas):
        """
        Solve the Lagrangian-relaxed problem:
        minimize sum_c [assign_cost[c,f] + lambda_f * demands[c]] + sum_f [open_cost[f] + lambda_f * (0 - capacities[f])]
        s.t. each customer assigned to one facility.
        The optimal facility for each customer is argmin_f (assign_cost[c,f] + lambda_f * demands[c]).
        Open facilities are those with at least one customer assigned.
        Returns (assignment, cost, loads).
        """
        # Effective cost for customer c at facility f: assign_cost[c,f] + lambda_f * demands[c]
        # But we also need to account for open_cost: if facility f is opened, add open_cost[f].
        # In the relaxed problem, the decision is per-customer independent once we decide which facilities are open.
        # However, the open decision is coupled. A common approach: for each customer, pick the facility
        # that minimizes assign_cost[c,f] + lambda_f * demands[c], then open all used facilities.
        # This is a heuristic for the relaxed problem.
        
        # Actually, the Lagrangian dual function is:
        # g(lambda) = min_{x in X} sum_c sum_f x_cf (a_cf + lambda_f d_c) + sum_f open_cost_f y_f + sum_f lambda_f (0 - cap_f)
        # where X is the assignment polytope (each customer assigned to one facility), y_f = 1 if facility f used.
        # The inner minimization is not trivially separable because of y_f.
        # A common approximation: ignore the coupling and assign each customer to argmin_f (a_cf + lambda_f d_c),
        # then open all facilities that are used. This gives a feasible solution to the relaxed problem
        # (but not necessarily optimal for the relaxed problem).
        
        # Let's use a better approach: for each facility f, compute the "effective" cost for customer c:
        # cost_cf = assign_cost[c, f] + lambda_f * demands[c]
        # Then assign each customer to the facility with minimum cost_cf.
        # The open cost is added for each facility that ends up being used.
        
        eff_cost = assign_cost + demands[:, np.newaxis] * lambdas[np.newaxis, :]
        assignment = np.argmin(eff_cost, axis=1)
        
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        used = load > 0
        cost = np.sum(assign_cost[np.arange(n_customers), assignment]) + np.sum(open_cost[used])
        
        return assignment, cost, load

    def subgradient_optimize(max_iter=500):
        """
        Use subgradient method to find good Lagrange multipliers.
        """
        lambdas = np.zeros(n_facilities)
        best_lambdas = lambdas.copy()
        best_assign = None
        best_cost = np.inf
        best_load = None
        
        # Initial: solve relaxed problem with lambdas=0
        assign_init, cost_init, load_init = solve_relaxed(lambdas)
        best_assign = assign_init.copy()
        best_cost = cost_init
        best_load = load_init.copy()
        
        # Subgradient: g'(lambda_f) = load_f - capacity_f
        step_size_scale = 2.0
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute subgradient
            _, _, load = solve_relaxed(lambdas)
            subgrad = load - capacities
            
            # Compute dual objective
            _, dual_cost, _ = solve_relaxed(lambdas)
            dual_obj = dual_cost - np.dot(lambdas, capacities)
            
            # Update step size
            # Use a diminishing step size
            step = step_size_scale / (it + 1)
            
            # Update lambdas
            lambdas = np.maximum(0, lambdas - step * subgrad)
            
            # Solve relaxed problem with new lambdas
            assign_new, cost_new, load_new = solve_relaxed(lambdas)
            
            # Track best feasible or near-feasible solution
            # For the purpose of getting a good initial solution, we track the best cost
            # among solutions that are close to feasible.
            max_violation = np.max(np.maximum(0, load_new - capacities))
            if max_violation < 1e-6:
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_assign = assign_new.copy()
                    best_load = load_new.copy()
            else:
                # Penalize infeasibility
                penalized_cost = cost_new + 10.0 * max_violation
                if penalized_cost < best_cost + 5.0 * max_violation:
                    # Keep as candidate if not too bad
                    if best_assign is None or cost_new < best_cost:
                        best_cost = cost_new
                        best_assign = assign_new.copy()
                        best_load = load_new.copy()
        
        return best_assign, best_cost

    def repair(assignment):
        """
        Repair an infeasible assignment by moving customers from overloaded facilities
        to the cheapest feasible alternative.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        
        # Identify overloaded facilities
        for f in range(n_facilities):
            if load[f] <= capacities[f] + 1e-9:
                continue
            
            # Find customers assigned to f, sort by demand (move largest first)
            customers_at_f = [c for c in range(n_customers) if int(assignment[c]) == f]
            customers_at_f.sort(key=lambda c: -demands[c])
            
            for c in customers_at_f:
                if load[f] <= capacities[f] + 1e-9:
                    break
                
                # Find cheapest feasible facility for c
                best_f = -1
                best_cost = np.inf
                for new_f in range(n_facilities):
                    if new_f == f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    # Cost: assign_cost + open_cost if new facility was empty
                    cost = assign_cost[c, new_f]
                    if load[new_f] == 0:
                        cost += open_cost[new_f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = new_f
                
                if best_f == -1:
                    # No feasible facility; try to swap with a customer at another facility
                    # For now, leave it (shouldn't happen if total demand <= total capacity)
                    continue
                
                # Move customer c from f to best_f
                load[f] -= demands[c]
                load[best_f] += demands[c]
                assignment[c] = best_f
        
        return assignment

    def local_search_with_swaps(assignment, max_iter=500):
        """
        Local search with single reassignments and pairwise swaps.
        """
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

            # Single reassignment
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

            # Pairwise swaps
            if not improved:
                for c1 in range(n_customers):
                    f1 = int(assignment[c1])
                    for c2 in range(c1 + 1, n_customers):
                        f2 = int(assignment[c2])
                        if f1 == f2:
                            continue
                        
                        # Try swapping: c1 -> f2, c2 -> f1
                        if load[f1] - demands[c1] + demands[c2] > capacities[f1] + 1e-9:
                            continue
                        if load[f2] - demands[c2] + demands[c1] > capacities[f2] + 1e-9:
                            continue
                        
                        # Compute delta
                        old_assign_cost = assign_cost[c1, f1] + assign_cost[c2, f2]
                        new_assign_cost = assign_cost[c1, f2] + assign_cost[c2, f1]
                        delta = new_assign_cost - old_assign_cost
                        
                        # Open cost changes
                        old_open = 0.0
                        new_open = 0.0
                        for f in [f1, f2]:
                            if load[f] == 0:
                                old_open += open_cost[f]
                            elif load[f] == demands[c1] if f == f1 else load[f] == demands[c2]:
                                # This facility will become empty after swap
                                new_open += 0  # already counted in old
                            else:
                                new_open += 0  # still open
                        # More carefully:
                        # f1: before load = load[f1], after load = load[f1] - demands[c1] + demands[c2]
                        # f2: before load = load[f2], after load = load[f2] - demands[c2] + demands[c1]
                        if load[f1] == 0:
                            pass  # not possible since c1 is at f1
                        elif load[f1] == demands[c1]:
                            # f1 becomes: demands[c2]
                            if demands[c2] > 0:
                                # stays open
                                pass
                        if load[f2] == 0:
                            pass
                        elif load[f2] == demands[c2]:
                            # f2 becomes: demands[c1]
                            if demands[c1] > 0:
                                # stays open
                                pass
                        
                        # If a facility becomes empty, we save its open cost
                        if load[f1] == demands[c1] and demands[c2] == 0:
                            delta -= open_cost[f1]
                        if load[f2] == demands[c2] and demands[c1] == 0:
                            delta -= open_cost[f2]
                        
                        # If a facility becomes open (was empty), we add its open cost
                        # But f1 and f2 are both non-empty before swap, so they stay non-empty after
                        # unless one of the customers has demand 0 (not the case here)
                        
                        if delta < -1e-9:
                            # Perform swap
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = load[f1] - demands[c1] + demands[c2]
                            load[f2] = load[f2] - demands[c2] + demands[c1]
                            current_cost += delta
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        return assignment

    # Phase 1: Lagrangian relaxation to get a good initial solution
    lagr_assign, lagr_cost = subgradient_optimize(max_iter=300)
    
    # Phase 2: Repair if infeasible
    if lagr_assign is not None:
        load_check = np.zeros(n_facilities)
        for c in range(n_customers):
            load_check[int(lagr_assign[c])] += demands[c]
        if np.any(load_check > capacities + 1e-9):
            lagr_assign = repair(lagr_assign)
    
    # Phase 3: Also generate a greedy solution as backup
    def greedy_assignment(order=None):
        if order is None:
            order = np.argsort(-demands)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            if f >= 0 and feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    greedy_sol = greedy_assignment()

    # Phase 4: Local search with swaps on the best initial solution
    best_solution = None
    best_cost = np.inf

    candidates = []
    if lagr_assign is not None:
        candidates.append(lagr_assign)
    if greedy_sol is not None:
        candidates.append(greedy_sol)

    for sol in candidates:
        if time.time() > deadline:
            break
        improved = local_search_with_swaps(sol, max_iter=300)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved.copy()

    # Phase 5: Final polish
    if best_solution is not None and time.time() < deadline:
        final = local_search_with_swaps(best_solution, max_iter=500)
        cost = compute_cost(final)
        if cost < best_cost:
            best_cost = cost
            best_solution = final

    if best_solution is None:
        best_solution = greedy_sol
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)
# EVOLVE-BLOCK-END
