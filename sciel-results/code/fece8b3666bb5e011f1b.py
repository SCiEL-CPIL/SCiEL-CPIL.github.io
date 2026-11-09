"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Precompute sorted facility indices for each customer (cheapest first)
    sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
    
    def calculate_cost(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        np.add.at(loads, assignment, demands)
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order, dual_prices=None):
        """
        Assign customers in given order to cheapest feasible facility.
        dual_prices: Lagrangian multipliers for facilities, adjust effective cost.
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if dual_prices is not None:
            # Effective cost: assign_cost[i,f] + dual_prices[f] * demands[i]
            eff_cost = assign_cost + dual_prices[np.newaxis, :] * demands[:, np.newaxis]
            adj_sorted = np.argsort(eff_cost, axis=1)
        else:
            adj_sorted = sorted_fac_by_cost
            
        for i in order:
            d_i = demands[i]
            assigned = False
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += d_i
                    assigned = True
                    break
            if not assigned:
                # Fallback: pick facility with most remaining capacity
                best_f = -1
                max_rem = -1
                for f in range(n_fac):
                    rem = capacities[f] - load[f]
                    if rem >= d_i and rem > max_rem:
                        max_rem = rem
                        best_f = f
                if best_f == -1:
                    return None
                result[i] = best_f
                load[best_f] += d_i
        return result

    def subgradient_optimize(max_time=15.0):
        """
        Use subgradient method to find dual prices that guide greedy to good solutions.
        Returns the best feasible solution found during the process.
        """
        best_solution = None
        best_cost = float('inf')
        
        # Initialize dual prices
        dual = np.zeros(n_fac)
        step_size = 10.0
        decay = 0.95
        
        local_deadline = time.time() + max_time
        order = np.argsort(-demands)  # Largest first for stability
        
        while time.time() < local_deadline:
            # Solve greedy with current dual prices
            sol = greedy_assign(order, dual)
            if sol is not None and is_valid(sol):
                cost = calculate_cost(sol)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol.copy()
            
            # Compute subgradient: for each facility, load - capacity
            loads = np.zeros(n_fac)
            np.add.at(loads, sol, demands)
            subgrad = loads - capacities
            
            # Update dual prices
            # Project onto non-negative orthant (duals >= 0)
            dual = np.maximum(0, dual + step_size * subgrad)
            step_size *= decay
            
            # Also try a few random perturbations of dual prices
            if time.time() < local_deadline - 2.0:
                for _ in range(3):
                    perturbed_dual = np.maximum(0, dual + np.random.randn(n_fac) * 2.0)
                    sol_p = greedy_assign(order, perturbed_dual)
                    if sol_p is not None and is_valid(sol_p):
                        cost_p = calculate_cost(sol_p)
                        if cost_p < best_cost:
                            best_cost = cost_p
                            best_solution = sol_p.copy()
        
        return best_solution, best_cost

    def local_search(initial_assignment, max_time=20.0):
        """
        Local search with relocate and swap moves.
        Uses best-improvement strategy for relocate.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        local_deadline = time.time() + max_time
        improved = True
        iter_count = 0
        max_iters = 20000
        
        while improved and iter_count < max_iters:
            if time.time() > local_deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves - best improvement
            best_delta = 0.0
            best_i = -1
            best_f_to = -1
            best_f_from = -1
            
            for i in range(n_cust):
                if time.time() > local_deadline:
                    break
                f_from = current_assignment[i]
                
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    delta_open = 0.0
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_i = i
                        best_f_to = f_to
                        best_f_from = f_from
            
            if best_i != -1:
                current_assignment[best_i] = best_f_to
                load[best_f_from] -= demands[best_i]
                load[best_f_to] += demands[best_i]
                cust_count[best_f_from] -= 1
                cust_count[best_f_to] += 1
                current_cost += best_delta
                improved = True
                continue  # Restart scan after improvement
            
            # 2. Swap moves - best improvement
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            best_swap_delta = 0.0
            best_i1 = -1
            best_i2 = -1
            best_f1 = -1
            best_f2 = -1
            
            for idx1, f1 in enumerate(used_facs):
                if time.time() > local_deadline:
                    break
                for f2 in used_facs[idx1+1:]:
                    if time.time() > local_deadline:
                        break
                    
                    custs_f1 = np.where(current_assignment == f1)[0]
                    custs_f2 = np.where(current_assignment == f2)[0]
                    
                    for i in custs_f1:
                        if time.time() > local_deadline:
                            break
                        for j in custs_f2:
                            if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-9:
                                continue
                            if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-9:
                                continue
                            
                            delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - \
                                           (assign_cost[i, f1] + assign_cost[j, f2])
                            
                            if delta_assign < best_swap_delta:
                                best_swap_delta = delta_assign
                                best_i1 = i
                                best_i2 = j
                                best_f1 = f1
                                best_f2 = f2
            
            if best_i1 != -1:
                current_assignment[best_i1] = best_f2
                current_assignment[best_i2] = best_f1
                load[best_f1] -= demands[best_i1]
                load[best_f1] += demands[best_i2]
                load[best_f2] -= demands[best_i2]
                load[best_f2] += demands[best_i1]
                current_cost += best_swap_delta
                improved = True
        
        return current_assignment, current_cost

    # Phase 1: Generate diverse initial solutions
    candidates = []
    
    # Basic greedy with different orderings
    orderings = [
        np.argsort(-demands),      # Largest first
        np.argsort(demands),       # Smallest first
        np.argsort(np.min(assign_cost, axis=1)),  # Cheapest first
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        orderings.append(rng.permutation(n_cust))
    
    for order in orderings:
        cand = greedy_assign(order)
        if cand is not None and is_valid(cand):
            candidates.append(cand)
    
    # Phase 2: Subgradient optimization to find good dual prices
    if time.time() < deadline - 60.0:
        sg_time = min(deadline - time.time() - 55.0, 20.0)
        sg_sol, sg_cost = subgradient_optimize(max_time=sg_time)
        if sg_sol is not None:
            candidates.append(sg_sol)
    
    # Select best candidate
    candidates.sort(key=calculate_cost)
    best_cand = candidates[0]
    best_cost = calculate_cost(best_cand)
    
    # Phase 3: Local search on the best candidate
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    ls_time = min(remaining - 10.0, 35.0)
    if ls_time < 2.0:
        ls_time = 2.0
        
    best_assignment, best_cost = local_search(best_cand, max_time=ls_time)
    
    # Phase 4: Final polish if time remains
    if time.time() < deadline - 5.0:
        polish_time = min(deadline - time.time() - 2.0, 10.0)
        if polish_time > 1.0:
            final_assignment, final_cost = local_search(best_assignment, max_time=polish_time)
            if final_cost < best_cost:
                best_assignment = final_assignment
                best_cost = final_cost

    # Ensure validity
    if not is_valid(best_assignment):
        # Fallback to a simple valid solution
        fallback = greedy_assign(np.argsort(-demands))
        if fallback is not None and is_valid(fallback):
            return fallback
        # Last resort: round-robin
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
        return result
    
    return best_assignment

# EVOLVE-BLOCK-END
