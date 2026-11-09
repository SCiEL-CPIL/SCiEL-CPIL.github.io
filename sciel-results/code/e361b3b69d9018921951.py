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

    def solve_lagrangian(max_time=60.0):
        """
        Use Lagrangian Relaxation to find high-quality feasible solutions.
        The Lagrangian function relaxes capacity constraints with multipliers lambda.
        Min sum_i min_f (c_if + lambda_f * d_i) + sum_f lambda_f * C_f
        We update lambda to drive load towards capacity.
        """
        best_sol = None
        best_cost = float('inf')
        
        # Initialize multipliers
        lam = np.zeros(n_fac)
        step_size = 1.0
        decay = 0.99
        
        # Best lower bound for subgradient
        best_lb = -float('inf')
        counter = 0
        
        local_deadline = time.time() + max_time
        
        while time.time() < local_deadline:
            # 1. Solve the relaxed problem
            # Effective cost for customer i at facility f: c_if + lam_f * d_i
            eff_costs = assign_cost + np.outer(demands, lam)
            
            # Each customer picks the facility with minimum effective cost
            assignment = np.argmin(eff_costs, axis=1)
            
            # Calculate load and check feasibility
            loads = np.zeros(n_fac)
            np.add.at(loads, assignment, demands)
            
            is_feasible = np.all(loads <= capacities + 1e-9)
            
            if is_feasible:
                cost = calculate_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = assignment.copy()
            
            # 2. Calculate Lagrangian Dual Value (Lower Bound)
            # L(lam) = sum_i min_f (c_if + lam_f * d_i) + sum_f lam_f * C_f
            min_eff_costs = np.min(eff_costs, axis=1)
            lagrangian_value = np.sum(min_eff_costs) + np.sum(lam * capacities)
            
            if lagrangian_value > best_lb:
                best_lb = lagrangian_value
                counter = 0
            else:
                counter += 1
                # Reduce step size if no improvement in lower bound
                if counter > 50:
                    step_size *= 0.9
                    counter = 0

            # 3. Update Multipliers (Subgradient Method)
            # Subgradient is (Load_f - Capacity_f)
            subgrad = loads - capacities
            
            # Update lambda: lam_new = max(0, lam_old + step_size * subgrad)
            lam = np.maximum(0, lam + step_size * subgrad)
            
            # Decay step size
            step_size *= decay
            
            # Optional: Random restart of multipliers occasionally to escape local dual minima
            if np.random.random() < 0.05 and time.time() < local_deadline - 5.0:
                lam = np.random.random(n_fac) * 10.0

        return best_sol, best_cost

    def local_search(initial_assignment, max_time=20.0):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        local_deadline = time.time() + max_time
        improved = True
        iter_count = 0
        max_iters = 10000
        
        while improved and iter_count < max_iters:
            if time.time() > local_deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
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
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
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
                # Accept best move
                current_assignment[best_i] = best_f_to
                load[best_f_from] -= demands[best_i]
                load[best_f_to] += demands[best_i]
                cust_count[best_f_from] -= 1
                cust_count[best_f_to] += 1
                current_cost += best_delta
                improved = True
                continue # Restart scan

            # 2. Swap moves
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            if time.time() < local_deadline - 0.5:
                best_swap_delta = 0.0
                best_i1 = -1
                best_i2 = -1
                best_f1 = -1
                best_f2 = -1
                
                for idx1, f1 in enumerate(used_facs):
                    for f2 in used_facs[idx1+1:]:
                        if time.time() > local_deadline:
                            break
                        
                        # Get customers in f1 and f2
                        # Optimization: Only consider swaps if it might improve cost
                        # Simple heuristic: try a few random swaps if lists are large
                        custs_f1 = np.where(current_assignment == f1)[0]
                        custs_f2 = np.where(current_assignment == f2)[0]
                        
                        # Limit swap search to top K candidates by potential gain to save time
                        # Potential gain for moving i from f1 to f2: c[i, f2] - c[i, f1]
                        # We want negative delta.
                        # For now, keep exhaustive but small N (100) and F (20) makes it okay.
                        # N^2 * F^2 is 10000 * 400 = 4M ops per full pass. Might be slow.
                        # Let's restrict to checking only a subset if time is tight.
                        
                        for i in custs_f1:
                            for j in custs_f2:
                                # Check capacity after swap
                                if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-9:
                                    continue
                                if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-9:
                                    continue
                                
                                # Cost change
                                delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                                
                                if delta_assign < best_swap_delta:
                                    best_swap_delta = delta_assign
                                    best_i1 = i
                                    best_i2 = j
                                    best_f1 = f1
                                    best_f2 = f2
                        
                        if best_i1 != -1:
                            # Accept best swap
                            current_assignment[best_i1] = best_f2
                            current_assignment[best_i2] = best_f1
                            load[best_f1] -= demands[best_i1]
                            load[best_f1] += demands[best_i2]
                            load[best_f2] -= demands[best_i2]
                            load[best_f2] += demands[best_i1]
                            current_cost += best_swap_delta
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    continue

        return current_assignment, current_cost

    def simulated_annealing(initial_assignment, max_time=10.0):
        """
        Simulated annealing to escape local optima.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1
        
        best_assignment = current_assignment.copy()
        best_cost = current_cost
        
        local_deadline = time.time() + max_time
        T = 50.0  # Initial temperature
        cooling_rate = 0.999
        
        while time.time() < local_deadline and T > 0.01:
            T *= cooling_rate
            
            # Pick a random customer and try to move it
            i = np.random.randint(n_cust)
            f_from = current_assignment[i]
            
            # Pick a random different facility
            candidates = np.array([f for f in range(n_fac) if f != f_from])
            if len(candidates) == 0: continue
            f_to = np.random.choice(candidates)
            
            # Check capacity
            if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                continue
            
            # Calculate cost delta
            delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
            delta_open = 0.0
            if cust_count[f_from] == 1:
                delta_open -= open_cost[f_from]
            if cust_count[f_to] == 0:
                delta_open += open_cost[f_to]
            delta = delta_assign + delta_open
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept the move
                current_assignment[i] = f_to
                load[f_from] -= demands[i]
                load[f_to] += demands[i]
                cust_count[f_from] -= 1
                cust_count[f_to] += 1
                current_cost += delta
                
                # Update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
            
            # Occasionally do a swap
            if np.random.random() < 0.2:
                used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
                if len(used_facs) >= 2:
                    f1, f2 = np.random.choice(used_facs, size=2, replace=False)
                    custs_f1 = np.where(current_assignment == f1)[0]
                    custs_f2 = np.where(current_assignment == f2)[0]
                    
                    if len(custs_f1) > 0 and len(custs_f2) > 0:
                        i1 = np.random.choice(custs_f1)
                        i2 = np.random.choice(custs_f2)
                        
                        # Check capacity
                        if load[f2] - demands[i2] + demands[i1] <= capacities[f2] + 1e-9 and \
                           load[f1] - demands[i1] + demands[i2] <= capacities[f1] + 1e-9:
                            delta = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                                    (assign_cost[i1, f1] + assign_cost[i2, f2])
                            
                            if delta < 0 or np.random.random() < np.exp(-delta / T):
                                current_assignment[i1] = f2
                                current_assignment[i2] = f1
                                load[f1] -= demands[i1]
                                load[f1] += demands[i2]
                                load[f2] -= demands[i2]
                                load[f2] += demands[i1]
                                current_cost += delta
                                
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_assignment = current_assignment.copy()
        
        return best_assignment, best_cost

    # 1. Lagrangian Relaxation Phase
    # This is the structural change: using dual decomposition to find strong initial solutions
    lag_time = min(60.0, deadline - start_time - 30.0) # Leave 30s for LS/SA
    if lag_time > 5.0:
        lag_sol, lag_cost = solve_lagrangian(max_time=lag_time)
    else:
        lag_sol = None
        lag_cost = float('inf')
    
    # Fallback if Lagrangian didn't find a feasible solution (unlikely but possible)
    if lag_sol is None:
        # Greedy fallback
        order = np.argsort(-demands)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        for i in order:
            eff_costs = assign_cost[i]
            fac_indices = np.argsort(eff_costs)
            assigned = False
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                # Fallback: most remaining capacity
                rem_cap = capacities - load
                fac_indices_cap = np.argsort(-rem_cap)
                for f in fac_indices_cap:
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
        if assigned:
            lag_sol = result
            lag_cost = calculate_cost(result)
        else:
            return result # Infeasible anyway

    best_assignment = lag_sol
    best_cost = lag_cost
    
    # 2. Local Search Polish
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    ls_time = min(remaining - 10.0, 40.0)
    if ls_time > 5.0:
        ls_assignment, ls_cost = local_search(best_assignment, max_time=ls_time)
        if ls_cost < best_cost:
            best_assignment = ls_assignment
            best_cost = ls_cost
    
    # 3. Simulated Annealing (if time permits)
    if time.time() < deadline - 10.0:
        sa_time = min(deadline - time.time() - 5.0, 15.0)
        if sa_time > 2.0:
            sa_assignment, sa_cost = simulated_annealing(best_assignment, max_time=sa_time)
            if sa_cost < best_cost:
                best_assignment = sa_assignment
                best_cost = sa_cost
            # Final quick LS if SA improved
            if time.time() < deadline - 2.0:
                final_assignment, final_cost = local_search(best_assignment, max_time=min(5.0, deadline - time.time() - 1.0))
                if final_cost < best_cost:
                    best_assignment = final_assignment
                    best_cost = final_cost

    return best_assignment

# EVOLVE-BLOCK-END
