"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return float('inf')
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order, dual_prices=None):
        """Greedy assignment given customer order and optional dual prices."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if dual_prices is not None:
            adjusted_cost = assign_cost + dual_prices[np.newaxis, :]
            adj_sorted = np.argsort(adjusted_cost, axis=1)
        else:
            adj_sorted = np.argsort(assign_cost, axis=1)
            
        for i in order:
            d_i = demands[i]
            for f in adj_sorted[i]:
                if load[f] + d_i <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += d_i
                    break
            if result[i] == -1:
                # Fallback: find facility with most remaining capacity
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

    def local_search(initial_assignment, max_iters=200):
        """Local search with relocate moves."""
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # Relocate moves
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                f_from = current_assignment[i]
                for f_to in sorted_fac_by_cost[i]:
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
                    
                    if total_delta < -1e-9:
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break

        return current_assignment, current_cost

    def lagrangian_subgradient(max_iters=100):
        """
        Lagrangian relaxation with subgradient optimization.
        Relaxes capacity constraints with dual prices lambda[f].
        The Lagrangian subproblem becomes: for each customer, choose facility
        minimizing assign_cost[c,f] + lambda[f], subject to no capacity constraint.
        Then use the resulting "relaxed" assignment to guide a feasible greedy.
        """
        # Initial dual prices: 0
        lam = np.zeros(n_fac)
        best_assignment = None
        best_cost = float('inf')
        
        # Subgradient parameters
        step_size_init = 100.0
        step_decay = 0.95
        step_size = step_size_init
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve Lagrangian subproblem: each customer picks argmin_f (assign_cost[c,f] + lam[f])
            adjusted_cost = assign_cost + lam[np.newaxis, :]
            relaxed_assign = np.argmin(adjusted_cost, axis=1)
            
            # Compute loads under relaxed assignment
            loads = np.zeros(n_fac)
            for c in range(n_cust):
                loads[relaxed_assign[c]] += demands[c]
            
            # Subgradient: loads - capacities (positive means over capacity)
            subgrad = loads - capacities
            
            # Compute Lagrangian bound (lower bound on optimal)
            # L(lam) = sum_c min_f (assign_cost[c,f] + lam[f]) + sum_f lam[f] * capacities[f]
            # Wait, that's not quite right. The Lagrangian is:
            # L(lam) = sum_c min_f (assign_cost[c,f] + lam[f]) - sum_f lam[f] * capacities[f]
            # Actually for <= constraints with lam >= 0:
            # L(lam) = sum_c min_f (assign_cost[c,f] + lam[f]) + sum_f lam[f] * (0 - capacities[f])
            # Hmm, let me think more carefully.
            # Original: min sum c_f x_cf + open_cost s.t. sum_c d_c x_cf <= cap_f y_f, x_cf <= y_f, sum_f x_cf = 1
            # Relax capacity: sum_c d_c x_cf - cap_f y_f <= 0 with multiplier lam_f >= 0
            # L = sum c_f x_cf + open_cost y_f + sum_f lam_f (sum_c d_c x_cf - cap_f y_f)
            #   = sum_c sum_f (c_f + lam_f d_c) x_cf + sum_f (open_cost_f - lam_f cap_f) y_f
            # For fixed x, optimal y: y_f = 1 if any x_cf=1, else 0 (but also need x_cf <= y_f)
            # This is getting complex. Let me use a simpler approach:
            # Just use the subgradient to update lam, and use the relaxed solution to guide greedy.
            
            # Update dual prices (subgradient method)
            # lam_f = max(0, lam_f + step * subgrad_f)
            lam = np.maximum(0, lam + step_size * subgrad)
            
            # Decay step size
            step_size *= step_decay
            
            # Use dual prices to guide a feasible greedy assignment
            # Order customers by demand (largest first) to help feasibility
            order = np.argsort(-demands)
            feasible_assign = greedy_assign(order, dual_prices=lam)
            
            if feasible_assign is not None and is_valid(feasible_assign):
                cost = calculate_cost(feasible_assign)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = feasible_assign.copy()
            
            # Also try without dual prices (pure cost-based greedy)
            order2 = np.argsort(-demands)
            feasible_assign2 = greedy_assign(order2, dual_prices=None)
            if feasible_assign2 is not None and is_valid(feasible_assign2):
                cost2 = calculate_cost(feasible_assign2)
                if cost2 < best_cost:
                    best_cost = cost2
                    best_assignment = feasible_assign2.copy()
        
        return best_assignment, best_cost

    # Phase 1: Lagrangian relaxation with subgradient
    lag_assign, lag_cost = lagrangian_subgradient(max_iters=80)
    
    # Phase 2: Generate additional diverse initial solutions
    candidates = []
    if lag_assign is not None:
        candidates.append(lag_assign)
    
    # Pure greedy with different orderings
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    # Cost-based ordering
    order3 = np.argsort(assign_cost.min(axis=1))
    cand3 = greedy_assign(order3)
    if cand3 is not None and is_valid(cand3):
        candidates.append(cand3)
    
    # Random orderings
    rng = np.random.default_rng(42)
    for _ in range(3):
        order_rand = rng.permutation(n_cust)
        cand_rand = greedy_assign(order_rand)
        if cand_rand is not None and is_valid(cand_rand):
            candidates.append(cand_rand)
    
    if not candidates:
        # Last resort: assign all to facility 0 (may be infeasible, but must return something)
        return np.zeros(n_cust, dtype=int)

    # Phase 3: Local search on best candidate
    best_cand = min(candidates, key=calculate_cost)
    best_assignment, _ = local_search(best_cand, max_iters=150)
    
    # Verify and return
    if is_valid(best_assignment):
        return best_assignment
    
    # Fallback
    return best_cand

# EVOLVE-BLOCK-END
