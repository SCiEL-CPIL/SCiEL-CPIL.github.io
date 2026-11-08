"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def calculate_cost(assignment):
        # assignment: array of facility indices
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        # Open cost: sum of open_cost for unique facilities used
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        # Check capacity
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities):
            return False
        # Check indices
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        return True

    def greedy_assign(order):
        """
        Assign customers in given order to cheapest feasible facility.
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        for i in order:
            # Sort facilities by cost for this customer
            fac_costs = assign_cost[i]
            fac_indices = np.argsort(fac_costs)
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
            # If no feasible facility found, we have a problem.
            # In this problem, it's assumed feasible. 
            # If not, we might need to repair, but let's assume standard feasibility.
            if result[i] == -1:
                # Fallback: assign to facility with most remaining capacity (least likely to be full)
                # or just break. For robustness, pick any with capacity.
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
                if result[i] == -1:
                    return None # Infeasible
        return result

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
        for i in range(n_cust):
            load[current_assignment[i]] += demands[i]
            
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
            for i in range(n_cust):
                if time.time() > local_deadline:
                    break
                f_from = current_assignment[i]
                # Try all other facilities
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to]:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Accept move
                        current_assignment[i] = f_to
                        load[f_from] -= demands[i]
                        load[f_to] += demands[i]
                        cust_count[f_from] -= 1
                        cust_count[f_to] += 1
                        current_cost += total_delta
                        improved = True
                        break # Move to next customer

            # 2. Swap moves
            # Iterate over pairs of used facilities
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            for idx1, f1 in enumerate(used_facs):
                for f2 in used_facs[idx1+1:]:
                    if time.time() > local_deadline:
                        break
                    # Get customers in f1 and f2
                    custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                    custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                    
                    for i in custs_f1:
                        for j in custs_f2:
                            # Check capacity after swap
                            if load[f2] - demands[j] + demands[i] > capacities[f2]:
                                continue
                            if load[f1] - demands[i] + demands[j] > capacities[f1]:
                                continue
                            
                            # Cost change
                            delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                            
                            # Open cost change for swap between two non-empty facilities is 0
                            # (Both f1 and f2 remain non-empty because i moves to f2 and j moves to f1)
                            # Wait, if cust_count[f1] == 1, then after moving i out, f1 becomes empty?
                            # No, j moves IN to f1. So f1 will have at least j.
                            # So cust_count[f1] becomes 1 (just j). It was 1 (just i).
                            # So it remains non-empty.
                            # Same for f2.
                            # So delta_open is always 0.
                            
                            if delta_assign < -1e-9:
                                # Accept swap
                                current_assignment[i] = f2
                                current_assignment[j] = f1
                                load[f1] -= demands[i]
                                load[f1] += demands[j]
                                load[f2] -= demands[j]
                                load[f2] += demands[i]
                                # cust_count doesn't change for swaps between non-empty facilities
                                current_cost += delta_assign
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return current_assignment, current_cost

    # 1. Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    # Strategy 2: Smallest demand first
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    # Strategy 3: Random orders
    rng = np.random.default_rng(42)
    for _ in range(10):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest first
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)

    if not candidates:
        # Emergency fallback
        return greedy_assign(np.argsort(-demands))

    # Select best candidate based on initial greedy cost
    best_cand = min(candidates, key=calculate_cost)
    
    # 2. Multi-start Local Search
    # We will run local search on the top few candidates to see if they converge to different local minima
    # or if one is significantly better.
    
    # Sort candidates by cost to prioritize better starting points
    candidates.sort(key=calculate_cost)
    
    best_final_assignment = best_cand
    best_final_cost = calculate_cost(best_cand)
    
    # Run local search on the best few candidates
    # We have ~110s. Let's give each LS run a budget.
    # If we run 5 LS runs, we have ~20s each.
    num_ls_runs = 5
    
    for idx in range(min(num_ls_runs, len(candidates))):
        if time.time() > deadline - 5.0: # Leave 5s for final safety
            break
            
        current_cand = candidates[idx]
        # Time budget for this LS run
        remaining_time = deadline - 5.0 - time.time()
        # Allocate time proportional to rank? Or just fixed?
        # Fixed is simpler. Let's say 20s max per run, but not exceeding remaining.
        ls_time = min(20.0, remaining_time)
        
        if ls_time <= 0:
            break
            
        ls_result, ls_cost = local_search(current_cand, max_time=ls_time)
        
        if ls_cost < best_final_cost:
            best_final_assignment = ls_result
            best_final_cost = ls_cost
            
        # If the best candidate (idx 0) converged, and the next one is worse, 
        # we might still want to try others if time permits, as they might find a better basin.
        # But usually the best greedy solution leads to the best LS solution.
        # Let's just run a few.

    return best_final_assignment

# EVOLVE-BLOCK-END
