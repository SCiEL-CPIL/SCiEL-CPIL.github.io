"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment. Returns inf if invalid."""
    n_fac = len(capacities)
    n_cust = len(demands)
    if len(result) != n_cust:
        return float('inf')
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_facilities = np.unique(result)
    cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
    return cost

def _initialize_open_all_then_close(demands, capacities, open_cost, assign_cost):
    """
    Initialization heuristic:
    1. Open all facilities.
    2. Assign each customer to the nearest facility.
    3. Resolve capacity violations by moving customers to next-best feasible facilities.
    4. Iteratively close facilities if the cost to relocate their customers is less than the open cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Step 1: Assign to nearest
    assignment = np.argmin(assign_cost, axis=1)
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[assignment[i]] += demands[i]
        
    # Step 2: Resolve capacity violations
    # We need to move customers from overloaded facilities.
    # To do this efficiently, we can repeatedly find the "best" customer to move.
    # "Best" means the move that causes the smallest increase in assignment cost.
    
    max_iterations = 1000 # Safety valve
    iter_count = 0
    while iter_count < max_iterations:
        # Find any overloaded facility
        overloaded = np.where(load > capacities + 1e-9)[0]
        if len(overloaded) == 0:
            break
            
        # Pick the first overloaded facility
        j = overloaded[0]
        
        # Find customers assigned to j
        customers_in_j = np.where(assignment == j)[0]
        
        best_move_cost = float('inf')
        best_customer = -1
        best_target = -1
        
        for i in customers_in_j:
            # Find feasible target facilities for i
            # Sort facilities by assign_cost[i, f]
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if f == j:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    delta = assign_cost[i, f] - assign_cost[i, j]
                    if delta < best_move_cost:
                        best_move_cost = delta
                        best_customer = i
                        best_target = f
            # If no feasible target found, we might have a problem, 
            # but the problem guarantees a feasible solution exists usually.
            
        if best_customer == -1:
            # If we can't move anyone, we are stuck. 
            # This shouldn't happen if total capacity >= total demand.
            break
            
        # Perform the move
        assignment[best_customer] = best_target
        load[j] -= demands[best_customer]
        load[best_target] += demands[best_customer]
        iter_count += 1
        
    # Step 3: Close facilities if beneficial
    # We iterate until no more facilities can be closed.
    improved = True
    while improved:
        improved = False
        # Get list of open facilities
        open_facs = np.unique(assignment)
        
        # We should try to close the facility with the highest "saving potential"
        # Saving = open_cost[j] - sum(delta_costs_to_relocate_customers)
        
        best_fac_to_close = -1
        best_saving = 0
        
        for j in open_facs:
            # Calculate cost to relocate all customers of j
            customers_in_j = np.where(assignment == j)[0]
            if len(customers_in_j) == 0:
                continue
                
            total_delta = 0.0
            feasible = True
            # We need to simulate the moves to check feasibility and cost
            # Since moves affect loads, we should be careful.
            # However, for an initial heuristic, we can approximate by checking 
            # if each customer can move to its next best *currently available* spot.
            # This is an approximation but often good enough for initialization.
            
            # To be more accurate, we can try to find a valid set of moves.
            # Given small N, we can just try to move them one by one greedily.
            
            temp_assignment = assignment.copy()
            temp_load = load.copy()
            
            # Sort customers by demand descending? Or by smallest delta?
            # Smallest delta first is usually better to keep options open for large ones?
            # Actually, large demands are harder to place. Let's sort by demand descending.
            cust_order = customers_in_j[np.argsort(-demands[customers_in_j])]
            
            for i in cust_order:
                # Find best feasible target for i in temp_load
                fac_order = np.argsort(assign_cost[i])
                found = False
                for f in fac_order:
                    if f == j:
                        continue
                    if temp_load[f] + demands[i] <= capacities[f]:
                        temp_delta = assign_cost[i, f] - assign_cost[i, j]
                        total_delta += temp_delta
                        # Apply to temp
                        temp_load[j] -= demands[i]
                        temp_load[f] += demands[i]
                        temp_assignment[i] = f
                        found = True
                        break
                if not found:
                    feasible = False
                    break
            
            if feasible:
                saving = open_cost[j] - total_delta
                if saving > best_saving:
                    best_saving = saving
                    best_fac_to_close = j
                    # Store the best assignment for this closure
                    # We need to be careful not to modify the main state yet
                    
        if best_fac_to_close != -1:
            # Re-run the relocation for the best facility to get the final assignment
            j = best_fac_to_close
            customers_in_j = np.where(assignment == j)[0]
            cust_order = customers_in_j[np.argsort(-demands[customers_in_j])]
            
            for i in cust_order:
                fac_order = np.argsort(assign_cost[i])
                for f in fac_order:
                    if f == j:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        assignment[i] = f
                        load[j] -= demands[i]
                        load[f] += demands[i]
                        break
            
            improved = True
            
    return assignment

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """
    Local search with single moves and pair swaps.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    current = result.astype(int).copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(current):
        load[f] += demands[i]
        
    fac_count = np.zeros(n_fac, dtype=int)
    for f in current:
        fac_count[f] += 1
        
    current_cost = _compute_cost(current, demands, capacities, open_cost, assign_cost)
    start_time = time.time()
    
    no_improve_passes = 0
    max_no_improve = 5
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Single Moves
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            cur_f = current[i]
            d_i = demands[i]
            # Sort other facilities by cost
            fac_order = np.argsort(assign_cost[i])
            for new_f in fac_order:
                if new_f == cur_f:
                    continue
                if load[new_f] + d_i > capacities[new_f]:
                    continue
                
                delta = 0.0
                delta += assign_cost[i, new_f] - assign_cost[i, cur_f]
                
                new_cur_count = fac_count[cur_f] - 1
                new_new_count = fac_count[new_f] + 1
                
                if new_cur_count == 0:
                    delta -= open_cost[cur_f]
                if new_new_count == 1:
                    delta += open_cost[new_f]
                    
                if delta < -1e-9:
                    # Apply move
                    load[cur_f] -= d_i
                    load[new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[new_f] += 1
                    current[i] = new_f
                    current_cost += delta
                    improved = True
                    break # Restart inner loop for this customer (First Improvement)
                    
        # Pair Swaps
        if time.time() - start_time < time_limit:
            for i in range(n_cust):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() - start_time > time_limit:
                        break
                    fi = current[i]
                    fj = current[j]
                    if fi == fj:
                        continue
                    di = demands[i]
                    dj = demands[j]
                    
                    if load[fi] - di + dj > capacities[fi]:
                        continue
                    if load[fj] - dj + di > capacities[fj]:
                        continue
                        
                    delta = 0.0
                    delta += assign_cost[i, fj] + assign_cost[j, fi] - assign_cost[i, fi] - assign_cost[j, fj]
                    
                    # Open cost doesn't change for swaps between different facilities
                    # unless one facility becomes empty? 
                    # Wait, fi loses i gains j. Count stays same.
                    # fj loses j gains i. Count stays same.
                    # So open cost delta is 0.
                    
                    if delta < -1e-9:
                        load[fi] = load[fi] - di + dj
                        load[fj] = load[fj] - dj + di
                        current[i] = fj
                        current[j] = fi
                        current_cost += delta
                        improved = True
                        # Don't break outer loops immediately, continue to find more improvements?
                        # Or break to restart? 
                        # For speed, First Improvement is better.
                        # But we are in nested loops. Let's just mark improved and continue scanning.
                        # Actually, breaking and restarting is safer for correctness of state, 
                        # but slower. Given small N, continuing is fine.
                        
        if not improved:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                break
        else:
            no_improve_passes = 0
            
    return current, current_cost

def _simulated_annealing(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """
    Simulated Annealing to escape local optima.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    current = result.astype(int).copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(current):
        load[f] += demands[i]
        
    fac_count = np.zeros(n_fac, dtype=int)
    for f in current:
        fac_count[f] += 1
        
    current_cost = _compute_cost(current, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = current.copy()
    
    start_time = time.time()
    T = 100.0 # Initial temperature
    cooling = 0.995 # Cooling rate
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        move_type = random.random()
        if move_type < 0.7:
            # Single move
            i = random.randint(0, n_cust - 1)
            cur_f = current[i]
            # Pick a random feasible target
            feasible_facs = []
            for f in range(n_fac):
                if f != cur_f and load[f] + demands[i] <= capacities[f]:
                    feasible_facs.append(f)
            if not feasible_facs:
                continue
            new_f = random.choice(feasible_facs)
            
            # Calculate delta
            delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
            
            new_cur_count = fac_count[cur_f] - 1
            new_new_count = fac_count[new_f] + 1
            
            if new_cur_count == 0:
                delta -= open_cost[cur_f]
            if new_new_count == 1:
                delta += open_cost[new_f]
                
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply
                load[cur_f] -= demands[i]
                load[new_f] += demands[i]
                fac_count[cur_f] -= 1
                fac_count[new_f] += 1
                current[i] = new_f
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current.copy()
                    
        else:
            # Pair swap
            i = random.randint(0, n_cust - 1)
            j = random.randint(0, n_cust - 1)
            if i == j:
                continue
            fi = current[i]
            fj = current[j]
            if fi == fj:
                continue
                
            di = demands[i]
            dj = demands[j]
            
            if load[fi] - di + dj > capacities[fi] or load[fj] - dj + di > capacities[fj]:
                continue
                
            delta = assign_cost[i, fj] + assign_cost[j, fi] - assign_cost[i, fi] - assign_cost[j, fj]
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply
                load[fi] = load[fi] - di + dj
                load[fj] = load[fj] - dj + di
                current[i] = fj
                current[j] = fi
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current.copy()
                    
        T *= cooling
        if T < 1e-3:
            break
            
    return best_result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Strong Initialization
    result = _initialize_open_all_then_close(demands, capacities, open_cost, assign_cost)
    
    # 2. Local Search
    remaining = deadline - time.time()
    if remaining > 5:
        result, _ = _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=min(remaining - 5, 10.0))
        
    # 3. Simulated Annealing
    remaining = deadline - time.time()
    if remaining > 5:
        result = _simulated_annealing(result, demands, capacities, open_cost, assign_cost, time_limit=min(remaining - 2, 20.0))
        
    # Final validation
    final_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if final_cost == float('inf'):
        # Fallback
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            else:
                result[i] = 0
                
    return result
# EVOLVE-BLOCK-END
