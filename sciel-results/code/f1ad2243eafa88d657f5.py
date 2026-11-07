import numpy as np
import time
import math
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost (open + assign)
    subject to capacity constraints.
    """
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initialization: Greedy Largest First
    # Sort customers by demand descending
    order = np.argsort(-demands)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=int)
    is_open = np.zeros(n_facilities, dtype=bool)
    
    for i in order:
        # Find cheapest feasible facility
        # We need to consider: if facility is closed, cost is assign_cost + open_cost
        # if open, cost is assign_cost
        # But we don't know "is_open" for future customers. 
        # Heuristic: Just pick min assign_cost among feasible. 
        # This is the seed strategy. It's okay for init.
        best_f = -1
        min_cost = float('inf')
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                # We prioritize facilities that are already open to avoid open cost?
                # Or just cheapest assignment?
                # Let's try: cost = assign_cost[i][f] + (0 if is_open[f] else open_cost[f])
                # This encourages clustering.
                cost = assign_cost[i][f]
                if not is_open[f]:
                    cost += open_cost[f]
                if cost < min_cost:
                    min_cost = cost
                    best_f = f
        if best_f == -1:
            # No feasible facility found? Should not happen if total cap > total demand
            # Fallback: pick any facility with space?
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # Infeasible instance? Return best effort or raise?
                # Task says "make the whole result invalid" if capacity exceeded.
                # We assume feasible.
                best_f = 0 
                # If we are forced to violate, we are in trouble. 
                # But let's assume the problem is feasible.
        
        assignment[i] = best_f
        load[best_f] += demands[i]
        is_open[best_f] = True
        
    # Calculate initial cost
    def calc_cost(assignment, load, is_open):
        total_assign = 0
        for i in range(n_customers):
            total_assign += assign_cost[i][assignment[i]]
        total_open = 0
        for f in range(n_facilities):
            if is_open[f]:
                total_open += open_cost[f]
        return total_assign + total_open

    current_cost = calc_cost(assignment, load, is_open)
    best_assignment = assignment.copy()
    best_load = load.copy()
    best_is_open = is_open.copy()
    best_cost = current_cost

    # 2. Local Search
    # We will alternate between Move and Swap improvements.
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    def local_search(assignment, load, is_open, current_cost):
        improved = True
        while improved:
            improved = False
            # 1. Single Moves
            for i in range(n_customers):
                f_old = assignment[i]
                if time.time() - start_time > time_limit:
                    return assignment, load, is_open, current_cost
                best_f_new = -1
                best_delta = 0
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    # Check capacity
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    # Calculate Delta
                    delta_assign = assign_cost[i][f_new] - assign_cost[i][f_old]
                    delta_open = 0
                    # If f_new was closed
                    if not is_open[f_new]:
                        delta_open += open_cost[f_new]
                    # If f_old becomes empty
                    if load[f_old] == demands[i]: # Only customer at f_old
                        # And f_new != f_old (already checked)
                        # So f_old becomes empty
                        delta_open -= open_cost[f_old]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1:
                    # Perform move
                    assignment[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    if load[f_old] == 0:
                        is_open[f_old] = False
                    is_open[best_f_new] = True
                    current_cost += best_delta
                    improved = True
            
            # 2. Swaps
            # To keep it fast, we might only check swaps if single moves didn't improve?
            # Or always check. N=100, swaps are O(N^2). 100^2 = 10000.
            # Inner loop is small.
            for i in range(n_customers):
                f_i = assignment[i]
                if time.time() - start_time > time_limit:
                    return assignment, load, is_open, current_cost
                for j in range(i + 1, n_customers):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    # Check capacities for swap
                    # f_i loses i, gains j
                    # f_j loses j, gains i
                    new_load_i = load[f_i] - demands[i] + demands[j]
                    new_load_j = load[f_j] - demands[j] + demands[i]
                    
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    
                    # Calculate Delta
                    # Assignment costs
                    delta_assign = (assign_cost[i][f_j] + assign_cost[j][f_i]) - (assign_cost[i][f_i] + assign_cost[j][f_j])
                    
                    # Open costs
                    # Does f_i become empty?
                    # f_i had load[f_i]. Now has new_load_i.
                    # If load[f_i] == demands[i] (only i was there) and new_load_i > 0 (j moves in), f_i stays open.
                    # If load[f_i] == demands[i] and j does NOT move in? But j DOES move in.
                    # So f_i is never empty if we swap with someone from another facility.
                    # UNLESS f_j == f_i, which we skip.
                    # So open costs DO NOT CHANGE in a swap between distinct facilities.
                    # Wait, what if f_i had ONLY i, and f_j had ONLY j?
                    # After swap, f_i has j, f_j has i. Both still open.
                    # So delta_open = 0.
                    
                    total_delta = delta_assign
                    if total_delta < 0:
                        # Perform swap
                        assignment[i] = f_j
                        assignment[j] = f_i
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        # is_open doesn't change because both facilities still have customers
                        current_cost += total_delta
                        improved = True
                        
            # If we found an improvement, loop again to find more
        return assignment, load, is_open, current_cost

    assignment, load, is_open, current_cost = local_search(assignment, load, is_open, current_cost)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_assignment = assignment.copy()
        best_load = load.copy()
        best_is_open = is_open.copy()

    # 3. Simulated Annealing
    # Only run if we have time
    if time.time() - start_time < time_limit - 5:
        T = 1000.0 # Initial temperature
        cooling = 0.995
        min_T = 1.0
        
        # We need to re-calculate or maintain state
        sa_assignment = best_assignment.copy()
        sa_load = best_load.copy()
        sa_is_open = best_is_open.copy()
        sa_cost = best_cost
        
        # To speed up SA, we can pre-compute some things or just be efficient
        while T > min_T and time.time() - start_time < time_limit - 5:
            # Random Move
            i = random.randint(0, n_customers - 1)
            f_old = sa_assignment[i]
            f_new = random.randint(0, n_facilities - 1)
            
            if f_new == f_old:
                T *= cooling
                continue
            
            # Check capacity
            if sa_load[f_new] + demands[i] > capacities[f_new]:
                T *= cooling
                continue
            
            # Calculate Delta
            delta_assign = assign_cost[i][f_new] - assign_cost[i][f_old]
            delta_open = 0
            if not sa_is_open[f_new]:
                delta_open += open_cost[f_new]
            if sa_load[f_old] == demands[i]:
                delta_open -= open_cost[f_old]
            
            total_delta = delta_assign + delta_open
            
            # Accept?
            if total_delta < 0 or random.random() < math.exp(-total_delta / T):
                # Perform move
                sa_assignment[i] = f_new
                sa_load[f_old] -= demands[i]
                sa_load[f_new] += demands[i]
                if sa_load[f_old] == 0:
                    sa_is_open[f_old] = False
                sa_is_open[f_new] = True
                sa_cost += total_delta
                
                if sa_cost < best_cost:
                    best_cost = sa_cost
                    best_assignment = sa_assignment.copy()
                    best_load = sa_load.copy()
                    best_is_open = sa_is_open.copy()
            
            T *= cooling

    # 4. Final Local Search on the best found
    if time.time() - start_time < time_limit - 2:
        final_assignment, final_load, final_is_open, final_cost = local_search(
            best_assignment.copy(), best_load.copy(), best_is_open.copy(), best_cost
        )
        if final_cost < best_cost:
            best_cost = final_cost
            best_assignment = final_assignment

    return best_assignment

