"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    
    Strategy:
    1. Greedy initial assignment (largest demand first, cheapest feasible facility)
    2. Local search: try reassigning customers to different facilities if it reduces cost
       while respecting capacities.
    3. Use a time budget to ensure we finish within the deadline.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # --- Initial greedy solution ---
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by descending demand
    order = np.argsort(-demands)
    for i in order:
        # Find cheapest feasible facility
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                c = assign_cost[i, f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        if best_f == -1:
            # Fallback: assign to facility with most remaining capacity
            remaining = capacities - load
            best_f = np.argmax(remaining)
            if load[best_f] + demands[i] > capacities[best_f]:
                # This shouldn't happen if total capacity >= total demand
                # But just in case, assign to any facility
                best_f = 0
        load[best_f] += demands[i]
        result[i] = best_f
    
    # --- Compute initial objective ---
    def compute_objective(assignment, load):
        open_cost_total = 0.0
        assign_cost_total = 0.0
        used = set()
        for c in range(n_cust):
            f = assignment[c]
            assign_cost_total += assign_cost[c, f]
            used.add(f)
        for f in used:
            open_cost_total += open_cost[f]
        return open_cost_total + assign_cost_total
    
    current_obj = compute_objective(result, load)
    
    # --- Local search ---
    # For each customer, try moving them to a different facility if it improves the objective
    # and doesn't violate capacity.
    
    # Precompute: for each customer, the sorted list of facilities by assignment cost
    sorted_fac = np.argsort(assign_cost, axis=1)  # shape (n_cust, n_fac)
    
    # Track which facilities are open
    is_open = np.zeros(n_fac, dtype=bool)
    for f in range(n_fac):
        if load[f] > 0:
            is_open[f] = True
    
    # Count of customers per facility
    cust_count = np.zeros(n_fac, dtype=int)
    for c in range(n_cust):
        cust_count[result[c]] += 1
    
    best_obj = current_obj
    best_result = result.copy()
    best_load = load.copy()
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try reassigning each customer
        for c in range(n_cust):
            if time.time() > deadline:
                break
            current_f = result[c]
            # Try other facilities in order of increasing assignment cost
            for f in sorted_fac[c]:
                if f == current_f:
                    continue
                # Check if moving c to f improves objective
                # Old: if current_f becomes empty, we save open_cost[current_f]
                # New: if f was not open, we add open_cost[f]
                old_open_change = 0.0
                new_open_change = 0.0
                if cust_count[current_f] == 1:
                    old_open_change = -open_cost[current_f]  # we close it
                if cust_count[f] == 0:
                    new_open_change = open_cost[f]  # we open it
                
                delta_assign = assign_cost[c, f] - assign_cost[c, current_f]
                delta_obj = delta_assign + old_open_change + new_open_change
                
                # Check capacity
                if load[f] + demands[c] <= capacities[f]:
                    if delta_obj < -1e-9:
                        # Make the move
                        load[current_f] -= demands[c]
                        load[f] += demands[c]
                        cust_count[current_f] -= 1
                        cust_count[f] += 1
                        if cust_count[current_f] == 0:
                            is_open[current_f] = False
                        if cust_count[f] > 0:
                            is_open[f] = True
                        result[c] = f
                        current_obj += delta_obj
                        improved = True
                        break  # Move to best improving facility for this customer
    
    # Update best if improved
    if current_obj < best_obj:
        best_obj = current_obj
        best_result = result.copy()
        best_load = load.copy()
    
    return best_result
# EVOLVE-BLOCK-END
