# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _objective(demands, capacities, open_cost, assign_cost, assignment):
    """Compute total cost: open costs for used facilities + assignment costs."""
    n_fac = len(capacities)
    used = np.zeros(n_fac, dtype=bool)
    for i, f in enumerate(assignment):
        if f >= 0 and f < n_fac:
            used[f] = True
    open_total = open_cost[used].sum()
    assign_total = assign_cost[np.arange(len(assignment)), assignment].sum()
    return open_total + assign_total

def _is_valid(demands, capacities, assignment):
    """Check if assignment is valid: all assigned, within capacity."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(assignment) != n_cust:
        return False
    if np.any(assignment < 0) or np.any(assignment >= n_fac):
        return False
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(assignment):
        load[f] += demands[i]
    return np.all(load <= capacities)

def _greedy_seed(demands, capacities, open_cost, assign_cost):
    """Greedy assignment: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    
    # Sort by demand descending, then by total assignment cost ascending
    total_cost = assign_cost.sum(axis=1)
    order = np.lexsort((total_cost, -demands))
    
    for i in order:
        # Try facilities in order of assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        # If no feasible facility found, leave as -1 (shouldn't happen with valid instances)
    
    return result

def _local_search(demands, capacities, open_cost, assign_cost, assignment, time_limit=10.0):
    """Local search: try moving customers to other facilities, and swapping pairs."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    assignment = assignment.copy()
    load = np.zeros(n_fac, dtype=float)
    for i, f in enumerate(assignment):
        load[f] += demands[i]
    
    current_cost = _objective(demands, capacities, open_cost, assign_cost, assignment)
    best_cost = current_cost
    best_assignment = assignment.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() - start_time >= time_limit:
                break
            old_f = assignment[i]
            if old_f < 0:
                continue
            
            # Try all other facilities
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] <= capacities[new_f]:
                    # Compute delta
                    old_used = (load[old_f] - demands[i] > 0) or (any(assignment[j] == old_f for j in range(n_cust) if j != i))
                    new_used = (load[new_f] > 0) or (new_f == old_f)
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    if not old_used and old_f != new_f:
                        delta_open -= open_cost[old_f]
                    if load[new_f] == 0 and new_f != old_f:
                        delta_open += open_cost[new_f]
                    
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply move
                        load[old_f] -= demands[i]
                        load[new_f] += demands[i]
                        assignment[i] = new_f
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost - 1e-9:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
        
        # Try swapping pairs of customers
        if time.time() - start_time >= time_limit:
            break
        
        for i in range(n_cust):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i+1, n_cust):
                if time.time() - start_time >= time_limit:
                    break
                fi = assignment[i]
                fj = assignment[j]
                if fi == fj or fi < 0 or fj < 0:
                    continue
                
                # Check if swap is feasible
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    
                    delta_assign = (assign_cost[i, fj] - assign_cost[i, fi]) + \
                                   (assign_cost[j, fi] - assign_cost[j, fj])
                    
                    # Open cost change: facilities fi and fj remain used (since both had at least these customers)
                    # But if after swap one facility becomes empty, we save its open cost
                    delta_open = 0.0
                    # Check if fi becomes empty after removing i and adding j
                    # Actually, fi still has j's demand added, so it's not empty unless j was the only one
                    # This is complex; for now, assume no open cost change from swaps (conservative)
                    
                    delta = delta_assign + delta_open
                    if delta < -1e-9:
                        # Apply swap
                        load[fi] = load[fi] - demands[i] + demands[j]
                        load[fj] = load[fj] - demands[j] + demands[i]
                        assignment[i] = fj
                        assignment[j] = fi
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost - 1e-9:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
        
        if not improved:
            no_improve_count += 100
            if no_improve_count > max_no_improve:
                break
    
    return best_assignment, best_cost

def _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, time_limit=5.0):
    """Lagrangian relaxation with subgradient optimization to guide facility selection."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    best_cost = np.inf
    best_assignment = None
    
    # Initialize multipliers
    mu = np.zeros(n_fac)
    
    for iteration in range(200):
        if time.time() - start_time >= time_limit:
            break
        
        # Solve relaxed problem: minimize sum_i sum_f (assign_cost[i,f] + mu[f]) x[i,f] + sum_f open_cost[f] y[f]
        # where y[f] = 1 if facility f is open
        # For each customer, assign to facility with min (assign_cost[i,f] + mu[f])
        relaxed_cost = assign_cost + mu[np.newaxis, :]
        
        # For each facility, decide if open based on whether any customer wants it
        # Simplified: open facility if it's the best for at least one customer
        best_fac = np.argmin(relaxed_cost, axis=1)
        
        # Count how many customers want each facility
        counts = np.bincount(best_fac, minlength=n_fac)
        open_fac = counts > 0
        
        # Compute objective
        obj = 0.0
        for i in range(n_cust):
            f = best_fac[i]
            obj += assign_cost[i, f]
        for f in range(n_fac):
            if open_fac[f]:
                obj += open_cost[f]
        
        # Update multipliers using subgradient
        # Subgradient for mu[f] is (load[f] - capacity[f])
        load = np.zeros(n_fac, dtype=float)
        for i in range(n_cust):
            f = best_fac[i]
            load[f] += demands[i]
        
        grad = load - capacities
        
        # Step size
        alpha = 2.0 / (iteration + 1)
        mu = np.maximum(0, mu + alpha * grad)
        
        # Check if this solution is feasible
        if np.all(load <= capacities):
            assignment = best_fac.astype(int)
            cost = _objective(demands, capacities, open_cost, assign_cost, assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
    
    return best_assignment, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    """Assign each customer to a facility, respecting capacity constraints."""
    n_cust = len(demands)
    n_fac = len(capacities)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Phase 1: Greedy seed
    assignment = _greedy_seed(demands, capacities, open_cost, assign_cost)
    
    if time.time() - start_time > deadline:
        return assignment
    
    # Phase 2: Lagrangian relaxation for guidance
    lagr_assignment, lagr_cost = _lagrangian_relaxation(demands, capacities, open_cost, assign_cost, time_limit=5.0)
    
    if time.time() - start_time > deadline:
        if lagr_assignment is not None and _is_valid(demands, capacities, lagr_assignment):
            return lagr_assignment
        return assignment
    
    # Choose better starting point
    greedy_cost = _objective(demands, capacities, open_cost, assign_cost, assignment)
    if lagr_assignment is not None and _is_valid(demands, capacities, lagr_assignment):
        if lagr_cost < greedy_cost:
            assignment = lagr_assignment
            current_cost = lagr_cost
        else:
            current_cost = greedy_cost
    else:
        current_cost = greedy_cost
    
    # Phase 3: Local search with remaining time
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1.0:
        assignment, current_cost = _local_search(demands, capacities, open_cost, assign_cost, assignment, time_limit=remaining_time - 1.0)
    
    return assignment
# EVOLVE-BLOCK-END
