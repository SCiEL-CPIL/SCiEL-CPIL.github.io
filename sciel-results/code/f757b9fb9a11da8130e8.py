"""Capacitated facility location solver with Lagrangian Relaxation and Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _lagrangian_relax(demands, capacities, open_cost, assign_cost, max_iters=100):
    """
    Solve CFLP using Lagrangian Relaxation of capacity constraints.
    Returns a feasible assignment.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initialize multipliers
    lambda_ = np.zeros(n_facilities)
    best_assignment = None
    best_cost = float('inf')
    
    # Initial step size
    alpha = 1.0
    
    for t in range(max_iters):
        # 1. Solve the Lagrangian subproblem
        # Effective cost for customer i at facility f: assign_cost[i, f] + lambda_[f] * demands[i]
        # Note: We do not explicitly add open_cost here in the subproblem minimization 
        # because the dual function maximization handles it, but for finding a 
        # good primal solution, we should consider it.
        # A common heuristic: minimize assign_cost[i, f] + lambda_[f] * demands[i]
        # Then evaluate the true cost.
        
        # Vectorized calculation of effective costs
        # shape: (n_customers, n_facilities)
        penalty = np.outer(np.ones(n_customers), lambda_ * demands)
        eff_cost = assign_cost + penalty
        
        # Assign each customer to the facility with minimum effective cost
        assignment = np.argmin(eff_cost, axis=1)
        
        # 2. Calculate loads and check capacity
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
            
        # 3. Repair: Make the assignment feasible
        # If a facility is overloaded, move customers out
        feasible = True
        for f in range(n_facilities):
            if load[f] > capacities[f]:
                feasible = False
                # Find customers assigned to f
                customers_in_f = np.where(assignment == f)[0]
                # Sort by demand descending? Or by cost regret?
                # Let's move the customer with the largest demand first to free up space quickly
                # Or the one with the highest cost to move?
                # Let's try moving the one with the largest demand
                for i in customers_in_f:
                    if load[f] <= capacities[f]:
                        break
                    # Find best alternative facility for i
                    # Must have capacity
                    # Minimize assign_cost[i, alt]
                    # Also consider open cost if alt was empty
                    best_alt = -1
                    best_alt_cost = float('inf')
                    
                    for alt_f in range(n_facilities):
                        if alt_f == f:
                            continue
                        if load[alt_f] + demands[i] <= capacities[alt_f]:
                            # Calculate cost of moving i to alt_f
                            # Current cost contribution of i to f: assign_cost[i, f]
                            # New cost contribution: assign_cost[i, alt_f]
                            # Open cost change:
                            # f loses i. If f becomes empty, save open_cost[f]
                            # alt_f gains i. If alt_f was empty, pay open_cost[alt_f]
                            
                            new_load_f = load[f] - demands[i]
                            new_load_alt = load[alt_f] + demands[i]
                            
                            open_cost_change = 0
                            if load[f] == demands[i]: # f becomes empty
                                open_cost_change -= open_cost[f]
                            if load[alt_f] == 0: # alt_f becomes non-empty
                                open_cost_change += open_cost[alt_f]
                                
                            total_cost_change = (assign_cost[i, alt_f] - assign_cost[i, f]) + open_cost_change
                            
                            if total_cost_change < best_alt_cost:
                                best_alt_cost = total_cost_change
                                best_alt = alt_f
                                
                    if best_alt >= 0:
                        # Perform move
                        load[f] -= demands[i]
                        load[best_alt] += demands[i]
                        assignment[i] = best_alt
                    else:
                        # No feasible move? This shouldn't happen if total capacity >= total demand
                        # If it does, we have a problem. Assign to facility with most remaining capacity
                        # just to ensure validity, even if cost is high
                        remaining = capacities - load
                        best_alt = np.argmax(remaining)
                        load[f] -= demands[i]
                        load[best_alt] += demands[i]
                        assignment[i] = best_alt
                        
        if not feasible:
            # If we couldn't repair (shouldn't happen), break
            break
            
        # 4. Calculate true cost of the feasible assignment
        current_cost = 0.0
        used_facilities = set(assignment.tolist())
        for f in used_facilities:
            current_cost += open_cost[f]
        for i in range(n_customers):
            current_cost += assign_cost[i, assignment[i]]
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()
            
        # 5. Update multipliers (Subgradient Method)
        # Subgradient is the capacity violation
        violations = load - capacities
        # Only update for overloaded facilities? Or all?
        # Standard: update all, but lambda stays non-negative
        # Step size alpha_t = alpha * (1 - (current_cost / UB)) / (||g||^2 + eps)
        # We don't have a tight UB, so use a decaying alpha
        alpha = 0.5 * alpha # Decay
        norm_sq = np.sum(violations ** 2)
        if norm_sq > 1e-9:
            step = alpha / np.sqrt(norm_sq)
            lambda_ = np.maximum(0, lambda_ + step * violations)
        else:
            # No violation, we are feasible and at a local optimum of the subproblem?
            # Stop if no violations
            break
            
    if best_assignment is None:
        # Fallback to simple greedy if something went wrong
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
        
    return best_assignment


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Simple greedy fallback."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    return result


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    open_cost_total = 0.0
    assign_cost_total = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_cost_total += open_cost[f]
    
    for i in range(n_customers):
        assign_cost_total += assign_cost[i, result[i]]
    
    return open_cost_total + assign_cost_total


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000):
    """Local search: try moving single customers to other facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        for i in range(n_customers):
            current_f = best_result[i]
            current_load_f = load[current_f]
            
            best_alt_f = -1
            best_delta = 0
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                
                delta = -assign_cost[i, current_f] + assign_cost[i, f]
                
                if current_load_f == demands[i]:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_alt_f = f
            
            if best_alt_f >= 0:
                load[current_f] -= demands[i]
                load[best_alt_f] += demands[i]
                best_result[i] = best_alt_f
                best_cost += best_delta
                improved = True
    
    return best_result


def _pair_swap_search(result, demands, capacities, open_cost, assign_cost, max_iter=2000):
    """Try swapping pairs of customers between facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        for i in range(n_customers):
            fi = best_result[i]
            for j in range(i + 1, n_customers):
                fj = best_result[j]
                if fi == fj:
                    continue
                
                if load[fi] - demands[i] + demands[j] > capacities[fi] + 1e-9:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj] + 1e-9:
                    continue
                
                delta = -assign_cost[i, fi] - assign_cost[j, fj] + assign_cost[i, fj] + assign_cost[j, fi]
                
                new_load_fi = load[fi] - demands[i] + demands[j]
                new_load_fj = load[fj] - demands[j] + demands[i]
                
                current_open = (1 if load[fi] > 0 else 0) * open_cost[fi] + (1 if load[fj] > 0 else 0) * open_cost[fj]
                new_open = (1 if new_load_fi > 0 else 0) * open_cost[fi] + (1 if new_load_fj > 0 else 0) * open_cost[fj]
                
                delta += new_open - current_open
                
                if delta < -1e-9:
                    load[fi] -= demands[i]
                    load[fj] -= demands[j]
                    load[fi] += demands[j]
                    load[fj] += demands[i]
                    best_result[i] = fj
                    best_result[j] = fi
                    best_cost += delta
                    improved = True
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: Lagrangian Relaxation for initial solution
    result = _lagrangian_relax(demands, capacities, open_cost, assign_cost, max_iters=50)
    
    # Phase 2: Local search
    if time.time() < deadline - 30:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=10000)
    
    # Phase 3: Pair swap search
    if time.time() < deadline - 20:
        result = _pair_swap_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Phase 4: More local search
    if time.time() < deadline - 10:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Ensure validity
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
            break
    
    return result
# EVOLVE-BLOCK-END
