"""Capacitated facility location: Lagrangian Relaxation solver with subgradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def repair(res, load):
        """
        Repair infeasible solution by moving customers from overloaded facilities
        to the cheapest feasible alternatives.
        """
        res = res.copy()
        load = load.copy()
        
        # Identify overloaded facilities
        overloaded = load > capacities + 1e-9
        
        # Process overloaded facilities
        for f in np.where(overloaded)[0]:
            excess = load[f] - capacities[f]
            if excess <= 0:
                continue
            
            # Get customers assigned to this facility, sorted by demand (smallest first to be easier to move)
            custs = np.where(res == f)[0]
            if len(custs) == 0:
                continue
                
            # Sort by demand ascending
            custs = custs[np.argsort(demands[custs])]
            
            # Move customers until feasible
            for i in custs:
                if excess <= 1e-9:
                    break
                    
                d_i = demands[i]
                
                # Find best feasible alternative
                feasible = (load + d_i <= capacities + 1e-9)
                feasible[f] = False
                
                if np.any(feasible):
                    feasible_indices = np.where(feasible)[0]
                    costs = assign_cost[i, feasible_indices]
                    best_idx = np.argmin(costs)
                    best_f = feasible_indices[best_idx]
                    
                    # Move customer
                    load[f] -= d_i
                    load[best_f] += d_i
                    res[i] = best_f
                    excess -= d_i
                else:
                    # If no feasible alternative, we have a problem.
                    # This shouldn't happen if total capacity >= total demand.
                    # If it does, we just leave it, but the solution will be invalid.
                    # To be safe, we can try to swap, but for now, let's assume it's feasible.
                    pass
        
        return res, load

    def local_search(res, load, max_passes=50):
        """
        Local search with relocate moves.
        """
        improved = True
        it = 0
        while improved and it < max_passes:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            
            for i in np.random.permutation(n_cust):
                if time.time() > deadline:
                    return res, load
                
                current_f = res[i]
                d_i = demands[i]
                
                base_delta = assign_cost[i, :] - assign_cost[i, current_f]
                
                feasible = (load + d_i <= capacities + 1e-9)
                feasible[current_f] = False
                
                if not np.any(feasible):
                    continue
                
                deltas = base_delta.copy()
                newly_opened = feasible & (load == 0)
                deltas[newly_opened] += open_cost[newly_opened]
                
                if load[current_f] == d_i:
                    deltas[feasible] -= open_cost[current_f]
                
                feasible_indices = np.where(feasible)[0]
                if len(feasible_indices) == 0:
                    continue
                
                min_idx = np.argmin(deltas[feasible_indices])
                best_f = feasible_indices[min_idx]
                best_delta = deltas[best_f]
                
                if best_delta < -1e-9:
                    load[current_f] -= d_i
                    load[best_f] += d_i
                    res[i] = best_f
                    improved = True
        
        return res, load

    # Lagrangian Relaxation
    best_res = None
    best_score = np.inf
    
    # Initialize multipliers
    mu = np.zeros(n_fac)
    
    # Subgradient optimization
    best_dual = -np.inf
    best_mu = mu.copy()
    
    for iteration in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Solve relaxed problem
        # Minimize sum(assign_cost[i, f] + mu[f]*demands[i]) subject to res[i]=f
        adjusted_cost = assign_cost + np.outer(demands, mu)
        res = np.argmin(adjusted_cost, axis=1)
        
        # Compute load
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # Compute dual function value (lower bound)
        # Dual function g(mu) = sum_i min_f (assign_cost[i,f] + mu[f]*demands[i]) + sum_f mu[f]*capacities[f]
        # Note: The standard Lagrangian for capacity constraints C_j >= sum_i x_ij d_i is:
        # L(x, mu) = sum x_ij c_ij + sum mu_j (cap_j - sum x_ij d_i)
        # = sum mu_j cap_j + sum x_ij (c_ij - mu_j d_i)
        # So min over x is sum_j mu_j cap_j + sum_i min_j (c_ij - mu_j d_i)
        # Wait, the sign of mu depends on constraint direction.
        # Constraint: sum x_ij d_i <= cap_j.
        # Lagrangian: sum c_ij x_ij + sum mu_j (sum x_ij d_i - cap_j)
        # = -sum mu_j cap_j + sum x_ij (c_ij + mu_j d_i)
        # Min over x: -sum mu_j cap_j + sum_i min_j (c_ij + mu_j d_i)
        # This is a lower bound on the primal (if mu >= 0).
        
        dual_val = -np.sum(mu * capacities) + np.sum(np.min(adjusted_cost, axis=1))
        
        if dual_val > best_dual:
            best_dual = dual_val
            best_mu = mu.copy()
            best_res_relaxed = res.copy()
            best_load_relaxed = load.copy()
            
        # Check if feasible
        if np.all(load <= capacities + 1e-9):
            score = compute_score(res)
            if score < best_score:
                best_score = score
                best_res = res.copy()
                
        # Subgradient update
        # s_j = cap_j - load_j (violation is load_j - cap_j, so subgrad is load_j - cap_j for minimization of L)
        # Actually, for max dual, we want to increase mu_j if load_j > cap_j.
        # Subgradient of dual with respect to mu is (cap_j - load_j).
        # So mu_new = mu + step * (cap - load)
        # But mu must be >= 0.
        
        subgrad = capacities - load
        
        # Step size
        # Standard: alpha_k = alpha_0 / sqrt(k) or alpha_0 / k
        # Let's use alpha_0 = 1.0
        alpha = 1.0 / (1 + iteration * 0.01)
        
        # Project onto mu >= 0
        mu = np.maximum(0, mu + alpha * subgrad)
        
        # Optional: Normalize mu to prevent divergence
        # mu = mu / (np.max(mu) + 1e-9)
        
    # Use the best relaxed solution found (or the last one if no feasible was found)
    if best_res is not None:
        res_final = best_res
    else:
        # Use the best dual solution's relaxed assignment
        res_final = best_res_relaxed
        load_final = np.zeros(n_fac)
        np.add.at(load_final, res_final, demands)
        
        # Repair if necessary
        if np.any(load_final > capacities + 1e-9):
            res_final, load_final = repair(res_final, load_final)
            
    # Final local search polish
    load_final = np.zeros(n_fac)
    np.add.at(load_final, res_final, demands)
    
    # Ensure feasibility before local search
    if np.any(load_final > capacities + 1e-9):
        res_final, load_final = repair(res_final, load_final)
        
    res_final, load_final = local_search(res_final, load_final, max_passes=100)
    
    return res_final.astype(int)

# EVOLVE-BLOCK-END
