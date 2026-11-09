"""
Capacitated facility location: Lagrangian Relaxation with Subgradient Optimization.

Strategy:
1. Relax capacity constraints with multipliers u_f.
2. Solve the relaxed problem: min sum_i (c_if + u_f * d_i) + sum_f (u_f * cap_f).
   This decouples customers, allowing O(N*M) assignment.
3. Repair capacity violations by moving customers from overloaded facilities to
   underloaded ones, minimizing the increase in the *relaxed* objective (which
   approximates the true objective when u is tuned).
4. Update u_f using subgradient method to improve the lower bound and guide repair.
5. Track the best feasible solution found during the iterations.
"""

import time
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 145s max
    deadline = time.time() + 145.0
    
    # Pre-compute indices for speed
    cust_idx = np.arange(n_cust)
    
    def calc_score(res, load=None):
        """Calculate total cost for a given assignment."""
        if load is None:
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[res[i]] += demands[i]
        
        if np.any(load > capacities):
            return np.inf, load
            
        used = np.where(load > 0)[0]
        open_sum = np.sum(open_cost[used])
        assign_sum = np.sum(assign_cost[cust_idx, res])
        return open_sum + assign_sum, load

    def solve_relaxed(u):
        """
        Solve the relaxed problem for given multipliers u.
        Returns assignment res.
        Cost for customer i to facility f is: assign_cost[i, f] + u[f] * demands[i]
        """
        # Effective cost matrix
        # shape: (n_cust, n_fac)
        # c_eff[i, f] = assign_cost[i, f] + u[f] * demands[i]
        # We can compute this efficiently
        c_eff = assign_cost + np.outer(demands, u)
        
        # Assign each customer to the facility with min effective cost
        res = np.argmin(c_eff, axis=1)
        return res

    def repair(res, load, u):
        """
        Repair capacity violations.
        Moves customers from overloaded facilities to underloaded ones.
        Minimize increase in relaxed cost: (assign_cost[i, new] + u[new]*d[i]) - (assign_cost[i, old] + u[old]*d[i])
        """
        res = res.copy()
        load = load.copy()
        
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline - 1.0:
                return res, load, False
                
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                return res, load, True
                
            f_over = overloaded[0]
            # Find customers in f_over
            custs_in_f = np.where(res == f_over)[0]
            if len(custs_in_f) == 0:
                break
                
            best_cust = -1
            best_new_fac = -1
            min_penalty = np.inf
            
            # For each customer in the overloaded facility, find the best move
            for i in custs_in_f:
                # Find feasible new facilities
                feasible = np.where(load + demands[i] <= capacities)[0]
                # Exclude current facility
                feasible = feasible[feasible != f_over]
                
                if len(feasible) == 0:
                    continue
                    
                # Calculate penalty for moving i to each feasible facility
                # Penalty = (assign_cost[i, f_new] + u[f_new] * d_i) - (assign_cost[i, f_over] + u[f_over] * d_i)
                # Since d_i and u[f_over] are constant for this i, we minimize:
                # assign_cost[i, f_new] + u[f_new] * d_i
                costs = assign_cost[i, feasible] + u[feasible] * demands[i]
                min_idx = np.argmin(costs)
                f_new = feasible[min_idx]
                penalty = costs[min_idx] - (assign_cost[i, f_over] + u[f_over] * demands[i])
                
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_cust = i
                    best_new_fac = f_new
            
            if best_cust == -1:
                return res, load, False
                
            # Perform move
            res[best_cust] = best_new_fac
            load[f_over] -= demands[best_cust]
            load[best_new_fac] += demands[best_cust]
            
        return res, load, np.all(load <= capacities)

    # Initialization
    # Start with u = 0 (pure assignment cost)
    u = np.zeros(n_fac)
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initial greedy solution for safety
    res_greedy = np.zeros(n_cust, dtype=int)
    load_greedy = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load_greedy + demands[i] <= capacities)[0]
        if len(feasible) > 0:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            res_greedy[i] = f_idx
            load_greedy[f_idx] += demands[i]
        else:
            # Fallback
            remaining = capacities - load_greedy
            f = np.argmax(remaining)
            res_greedy[i] = f
            load_greedy[f] += demands[i]
            
    score_greedy, _ = calc_score(res_greedy)
    if score_greedy < best_score:
        best_score = score_greedy
        best_result = res_greedy.copy()

    # Subgradient Optimization Loop
    # Step size parameter
    step_size = 1.0
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Solve relaxed problem
        res = solve_relaxed(u)
        
        # 2. Calculate load
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
        # 3. Calculate subgradient (violation of capacity)
        # g_f = load_f - cap_f
        subgrad = load - capacities
        
        # 4. Calculate lower bound (for step size adjustment, optional but good practice)
        # LB = sum_i min_f (c_if + u_f d_i) + sum_f u_f cap_f
        # We already have the assignment, so sum_i (c_i, res[i] + u[res[i]] d_i)
        relaxed_obj = np.sum(assign_cost[cust_idx, res] + u[res] * demands) + np.sum(u * capacities)
        
        # 5. Update u
        # u_f = max(0, u_f + step_size * (load_f - cap_f))
        # Adaptive step size: step_size = alpha * (UB - LB) / ||g||^2
        # We need an Upper Bound (UB) from a feasible solution.
        # Use best_score as UB if feasible, else inf.
        ub = best_score
        if np.isinf(ub):
            # If no feasible solution yet, use a large number
            ub = np.sum(assign_cost) + np.sum(open_cost)
            
        norm_sq = np.sum(subgrad ** 2)
        if norm_sq > 0:
            step = step_size * (ub - relaxed_obj) / norm_sq
            if step < 0:
                step = 0
        else:
            step = 0
            
        u = np.maximum(0, u + step * subgrad)
        
        # 6. Repair and check feasibility
        res_rep, load_rep, valid = repair(res, load, u)
        
        if valid:
            score, _ = calc_score(res_rep, load_rep)
            if score < best_score:
                best_score = score
                best_result = res_rep.copy()
                
        # Optional: If relaxed solution is already feasible, it's a strong candidate
        if np.all(load <= capacities):
            score, _ = calc_score(res, load)
            if score < best_score:
                best_score = score
                best_result = res.copy()

    # Final polish: Simple local search on the best result found
    # This helps escape shallow local optima of the repair step
    if best_score < np.inf:
        res_ls = best_result.copy()
        load_ls = np.zeros(n_fac)
        for i in range(n_cust):
            load_ls[res_ls[i]] += demands[i]
            
        # Single move local search
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n_cust):
                if time.time() > deadline - 1.0:
                    break
                f_old = res_ls[i]
                best_delta = np.inf
                best_f_new = -1
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load_ls[f_new] + demands[i] > capacities[f_new]:
                        continue
                        
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    if load_ls[f_old] == demands[i]:
                        delta_open -= open_cost[f_old]
                    if load_ls[f_new] == 0:
                        delta_open += open_cost[f_new]
                        
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1 and best_delta < -1e-9:
                    res_ls[i] = best_f_new
                    load_ls[f_old] -= demands[i]
                    load_ls[best_f_new] += demands[i]
                    improved = True
                    
        if time.time() < deadline - 1.0:
            final_score, _ = calc_score(res_ls)
            if final_score < best_score:
                best_score = final_score
                best_result = res_ls

    return best_result
