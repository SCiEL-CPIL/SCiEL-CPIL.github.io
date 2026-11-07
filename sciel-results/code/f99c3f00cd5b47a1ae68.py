import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    # Precompute
    # For local search, we need fast access to costs
    
    best_solution = None
    best_cost = float('inf')
    
    # Initialize lambda
    lambdas = np.zeros(n_facilities)
    
    # Upper bound for Polyak step size
    # We need a feasible solution first to get a valid upper bound.
    # Let's run a simple greedy first to get a valid upper bound.
    
    def greedy_init(lambdas):
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort indices by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Find feasible facilities
            # cost_eff = assign_cost[i] + lambdas * demands[i]
            cost_eff = assign_cost[i] + lambdas * demands[i]
            # Mask infeasible
            feasible_mask = (load + demands[i]) <= capacities
            if not np.any(feasible_mask):
                # Fallback: pick least overloaded
                # This shouldn't happen if problem is feasible
                f = np.argmin(load)
            else:
                # Set infeasible to inf
                cost_eff_masked = cost_eff.copy()
                cost_eff_masked[~feasible_mask] = np.inf
                f = np.argmin(cost_eff_masked)
            load[f] += demands[i]
            result[i] = f
        return result

    # Initial greedy
    sol = greedy_init(lambdas)
    
    def calculate_cost(sol):
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[sol[i]] += demands[i]
        open_sum = 0.0
        for f in range(n_facilities):
            if load[f] > 0:
                open_sum += open_cost[f]
        assign_sum = np.sum(assign_cost[np.arange(n_customers), sol])
        return open_sum + assign_sum, load

    current_cost, current_load = calculate_cost(sol)
    if current_cost < best_cost:
        best_cost = current_cost
        best_solution = sol.copy()
        
    # Lagrangian Loop
    # U is the upper bound (best feasible cost found)
    U = best_cost
    L_best = -np.inf
    
    # Step size parameters
    # Standard Polyak: alpha = (U - L) / ||g||^2
    # We need to track L (Lagrangian value)
    
    # To speed up, we can vectorize the relaxed solve
    # Cost matrix for relaxed problem: C_relax[i, f] = assign_cost[i, f] + lambdas[f] * demands[i]
    
    while time.time() < deadline:
        # 1. Solve Relaxed Problem
        # C_relax = assign_cost + np.outer(demands, lambdas)
        # f_relax[i] = argmin_f C_relax[i, f]
        
        # We can compute this efficiently
        # assign_cost is (N, F). lambdas is (F,). demands is (N,).
        # term = demands[:, None] * lambdas[None, :]
        # C_relax = assign_cost + term
        # f_relax = np.argmin(C_relax, axis=1)
        
        term = demands[:, None] * lambdas[None, :]
        C_relax = assign_cost + term
        f_relax = np.argmin(C_relax, axis=1)
        
        # Calculate loads from relaxed solution
        load_relax = np.zeros(n_facilities)
        for i in range(n_customers):
            load_relax[f_relax[i]] += demands[i]
            
        # Calculate Lagrangian Value L
        # L = sum(c_{i, f_relax[i]}) + sum_f lambda_f * (load_relax_f - cap_f)
        assign_sum_relax = np.sum(assign_cost[np.arange(n_customers), f_relax])
        penalty_sum = np.sum(lambdas * (load_relax - capacities))
        L_val = assign_sum_relax + penalty_sum
        
        # Update L_best
        if L_val > L_best:
            L_best = L_val
            
        # 2. Update Lambda
        # Subgradient g = load_relax - capacities
        g = load_relax - capacities
        g_norm_sq = np.sum(g**2)
        
        if g_norm_sq < 1e-10:
            break # Converged
            
        # Polyak step size
        # Ensure U > L_val. If U <= L_val, step size is non-positive or zero.
        # In theory, U >= L_val for minimization.
        if U <= L_val:
            alpha = 0.1 # Fallback small step
        else:
            alpha = (U - L_val) / g_norm_sq
            
        # Limit alpha to prevent huge jumps
        alpha = min(alpha, 1.0)
        
        lambdas = np.maximum(0, lambdas + alpha * g)
        
        # 3. Repair to Feasible Solution
        # Use greedy guided by current lambdas
        sol = greedy_init(lambdas)
        
        # 4. Local Search
        # 1-Move
        improved = True
        passes = 0
        while improved and passes < 50: # Limit passes
            improved = False
            passes += 1
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[sol[i]] += demands[i]
                
            for i in range(n_customers):
                f_curr = sol[i]
                # Try moving to other facilities
                # We can vectorize the cost check
                # But capacity check is dynamic.
                # Simple loop is fine for N=100.
                best_f = f_curr
                best_delta = 0
                # Current cost contribution: assign_cost[i, f_curr]
                # Potential change in open cost is complex (if f_curr becomes empty, or new f becomes non-empty).
                # For 1-move, we only consider moves that don't change the set of open facilities?
                # Or we calculate full cost change.
                # Let's just calculate the change in assignment cost and check if open cost changes.
                
                # Optimization: Only consider moves to facilities that are already open or if we are willing to pay open cost.
                # Actually, simpler: Calculate new cost if we move i to f.
                # But doing this for all i, f is O(N*F).
                
                # Let's stick to a simpler heuristic: 
                # Only move if it reduces assignment cost and doesn't violate capacity.
                # We ignore open cost changes in the local search for speed, 
                # relying on the global greedy to handle open costs?
                # No, that's bad. Moving a customer to a new facility adds open_cost.
                
                # Let's do a proper delta calculation.
                # For each i:
                #   for each f != sol[i]:
                #     if load[f] + d[i] <= cap[f]:
                #       delta_assign = assign_cost[i, f] - assign_cost[i, sol[i]]
                #       delta_open = 0
                #       if load[sol[i]] == d[i]: # source becomes empty
                #           delta_open -= open_cost[sol[i]]
                #       if load[f] == 0: # dest becomes non-empty
                #           delta_open += open_cost[f]
                #       if delta_assign + delta_open < 0:
                #           move
                #           break
                #
                # This is O(N*F) per pass. 100*20 = 2000 ops. Very fast.
                
                for f in range(n_facilities):
                    if f == f_curr:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        delta_assign = assign_cost[i, f] - assign_cost[i, f_curr]
                        delta_open = 0
                        if load[f_curr] == demands[i]:
                            delta_open -= open_cost[f_curr]
                        if load[f] == 0:
                            delta_open += open_cost[f]
                        
                        if delta_assign + delta_open < 1e-9:
                            # Perform move
                            load[f_curr] -= demands[i]
                            load[f] += demands[i]
                            sol[i] = f
                            improved = True
                            break # Move to next customer immediately after a move to update loads?
                            # Actually, updating loads immediately is correct.
                            # But we need to be careful.
                            # Let's restart the pass or just continue?
                            # Continuing is fine.
                            # But we need to update the loop variable f? No, we break inner loop.
                            # We should break out of the f-loop and go to next i.
                            # The 'break' above breaks the f-loop.
                            # We need to ensure we don't try to move i again in this pass.
                            # The structure is:
                            # for i:
                            #   for f:
                            #     if move:
                            #       update
                            #       break
                            # This works.
        
        # 2-Swap
        # Try swapping customers i and j.
        # This is O(N^2).
        # We do a few passes.
        for _ in range(5):
            improved_swap = False
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[sol[i]] += demands[i]
                
            for i in range(n_customers):
                f_i = sol[i]
                d_i = demands[i]
                for j in range(i + 1, n_customers):
                    f_j = sol[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check feasibility of swap
                    # New load at f_i: load[f_i] - d_i + d_j
                    # New load at f_j: load[f_j] - d_j + d_i
                    if load[f_i] - d_i + d_j <= capacities[f_i] and load[f_j] - d_j + d_i <= capacities[f_j]:
                        # Calculate cost change
                        # Assign cost change:
                        # Old: c[i, f_i] + c[j, f_j]
                        # New: c[i, f_j] + c[j, f_i]
                        delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                       (assign_cost[j, f_i] - assign_cost[j, f_j])
                        
                        # Open cost change:
                        # f_i becomes empty if load[f_i] == d_i
                        # f_j becomes empty if load[f_j] == d_j
                        # f_i becomes non-empty? No, it was non-empty.
                        # f_j becomes non-empty? No, it was non-empty.
                        # So open cost only changes if a facility becomes empty.
                        delta_open = 0
                        if load[f_i] == d_i:
                            delta_open -= open_cost[f_i]
                        if load[f_j] == d_j:
                            delta_open -= open_cost[f_j]
                        # Note: If both become empty, we subtract both.
                        # If one becomes empty, subtract one.
                        # Can a facility become non-empty? No, we are moving away from it, so load decreases.
                        # So it can only become empty.
                        
                        if delta_assign + delta_open < 1e-9:
                            # Perform swap
                            sol[i] = f_j
                            sol[j] = f_i
                            load[f_i] -= d_i
                            load[f_i] += d_j
                            load[f_j] -= d_j
                            load[f_j] += d_i
                            improved_swap = True
                            # Don't break, as swapping might allow more swaps?
                            # Or break to restart?
                            # Let's just continue.
                            
            if not improved_swap:
                break
                
        # Update best
        current_cost, _ = calculate_cost(sol)
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = sol.copy()
            U = best_cost
            
        # Check time
        if time.time() > deadline - 2.0:
            break
            
    if best_solution is None:
        # Fallback to simple greedy if something went wrong
        best_solution = greedy_init(np.zeros(n_facilities))
        
    return best_solution.astype(int)
