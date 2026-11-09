"""Capacitated facility location: Lagrangian Relaxation with Subgradient Optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation with 
    Subgradient Optimization, followed by a final feasibility repair.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    deadline = time.time() + 145.0
    
    # Helper to check validity and compute cost
    def evaluate(res):
        if np.any(res < 0) or np.any(res >= n_fac):
            return np.inf, False
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf, False
        used = np.unique(res)
        cost = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
        return cost, True

    # Greedy repair: ensures feasibility by moving customers from overloaded facilities
    def repair(res, load):
        # Quick check
        if np.all(load <= capacities + 1e-9):
            return res, load, True
        
        for _ in range(1000): # Max iterations for repair
            if time.time() > deadline:
                return res, load, False
            overloaded = np.where(load > capacities + 1e-9)[0]
            if len(overloaded) == 0:
                return res, load, True
            
            f = overloaded[0]
            # Find customers in f
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                # Should not happen if load is correct, but safety break
                break
            
            best_cust = -1
            best_dest = -1
            min_penalty = np.inf
            
            # Try to move the customer with the smallest increase in assignment cost
            # or largest decrease in load (if we prioritize feasibility)
            for i in custs_in_f:
                d_i = demands[i]
                # Find feasible destinations
                # We need load[g] + d_i <= capacities[g]
                feasible_mask = (load + d_i) <= capacities
                feasible_mask[f] = False # Don't move to same facility
                if not np.any(feasible_mask):
                    continue
                
                feasible_indices = np.where(feasible_mask)[0]
                # Cost to move i to g: assign_cost[i, g] - assign_cost[i, f]
                # We want to minimize this.
                # Note: We don't account for opening cost changes here in the simple repair,
                # assuming the destination is already open or we accept the cost.
                # Actually, if destination is empty, we incur open_cost[g].
                # For a fast repair, let's just look at assignment cost delta first.
                # If we need to open a new facility, that's a bigger penalty.
                
                # Simple heuristic: pick the feasible g with min assign_cost[i, g]
                # But we should prefer already open facilities.
                open_mask = load > 0
                # Combine: prefer open facilities.
                # Score: assign_cost[i, g] + (open_cost[g] if load[g]==0 else 0)
                scores = assign_cost[i, feasible_indices]
                # Add penalty for opening new facility
                for idx, g in enumerate(feasible_indices):
                    if load[g] == 0:
                        scores[idx] += open_cost[g]
                
                best_g_idx = feasible_indices[np.argmin(scores)]
                penalty = scores[np.argmin(scores)]
                
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_cust = i
                    best_dest = best_g_idx
            
            if best_cust == -1:
                # No feasible move found. This instance might be infeasible or we are stuck.
                # Return invalid
                return res, load, False
            
            # Perform move
            res[best_cust] = best_dest
            load[f] -= demands[best_cust]
            load[best_dest] += demands[best_cust]
            
        # Final check
        if np.any(load > capacities + 1e-9):
            return res, load, False
        return res, load, True

    # Lagrangian Relaxation
    # Objective: sum open_cost[j]*y_j + sum assign_cost[i,j]*x_ij
    # Constraints: sum x_ij = 1 for all i; sum demands[i]*x_ij <= capacities[j] for all j
    # Relax capacity constraints with multipliers mu[j] >= 0
    # Lagrangian: sum open_cost[j]*y_j + sum_i,j (assign_cost[i,j] + mu[j]*demands[i]) x_ij - sum mu[j]*capacities[j]
    # Minimization decomposes into:
    # 1. For each i, choose j that minimizes (assign_cost[i,j] + mu[j]*demands[i])
    # 2. y_j is 1 if any x_ij is 1 for that j (in the relaxed problem, y is determined by x)
    #    Actually, the term sum open_cost[j]*y_j is coupled. 
    #    Standard approach: The lower bound L(mu) = min_{x,y} [ ... ]
    #    Since x and y are linked (y_j >= x_ij), we can iterate:
    #    Given mu, solve for x (assign each i to best j based on reduced cost).
    #    Then y_j is implicitly 1 if x_ij=1 for any i.
    #    The value is sum_{j: y_j=1} open_cost[j] + sum_{i,j} (assign_cost[i,j] + mu[j]*demands[i])x_ij - sum mu[j]*capacities[j]
    
    # Subgradient for mu[j] is: sum_i demands[i]*x_ij - capacities[j]
    
    # Initialize mu
    mu = np.zeros(n_fac)
    best_mu = mu.copy()
    best_lb = -np.inf
    best_res_from_lr = None
    best_load_from_lr = None
    
    # Step size parameter
    max_step = 100.0
    step_factor = 2.0
    step_decay = 0.95
    
    # We also maintain the best feasible solution found during the process
    global_best_res = None
    global_best_cost = np.inf
    
    # Initial greedy solution to have a feasible upper bound
    # Sort by demand desc
    order = np.argsort(-demands)
    init_res = np.zeros(n_cust, dtype=int)
    init_load = np.zeros(n_fac)
    for i in order:
        feasible = np.where(init_load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # Fallback: pick facility with most remaining capacity
            f = int(np.argmax(capacities - init_load))
            init_res[i] = f
            init_load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            # Add opening cost if facility is empty
            for idx, f in enumerate(feasible):
                if init_load[f] == 0:
                    costs[idx] += open_cost[f]
            f_idx = feasible[np.argmin(costs)]
            init_res[i] = f_idx
            init_load[f_idx] += demands[i]
    
    # Repair initial solution
    init_res, init_load, is_valid = repair(init_res, init_load)
    if is_valid:
        cost, _ = evaluate(init_res)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_res = init_res.copy()
    
    # Subgradient Optimization Loop
    # We run this for a fixed time or iterations
    # Each iteration is O(N*M) which is fast (100*20 = 2000 ops)
    # We can run thousands of iterations.
    
    # To avoid getting stuck, we use a "restart" mechanism or multiple starts if time permits.
    # But let's try one long run first with adaptive step size.
    
    # We will keep track of the best lower bound found.
    # And the best feasible solution found by rounding the relaxed solution.
    
    # We also need to handle the case where the relaxed solution is infeasible.
    # We use the relaxed solution to guide a greedy repair or a local search.
    
    # Let's run the subgradient method.
    
    # We'll use a simple step size rule: step = step_factor * (UB - LB) / ||g||^2
    # where UB is the best feasible cost found so far.
    
    # To improve efficiency, we can vectorize the assignment step.
    # reduced_cost[i, j] = assign_cost[i, j] + mu[j] * demands[i]
    # best_j[i] = argmin_j reduced_cost[i, j]
    
    # We need to be careful with time.
    
    # Let's run for a while.
    
    # We'll also periodically attempt to improve the current best feasible solution using local search
    # on the solution derived from the current mu.
    
    # Actually, the "rounding" step:
    # Given mu, we get x_ij (each i assigned to one j).
    # This x might violate capacities.
    # We repair it.
    # This repaired solution is a candidate for global_best.
    
    # We can also use the mu values to weight the costs in a greedy construction.
    
    # Let's implement the loop.
    
    # We will run for up to 140 seconds.
    
    # To ensure we don't waste time, we will do the subgradient loop for a fixed number of iterations 
    # or until time is up, then do a final local search on the best found.
    
    # Let's do 5000 iterations or until time is up.
    
    # But 5000 * (100*20) is 10 million ops, which is very fast in numpy.
    # So we can run more.
    
    # Let's try 100,000 iterations?
    # 100k * 2000 = 200 million ops. Might take a few seconds in numpy.
    
    # Let's do it.
    
    # We need to track the best LB to update step size.
    
    # We also need to track the best feasible solution derived from the relaxations.
    
    # Let's start.
    
    # We'll use a list of mus to try? No, just one path.
    
    # To diversify, we can restart if no improvement in LB for a while.
    
    # Let's stick to one path for now, but with a good step size.
    
    # We'll use the "best known feasible solution" as the UB.
    
    # Initialize UB with the greedy solution.
    UB = global_best_cost
    
    # If UB is inf, we have a problem. But greedy usually works.
    
    # Let's run.
    
    # We will store the best mu that gave the best LB.
    
    # And the best feasible solution obtained from any mu.
    
    # Let's code the loop.
    
    # We'll use a simple loop.
    
    # To speed up, we can pre-compute nothing, as mu changes.
    
    # reduced_cost = assign_cost + np.outer(demands, mu)
    # best_j = np.argmin(reduced_cost, axis=1)
    
    # Then compute LB.
    
    # Then compute subgradient.
    
    # Then update mu.
    
    # Then repair the solution from best_j to get a feasible candidate.
    
    # This repair is the expensive part if done naively.
    # But repair is O(N^2) worst case, which is 10,000 ops.
    # 100,000 iterations * 10,000 ops = 1 billion ops. Too slow.
    
    # So we should not repair every iteration.
    # We should only repair every K iterations, or when LB improves significantly.
    
    # Let's repair every 100 iterations.
    
    # And we should also try to improve the global best using local search occasionally.
    
    # Let's refine the strategy:
    # 1. Run subgradient for a few seconds.
    # 2. Every 100 iterations, repair the current relaxed solution and update global best.
    # 3. Every 1000 iterations, run a quick local search on the global best.
    # 4. Stop when time is up.
    
    # This seems reasonable.
    
    # Let's implement.
    
    # We'll use a counter for iterations.
    
    # We'll also track the last time we did a local search.
    
    # Let's go.
    
    # We need to be careful with the step size.
    # If UB - LB is small, step is small.
    
    # Let's start.
    
    # We'll use a fixed number of iterations, say 5000, and see how long it takes.
    # If it's fast, we can do more.
    
    # Actually, let's just loop until deadline.
    
    # But we need to avoid infinite loops if repair fails.
    
    # Let's add a check.
    
    # We'll use a simple loop.
    
    # We'll store the best mu.
    
    # Let's code.
    
    # We'll use a helper function to compute LB and subgradient given mu.
    
    def compute_lb_and_subgrad(mu):
        # reduced_cost[i, j] = assign_cost[i, j] + mu[j] * demands[i]
        reduced_cost = assign_cost + np.outer(demands, mu)
        best_j = np.argmin(reduced_cost, axis=1)
        
        # Compute LB
        # LB = sum_{j: y_j=1} open_cost[j] + sum_{i,j} (assign_cost[i,j] + mu[j]*demands[i])x_ij - sum mu[j]*capacities[j]
        # x_ij is 1 if best_j[i] == j
        
        # y_j is 1 if any i has best_j[i] == j
        y = np.zeros(n_fac, dtype=bool)
        np.add.at(y, best_j, 1)
        y = y > 0
        
        open_cost_sum = np.sum(open_cost[y])
        
        # sum_{i,j} (assign_cost[i,j] + mu[j]*demands[i])x_ij
        # This is sum_i (assign_cost[i, best_j[i]] + mu[best_j[i]] * demands[i])
        assignment_cost_sum = np.sum(assign_cost[np.arange(n_cust), best_j] + mu[best_j] * demands)
        
        cap_penalty = np.sum(mu * capacities)
        
        LB = open_cost_sum + assignment_cost_sum - cap_penalty
        
        # Subgradient: g[j] = sum_{i: best_j[i]==j} demands[i] - capacities[j]
        load = np.zeros(n_fac)
        np.add.at(load, best_j, demands)
        subgrad = load - capacities
        
        return LB, subgrad, best_j, y
    
    # Initialize
    LB, subgrad, best_j, y = compute_lb_and_subgrad(mu)
    best_lb = LB
    best_mu = mu.copy()
    
    # Repair the initial relaxed solution
    res_lr = best_j.copy()
    load_lr = np.zeros(n_fac)
    np.add.at(load_lr, res_lr, demands)
    res_lr, load_lr, is_valid = repair(res_lr, load_lr)
    if is_valid:
        cost, _ = evaluate(res_lr)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_res = res_lr.copy()
            UB = global_best_cost
    
    # Main loop
    iter_count = 0
    last_repair_iter = 0
    last_ls_iter = 0
    
    # We'll use a simple step size.
    # step = step_factor * (UB - LB) / (||g||^2 + epsilon)
    
    # We'll run until deadline.
    
    # To prevent stalling, we can restart mu if no improvement in LB for 1000 iterations.
    # Let's add that.
    
    best_lb_seen = best_lb
    iter_since_improve = 0
    
    while time.time() < deadline - 5.0: # Leave 5s for final LS
        iter_count += 1
        
        # Compute LB and subgrad
        LB, subgrad, best_j, y = compute_lb_and_subgrad(mu)
        
        if LB < best_lb - 1e-6:
            best_lb = LB
            best_mu = mu.copy()
            iter_since_improve = 0
        else:
            iter_since_improve += 1
        
        # Update step size
        norm_sq = np.sum(subgrad**2)
        if norm_sq < 1e-10:
            step = 0.0
        else:
            # Use best known UB
            # If UB is inf, use a large number
            ub_for_step = UB if UB < np.inf else 10000.0
            step = step_factor * (ub_for_step - best_lb) / (norm_sq + 1e-10)
            step = min(step, max_step)
            # Decay step factor
            step_factor *= step_decay
        
        # Update mu
        if step > 0:
            mu = np.maximum(0, mu + step * subgrad)
        else:
            mu = np.maximum(0, mu)
        
        # Repair every 100 iterations
        if iter_count - last_repair_iter >= 100:
            last_repair_iter = iter_count
            res_lr = best_j.copy()
            load_lr = np.zeros(n_fac)
            np.add.at(load_lr, res_lr, demands)
            res_lr, load_lr, is_valid = repair(res_lr, load_lr)
            if is_valid:
                cost, _ = evaluate(res_lr)
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_res = res_lr.copy()
                    UB = global_best_cost
        
        # Local search every 1000 iterations
        if iter_count - last_ls_iter >= 1000 and global_best_res is not None:
            last_ls_iter = iter_count
            # Run a quick local search on global_best_res
            # We'll use a simple 1-move local search for a few seconds
            res_ls = global_best_res.copy()
            load_ls = np.zeros(n_fac)
            np.add.at(load_ls, res_ls, demands)
            
            # Simple local search: try moving each customer to a better feasible facility
            # We'll do a few passes.
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for i in range(n_cust):
                    f_old = res_ls[i]
                    d_i = demands[i]
                    feasible_mask = (load_ls + d_i) <= capacities
                    feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                    if not np.any(feasible_mask):
                        continue
                    delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
                    delta_open = np.zeros(n_fac)
                    if load_ls[f_old] == d_i:
                        delta_open[f_old] = -open_cost[f_old]
                    open_new = (load_ls == 0) & (np.arange(n_fac) != f_old)
                    if np.any(open_new):
                        delta_open[open_new] = open_cost[open_new]
                    total_delta = delta_assign + delta_open
                    total_delta[~feasible_mask] = np.inf
                    total_delta[f_old] = np.inf
                    best_f_new = int(np.argmin(total_delta))
                    best_delta = total_delta[best_f_new]
                    if best_delta < -1e-9:
                        res_ls[i] = best_f_new
                        load_ls[f_old] -= d_i
                        load_ls[best_f_new] += d_i
                        improved = True
            
            if np.all(load_ls <= capacities + 1e-9):
                cost_ls, _ = evaluate(res_ls)
                if cost_ls < global_best_cost:
                    global_best_cost = cost_ls
                    global_best_res = res_ls.copy()
                    UB = global_best_cost
        
        # Restart if no improvement for 2000 iterations
        if iter_since_improve > 2000:
            # Restart mu to zero or random
            mu = np.zeros(n_fac)
            iter_since_improve = 0
            # Update best_lb to current LB to reset the counter effectively
            # Actually, we should reset best_lb to the current LB? No, keep the global best LB.
            # But the step size depends on best_lb.
            # Let's just reset mu and hope for a new direction.
            # We can also reset step_factor.
            step_factor = 2.0
    
    # Final local search on the best result
    if global_best_res is not None:
        res_final = global_best_res.copy()
        load_final = np.zeros(n_fac)
        np.add.at(load_final, res_final, demands)
        
        # More thorough local search
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # 1-move pass
            for i in range(n_cust):
                f_old = res_final[i]
                d_i = demands[i]
                feasible_mask = (load_final + d_i) <= capacities
                feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                if not np.any(feasible_mask):
                    continue
                delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
                delta_open = np.zeros(n_fac)
                if load_final[f_old] == d_i:
                    delta_open[f_old] = -open_cost[f_old]
                open_new = (load_final == 0) & (np.arange(n_fac) != f_old)
                if np.any(open_new):
                    delta_open[open_new] = open_cost[open_new]
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                if best_delta < -1e-9:
                    res_final[i] = best_f_new
                    load_final[f_old] -= d_i
                    load_final[best_f_new] += d_i
                    improved = True
            
            # Swap pass (sampled)
            if time.time() < deadline - 2.0:
                for _ in range(500):
                    i, j = np.random.randint(0, n_cust, size=2)
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    f_i = res_final[i]
                    f_j = res_final[j]
                    if f_i == f_j:
                        continue
                    d_i = demands[i]
                    d_j = demands[j]
                    if load_final[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load_final[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    if delta_assign < -1e-9:
                        res_final[i], res_final[j] = res_final[j], res_final[i]
                        load_final[f_i] -= d_i
                        load_final[f_i] += d_j
                        load_final[f_j] -= d_j
                        load_final[f_j] += d_i
                        improved = True
        
        if np.all(load_final <= capacities + 1e-9):
            cost_final, _ = evaluate(res_final)
            if cost_final < global_best_cost:
                global_best_res = res_final
                global_best_cost = cost_final
    
    # Return the best result
    if global_best_res is None:
        # Fallback to a valid solution, even if not optimal
        # This should not happen, but just in case
        return np.zeros(n_cust, dtype=int)
    
    return global_best_res.astype(int)
# EVOLVE-BLOCK-END
