"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initial solution: Greedy (Largest demand first)
    # This ensures we have a valid baseline immediately
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    order = np.argsort(-demands)
    for i in order:
        # Find cheapest feasible facility
        # Sort facilities by cost for this customer
        fac_order = np.argsort(assign_cost[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            # This shouldn't happen if total capacity >= total demand, but safety first
            rem_cap = capacities - load
            f = np.argmax(rem_cap)
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
            else:
                # If even that fails, just pick 0 (invalid but prevents crash, evaluator will mark invalid)
                # But problem states capacities are sufficient presumably.
                result[i] = 0
    
    def evaluate(res):
        """Calculate objective value. Returns inf if invalid."""
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            f = res[i]
            if f < 0 or f >= n_facilities:
                return float('inf')
            load[f] += demands[i]
        
        if np.any(load > capacities + 1e-9):
            return float('inf')
            
        used = np.zeros(n_facilities, dtype=bool)
        for f in res:
            used[f] = True
            
        obj = np.sum(open_cost[used])
        obj += np.sum(assign_cost[np.arange(n_customers), res])
        return obj

    current_best = evaluate(result)
    
    # Lagrangian Relaxation Approach
    # Multipliers lambda_f >= 0
    lambdas = np.zeros(n_facilities)
    
    # Precompute costs
    # adjusted_cost[i, f] = assign_cost[i, f] + lambdas[f]
    
    # Subgradient method
    # We want to minimize L(lambda) = sum_i min_f (c_if + lambda_f) + sum_f lambda_f(cap_f - sum_i x_if d_i) + sum_f F_f y_f
    # Actually, standard Lagrangian for CFLP:
    # Min sum F_f y_f + sum c_if x_if
    # s.t. sum_i x_if d_i <= cap_f, x_if in {0,1}, sum_f x_if = 1, y_f >= x_if
    
    # Relax capacity: sum_i x_if d_i <= cap_f
    # L(lambda) = sum_f lambda_f (cap_f - sum_i x_if d_i) + sum F_f y_f + sum c_if x_if
    #            = sum_f lambda_f cap_f + sum_f (F_f y_f - lambda_f sum_i x_if d_i) + sum c_if x_if
    
    # For fixed lambda, we can solve the subproblem:
    # For each customer i, choose f to minimize c_if + lambda_f d_i (effectively, if we group by facility)
    # Wait, the term is - lambda_f sum_i x_if d_i. So the cost for assigning i to f is c_if - lambda_f d_i.
    # If we assign i to f, we pay c_if - lambda_f d_i.
    # Then we decide which facilities to open. If facility f is opened, we pay F_f.
    # However, the opening decision is coupled.
    
    # Standard Lagrangian heuristic:
    # 1. Update lambda
    # 2. Solve subproblem:
    #    For each customer i, find best facility f minimizing c_if + lambda_f d_i ? 
    #    No, the standard relaxation usually relaxes the linking constraints.
    #    Here, let's use a simpler heuristic based on "Effective Cost".
    
    # Alternative: Simulated Annealing or Tabu Search is robust.
    # Given the small size (100x20), Tabu Search with a good neighborhood is very effective.
    
    # Let's implement a robust Tabu Search.
    
    def local_search(res, max_iter=1000):
        """Perform First-Improvement Local Search."""
        current_res = res.copy()
        current_cost = evaluate(current_res)
        if np.isinf(current_cost):
            return current_res, current_cost
            
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[current_res[i]] += demands[i]
            
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
                
            # Try moving each customer to a different facility
            # Optimization: Only consider moves that improve cost
            # Cost change for moving i from f_old to f_new:
            # 1. assign_cost[i, f_new] - assign_cost[i, f_old]
            # 2. Change in open cost:
            #    - If f_old becomes empty: - open_cost[f_old]
            #    - If f_new was empty: + open_cost[f_new]
            
            # To speed up, we can iterate customers in random order or fixed order.
            # Fixed order is deterministic and faster.
            for i in range(n_customers):
                f_old = current_res[i]
                if time.time() > deadline:
                    break
                    
                # Calculate potential gains
                # We need to check all other facilities f_new
                # delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                # delta_open = 0
                # if load[f_old] == demands[i]: # f_old becomes empty
                #     delta_open -= open_cost[f_old]
                # if load[f_new] == 0: # f_new becomes non-empty
                #     delta_open += open_cost[f_new]
                
                # Total delta = delta_assign + delta_open
                
                # We can vectorize the cost calculation for a single customer i
                # costs = assign_cost[i, :]
                # base_cost = costs[f_old]
                # deltas_assign = costs - base_cost
                
                # open_cost_deltas = np.zeros(n_facilities)
                # if load[f_old] == demands[i]:
                #     open_cost_deltas[f_old] = -open_cost[f_old]
                # # For f_new where load is 0, we add open_cost[f_new]
                # empty_facs = np.where(load == 0)[0]
                # open_cost_deltas[empty_facs] += open_cost[empty_facs]
                
                # Note: if f_old is empty (impossible since i is there) or f_new is f_old (skip)
                # We must ensure f_new != f_old
                
                # Let's just loop f_new, it's only 20 facilities.
                best_delta = 0
                best_f_new = -1
                
                # Pre-check: if f_old becomes empty, we save open_cost[f_old]
                f_old_becomes_empty = (load[f_old] == demands[i])
                saved_open = open_cost[f_old] if f_old_becomes_empty else 0.0
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                        
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    # Open cost change
                    delta_open = 0.0
                    if f_old_becomes_empty:
                        delta_open -= saved_open
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                        
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta - 1e-9:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Perform move
                    current_res[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    current_cost += best_delta
                    improved = True
                    if time.time() > deadline:
                        break
                        
        return current_res, current_cost

    # Run Local Search from the greedy solution
    result, current_best = local_search(result)
    
    # Now, let's try a few random restarts with Local Search to escape local optima
    # Or better, use the Lagrangian ideas to generate diverse starting points.
    
    # Let's implement a simple "Lagrangian-guided" restart.
    # Generate a random feasible solution biased by "effective costs".
    
    def generate_lagrangian_solution(lambdas, rng):
        # Effective cost for i to f: assign_cost[i, f] + lambdas[f] * demands[i]
        # This penalizes facilities with high lambda (tight capacity)
        # We assign customers to argmin effective cost, then repair.
        
        res = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending (Largest First is good for feasibility)
        order = np.argsort(-demands)
        
        for i in order:
            eff_costs = assign_cost[i, :] + lambdas * demands[i]
            # Add small random noise to break ties and diversify
            eff_costs += rng.uniform(0, 1, n_facilities) * 10.0 # Scale noise
            
            fac_order = np.argsort(eff_costs)
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    res[i] = f
                    assigned = True
                    break
            
            if not assigned:
                # Repair: assign to facility with most remaining capacity
                rem_cap = capacities - load
                # Only consider facilities with enough capacity
                feasible = np.where(rem_cap >= demands[i])[0]
                if len(feasible) > 0:
                    # Pick random from feasible to diversify
                    f = rng.choice(feasible)
                    load[f] += demands[i]
                    res[i] = f
                else:
                    # Should not happen if total cap >= total demand
                    f = np.argmax(rem_cap)
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        res[i] = f
                    else:
                        res[i] = 0 # Invalid, but safe stop
                        
        return res

    rng = np.random.default_rng(42)
    
    # Run a few iterations of Lagrangian restarts + Local Search
    # We have ~100s left. Local search on 100x20 is fast.
    # Let's try 20-50 restarts.
    
    n_restarts = 0
    max_restarts = 50
    
    # Initial lambdas are 0. We can update them based on load.
    # If a facility is overloaded in the "unconstrained" optimal assignment, increase lambda.
    # But since we repair, we can just use random lambdas or update based on previous solution.
    
    current_lambdas = np.zeros(n_facilities)
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate solution
        cand_res = generate_lagrangian_solution(current_lambdas, rng)
        
        # Local Search
        cand_res, cand_cost = local_search(cand_res, max_iter=500)
        
        if cand_cost < current_best - 1e-9:
            current_best = cand_cost
            result = cand_res
            
        # Update lambdas: 
        # Increase lambda for facilities that are "tight" (high load/capacity)
        # Decrease for others.
        # This encourages the next restart to avoid tight facilities.
        load_cand = np.zeros(n_facilities)
        for i in range(n_customers):
            load_cand[result[i]] += demands[i]
            
        # Subgradient step for lambda
        # Step size alpha
        alpha = 0.5
        # Gradient of L wrt lambda is (cap_f - load_f)
        # We want to minimize L, so if load < cap (slack), gradient is positive -> decrease lambda?
        # Actually, standard subgradient: lambda_new = max(0, lambda_old + alpha * (cap - load))
        # If load > cap (infeasible in subproblem), cap-load is negative -> lambda decreases?
        # Wait, the Lagrangian L(lambda) is a lower bound. We want to maximize the lower bound to close the gap.
        # So we maximize L(lambda).
        # Max L(lambda) => lambda_new = max(0, lambda_old - alpha * grad)
        # grad = cap - load.
        # If load > cap (tight), grad is negative. -alpha * grad is positive. Lambda increases.
        # This makes sense: if facility is tight, increase penalty to push assignments away.
        
        current_lambdas = np.maximum(0, current_lambdas - alpha * (capacities - load_cand))
        
        n_restarts += 1
        
    return result

# EVOLVE-BLOCK-END
