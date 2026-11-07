"""Facility location solver with greedy init, local search (1-swap + 2-swap), and multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: kill after 140s to be safe (evaluator kills at 150s)
    deadline = time.time() + 140.0
    
    def compute_cost(assignment, load):
        """Compute total cost: open costs for used facilities + assignment costs."""
        used = np.unique(assignment[assignment >= 0])
        if len(used) == 0:
            return 0.0
        open_total = np.sum(open_cost[used])
        assign_total = np.sum(assign_cost[np.arange(n_cust), assignment])
        return open_total + assign_total
    
    def greedy_init(rng):
        """Greedy largest-first assignment with randomization."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # Order customers by descending demand with some randomization
        order = np.argsort(-demands)
        # Add some noise to the ordering for diversity
        if rng.random() < 0.3:
            # Perturb: swap some adjacent elements
            order_list = list(order)
            for _ in range(n_cust // 10):
                i = rng.randint(0, n_cust - 2)
                order_list[i], order_list[i+1] = order_list[i+1], order_list[i]
            order = np.array(order_list)
        
        for i in order:
            d = demands[i]
            # Get facilities sorted by assignment cost for this customer
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + d <= capacities[f]:
                    load[f] += d
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Try any facility with remaining capacity (shouldn't happen if feasible)
                for f in range(n_fac):
                    if load[f] + d <= capacities[f]:
                        load[f] += d
                        result[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Last resort: assign to least-loaded facility
                    f = np.argmin(load)
                    load[f] += d
                    result[i] = f
        return result, load
    
    def local_search(assignment, load, max_iters=5000):
        """Local search with 1-swap and 2-swap moves."""
        current_cost = compute_cost(assignment, load)
        no_improve = 0
        max_no_improve = 200
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if no_improve >= max_no_improve:
                break
            
            improved = False
            
            # 1-swap: try moving each customer to a different facility
            # Process in random order
            order = np.random.permutation(n_cust)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                cur_f = assignment[i]
                d = demands[i]
                
                # Try all other facilities
                for f in range(n_fac):
                    if f == cur_f:
                        continue
                    if load[f] + d > capacities[f]:
                        continue
                    
                    # Compute cost change
                    # Remove i from cur_f, add i to f
                    # Open cost change:
                    # - If cur_f becomes empty, we save open_cost[cur_f]
                    # - If f was empty, we pay open_cost[f]
                    cur_load_after = load[cur_f] - d
                    f_load_before = load[f]
                    
                    delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                    delta_open = 0.0
                    if cur_load_after == 0:
                        delta_open -= open_cost[cur_f]
                    if f_load_before == 0:
                        delta_open += open_cost[f]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Accept move
                        load[cur_f] -= d
                        load[f] += d
                        assignment[i] = f
                        current_cost += total_delta
                        improved = True
                        break  # Move this customer to best found facility
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return assignment, load, current_cost
    
    def two_swap(assignment, load, max_iters=2000):
        """Try swapping two customers between facilities."""
        current_cost = compute_cost(assignment, load)
        no_improve = 0
        max_no_improve = 100
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if no_improve >= max_no_improve:
                break
            
            improved = False
            # Randomly pick pairs of customers
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                i, j = np.random.choice(n_cust, 2, replace=False)
                fi = assignment[i]
                fj = assignment[j]
                if fi == fj:
                    continue
                
                di = demands[i]
                dj = demands[j]
                
                # Check capacity: i goes to fj, j goes to fi
                if load[fj] - dj + di > capacities[fj]:
                    continue
                if load[fi] - di + dj > capacities[fi]:
                    continue
                
                # Compute cost change
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                delta_open = 0.0
                fi_load_after = load[fi] - di + dj
                fj_load_after = load[fj] - dj + di
                fi_load_before = load[fi]
                fj_load_before = load[fj]
                
                if fi_load_before > 0 and fi_load_after == 0:
                    delta_open -= open_cost[fi]
                if fi_load_before == 0 and fi_load_after > 0:
                    delta_open += open_cost[fi]
                if fj_load_before > 0 and fj_load_after == 0:
                    delta_open -= open_cost[fj]
                if fj_load_before == 0 and fj_load_after > 0:
                    delta_open += open_cost[fj]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept swap
                    load[fi] = fi_load_after
                    load[fj] = fj_load_after
                    assignment[i] = fj
                    assignment[j] = fi
                    current_cost += total_delta
                    improved = True
                    break
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return assignment, load, current_cost
    
    # Multi-start approach
    best_assignment = None
    best_load = None
    best_cost = float('inf')
    
    # Run multiple starts
    num_starts = 8
    for start in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(42 + start)
        assignment, load = greedy_init(rng)
        
        # Local search phase 1
        assignment, load, cost = local_search(assignment.copy(), load.copy(), max_iters=3000)
        
        # Local search phase 2 (2-swap)
        assignment, load, cost = two_swap(assignment.copy(), load.copy(), max_iters=1000)
        
        # Final 1-swap polish
        assignment, load, cost = local_search(assignment.copy(), load.copy(), max_iters=2000)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
            best_load = load.copy()
        
        # Check if we have time for more starts
        elapsed = time.time() - (deadline - 140.0)
        remaining = deadline - 10.0 - time.time()
        if remaining < 15.0:
            break
    
    # Ensure valid result
    if best_assignment is None or np.any(best_assignment < 0):
        # Fallback: simple greedy
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        return result
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
