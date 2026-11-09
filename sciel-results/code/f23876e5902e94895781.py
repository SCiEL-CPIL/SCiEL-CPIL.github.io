"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    best_cost = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    rng = np.random.default_rng(12345)
    
    def compute_total_cost(result, load):
        """Compute total cost: open cost + assignment cost."""
        open_total = open_cost[load > 0].sum()
        assign_total = assign_cost[np.arange(n_cust), result].sum()
        return open_total + assign_total
    
    def local_search(result, load):
        """Local search: try moving each customer to a better facility."""
        improved = True
        max_passes = 50
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            current_cost = compute_total_cost(result, load)
            
            for i in range(n_cust):
                if time.time() > deadline - 5.0:
                    break
                current_f = result[i]
                
                # Try all other facilities
                for f in range(n_fac):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Compute cost delta
                    # Open cost change
                    if load[current_f] == demands[i]:
                        # current_f becomes empty
                        open_delta = -open_cost[current_f]
                    else:
                        open_delta = 0.0
                    
                    if load[f] == 0:
                        # f becomes non-empty
                        open_delta += open_cost[f]
                    
                    # Assign cost change
                    assign_delta = assign_cost[i, f] - assign_cost[i, current_f]
                    
                    total_delta = open_delta + assign_delta
                    
                    if total_delta < -1e-9:
                        # Apply move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        current_cost += total_delta
                        improved = True
                        break
        
        return result, load, current_cost
    
    # GRASP iterations
    iteration = 0
    rcl_size = 5
    
    while time.time() < deadline:
        # Randomized construction using RCL
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        # Process in order of decreasing demand with some randomization
        if iteration == 0:
            order = np.argsort(-demands)
        else:
            # Mix: mostly largest first, but shuffle some
            order = np.argsort(-demands).copy()
            # Shuffle a random subset
            n_shuffle = min(30, n_cust)
            idx = rng.choice(n_cust, size=n_shuffle, replace=False)
            order[idx] = order[idx][rng.permutation(n_shuffle)]
        
        feasible = True
        for i in order:
            # Find feasible facilities
            feasible_mask = (load + demands[i]) <= capacities
            feasible_facs = np.where(feasible_mask)[0]
            
            if len(feasible_facs) == 0:
                feasible = False
                break
            
            # RCL: top-k cheapest among feasible
            k = min(rcl_size, len(feasible_facs))
            # Get costs for feasible facilities
            costs = assign_cost[i, feasible_facs]
            # Get indices of top-k
            top_k_idx = np.argsort(costs)[:k]
            rcl_facs = feasible_facs[top_k_idx]
            
            # Randomly pick from RCL
            chosen = rng.choice(rcl_facs)
            result[i] = chosen
            load[chosen] += demands[i]
        
        if not feasible:
            continue
        
        # Verify feasibility
        if np.any(load > capacities):
            continue
        
        # Local search
        result, load, cost = local_search(result, load)
        
        # Final feasibility check
        if np.any(load > capacities):
            continue
        
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
        
        iteration += 1
        
        # Adapt RCL size: start small, increase for diversity
        if iteration % 50 == 0:
            rcl_size = min(10, 3 + iteration // 20)
        
        # Safety break
        if iteration >= 5000:
            break
    
    # Fallback: greedy if no valid solution found
    if best_cost == np.inf:
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
        best_cost = compute_total_cost(best_result, load)
    
    return best_result
# EVOLVE-BLOCK-END
