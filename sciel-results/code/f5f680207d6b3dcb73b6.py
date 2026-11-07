"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost: sum of assign costs + sum of open costs for used facilities."""
    n_cust = len(result)
    n_fac = len(capacities)
    total = 0.0
    used = np.zeros(n_fac, dtype=bool)
    for i in range(n_cust):
        f = result[i]
        total += assign_cost[i, f]
        used[f] = True
    for f in range(n_fac):
        if used[f]:
            total += open_cost[f]
    return total

def _greedy_assign(demands, capacities, open_cost, assign_cost, order):
    """Greedy: assign customers in given order to cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        if result[i] == -1:
            return None
    return result

def _local_search(demands, capacities, open_cost, assign_cost, init_result, deadline):
    """
    Efficient local search with incremental cost tracking.
    Tries single-customer moves and 2-customer swaps.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = init_result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    # Track open facilities count per facility
    cust_count = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        cust_count[best_result[i]] += 1
    
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    # Precompute facility cost vectors for each customer
    # Sort facilities by assignment cost for each customer
    fac_sorted = np.argsort(assign_cost, axis=1)  # shape (n_cust, n_fac)
    
    start_time = time.time()
    improved = True
    max_passes = 200
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        customer_order = np.random.permutation(n_cust)
        
        for i in customer_order:
            if time.time() > deadline:
                break
            cur_f = best_result[i]
            
            # Try moving customer i to each other facility
            # Compute cost delta for moving i from cur_f to new_f
            # Delta = assign_cost[i,new_f] - assign_cost[i,cur_f]
            #       + (open_cost[new_f] if cust_count[new_f]==0 else 0)
            #       - (open_cost[cur_f] if cust_count[cur_f]==1 else 0)
            
            for new_f in fac_sorted[i]:
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Compute cost delta
                delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                if cust_count[new_f] == 0:
                    delta += open_cost[new_f]
                if cust_count[cur_f] == 1:
                    delta -= open_cost[cur_f]
                
                if delta < -1e-9:
                    # Accept the move
                    load[cur_f] -= demands[i]
                    cust_count[cur_f] -= 1
                    load[new_f] += demands[i]
                    cust_count[new_f] += 1
                    best_result[i] = new_f
                    best_cost += delta
                    improved = True
                    break  # Move to next customer after one improvement
        
        # Try 2-customer swaps
        if time.time() > deadline:
            break
        for _ in range(50):  # Limited swap attempts per pass
            if time.time() > deadline:
                break
            i1, i2 = np.random.choice(n_cust, 2, replace=False)
            f1, f2 = best_result[i1], best_result[i2]
            if f1 == f2:
                continue
            
            # Swap: i1 goes to f2, i2 goes to f1
            if load[f2] - demands[i2] + demands[i1] > capacities[f2]:
                continue
            if load[f1] - demands[i1] + demands[i2] > capacities[f1]:
                continue
            
            delta = (assign_cost[i1, f2] - assign_cost[i1, f1]) + \
                    (assign_cost[i2, f1] - assign_cost[i2, f2])
            
            # Open cost changes
            if cust_count[f1] == 1 and cust_count[f2] == 1:
                # Both facilities lose one customer; if one becomes 0, we save its open cost
                # After swap: f1 has cust_count-1+1=cust_count, f2 has cust_count-1+1=cust_count
                # So no change in open costs if both had exactly 1 customer before
                # Actually after swap, f1 still has 1 customer (i2), f2 still has 1 (i1)
                pass
            elif cust_count[f1] == 1:
                # f1 loses i1 but gains i2, so still has 1 customer - no change
                pass
            elif cust_count[f2] == 1:
                pass
            
            # More careful: check if any facility goes from 0 to >0 or >0 to 0
            # Before: f1 has cust_count[f1], f2 has cust_count[f2]
            # After: f1 has cust_count[f1]-1+1 = cust_count[f1], f2 has cust_count[f2]-1+1 = cust_count[f2]
            # So open costs don't change for swaps!
            
            if delta < -1e-9:
                load[f1] -= demands[i1]
                load[f2] -= demands[i2]
                load[f2] += demands[i1]
                load[f1] += demands[i2]
                cust_count[f1] -= 1
                cust_count[f2] -= 1
                cust_count[f2] += 1
                cust_count[f1] += 1
                best_result[i1] = f2
                best_result[i2] = f1
                best_cost += delta
                improved = True
    
    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Generate multiple initial solutions
    best_result = None
    best_cost = float('inf')
    
    # Strategy 1: Largest demand first (hard to place)
    order1 = np.argsort(-demands)
    res1 = _greedy_assign(demands, capacities, open_cost, assign_cost, order1)
    if res1 is not None:
        cost1 = _compute_cost(res1, demands, capacities, open_cost, assign_cost)
        if cost1 < best_cost:
            best_cost = cost1
            best_result = res1
    
    # Strategy 2: Cheapest assignment cost first
    min_costs = np.min(assign_cost, axis=1)
    order2 = np.argsort(min_costs)
    res2 = _greedy_assign(demands, capacities, open_cost, assign_cost, order2)
    if res2 is not None:
        cost2 = _compute_cost(res2, demands, capacities, open_cost, assign_cost)
        if cost2 < best_cost:
            best_cost = cost2
            best_result = res2
    
    # Strategy 3: Average cost first
    avg_costs = np.mean(assign_cost, axis=1)
    order3 = np.argsort(avg_costs)
    res3 = _greedy_assign(demands, capacities, open_cost, assign_cost, order3)
    if res3 is not None:
        cost3 = _compute_cost(res3, demands, capacities, open_cost, assign_cost)
        if cost3 < best_cost:
            best_cost = cost3
            best_result = res3
    
    # Strategy 4: Random orders
    for _ in range(20):
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
        order_rand = np.random.permutation(n_cust)
        res_r = _greedy_assign(demands, capacities, open_cost, assign_cost, order_rand)
        if res_r is not None:
            cost_r = _compute_cost(res_r, demands, capacities, open_cost, assign_cost)
            if cost_r < best_cost:
                best_cost = cost_r
                best_result = res_r
    
    if best_result is None:
        # Fallback: assign all to facility 0 (may violate capacity but better than nothing)
        # Actually try a simple round-robin
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    best_result[i] = f
                    load[f] += demands[i]
                    break
            else:
                best_result[i] = 0
        return best_result
    
    # Local search to improve
    final_result, final_cost = _local_search(
        demands, capacities, open_cost, assign_cost, best_result, deadline
    )
    
    return final_result

# EVOLVE-BLOCK-END
