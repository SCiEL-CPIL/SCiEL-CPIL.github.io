"""Capacitated facility location: model-and-optimize with local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _true_obj(result, open_cost, assign_cost):
    """Sum of open costs (facilities used) + sum of assignment costs."""
    used = np.unique(result)
    open_part = float(open_cost[used].sum())
    assign_part = float(assign_cost[np.arange(len(result)), result].sum())
    return open_part + assign_part


def _greedy_seed(demands, capacities, open_cost, assign_cost, share_param):
    """Largest demand first; choose cheapest facility w.r.t. effective cost."""
    n_fac = len(open_cost)
    n_cust = len(demands)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    # effective cost: assign_cost + share of open cost per unit demand
    eff = assign_cost + share_param * open_cost[None, :] / np.maximum(demands.mean(), 1)
    order = np.argsort(-demands)
    for i in order:
        # find cheapest feasible facility
        feasible = load + demands[i] <= capacities
        if not feasible.any():
            # fallback: cheapest facility overall
            f = int(np.argmin(eff[i]))
        else:
            eff_masked = eff[i].copy()
            eff_masked[~feasible] = np.inf
            f = int(np.argmin(eff_masked))
        load[f] += demands[i]
        result[i] = f
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, max_iters=200, deadline=None):
    """Improve via single reassignments and swaps."""
    n_cust = len(demands)
    n_fac = len(open_cost)
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_obj = _true_obj(result, open_cost, assign_cost)
    current = result.copy()
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n_cust):
            cur_f = current[i]
            cur_cost_i = assign_cost[i, cur_f]
            # Try moving customer i to a different facility
            for f in range(n_fac):
                if f == cur_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Compute change in objective
                new_cost_i = assign_cost[i, f]
                delta_assign = new_cost_i - cur_cost_i
                
                # Open cost change:
                # If cur_f loses its last customer, we save open_cost[cur_f]
                # If f gains its first customer, we pay open_cost[f]
                open_delta = 0.0
                if load[cur_f] == demands[i]:
                    open_delta -= open_cost[cur_f]
                if load[f] == 0:
                    open_delta += open_cost[f]
                
                total_delta = delta_assign + open_delta
                if total_delta < -1e-9:
                    # Apply move
                    load[cur_f] -= demands[i]
                    load[f] += demands[i]
                    current[i] = f
                    best_obj += total_delta
                    improved = True
                    break  # restart scan after a move
    
    return current, best_obj


def _try_reassign_empty(demands, capacities, open_cost, assign_cost, result, deadline=None):
    """Try to close empty facilities by reassigning their customers."""
    n_cust = len(demands)
    n_fac = len(open_cost)
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    current = result.copy()
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for f in range(n_fac):
            if load[f] == 0:
                continue
            # Check if we can move all customers from f to other facilities
            custs_at_f = [i for i in range(n_cust) if current[i] == f]
            if not custs_at_f:
                continue
            # Try to move each customer to cheapest alternative
            temp_load = load.copy()
            temp_current = current.copy()
            feasible = True
            new_cost = 0.0
            for i in custs_at_f:
                best_alt = None
                best_alt_cost = np.inf
                for f2 in range(n_fac):
                    if f2 == f:
                        continue
                    if temp_load[f2] + demands[i] <= capacities[f2]:
                        c = assign_cost[i, f2]
                        if c < best_alt_cost:
                            best_alt_cost = c
                            best_alt = f2
                if best_alt is None:
                    feasible = False
                    break
                temp_load[f2] += demands[i]
                temp_current[i] = f2
                new_cost += best_alt_cost
            if feasible:
                # Check if saving open_cost[f] outweighs extra assignment cost
                old_assign_cost = sum(assign_cost[i, f] for i in custs_at_f)
                if new_cost - old_assign_cost < open_cost[f] - 1e-9:
                    load = temp_load
                    current = temp_current
                    improved = True
    
    return current


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(open_cost)
    
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Try multiple share parameters for greedy seed
    best_result = None
    best_obj = np.inf
    
    share_params = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for sp in share_params:
        if time.time() > deadline:
            break
        result = _greedy_seed(demands, capacities, open_cost, assign_cost, sp)
        obj = _true_obj(result, open_cost, assign_cost)
        if obj < best_obj:
            best_obj = obj
            best_result = result.copy()
    
    if best_result is None:
        # Fallback: simple greedy
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, 0.0)
        best_obj = _true_obj(best_result, open_cost, assign_cost)
    
    # Local search on best result
    if time.time() < deadline:
        improved_result, improved_obj = _local_search(
            demands, capacities, open_cost, assign_cost, 
            best_result, max_iters=300, deadline=deadline
        )
        if improved_obj < best_obj:
            best_result = improved_result
            best_obj = improved_obj
    
    # Try to close empty facilities
    if time.time() < deadline:
        closed_result = _try_reassign_empty(
            demands, capacities, open_cost, assign_cost,
            best_result, deadline=deadline
        )
        closed_obj = _true_obj(closed_result, open_cost, assign_cost)
        if closed_obj < best_obj:
            best_result = closed_result
            best_obj = closed_obj
    
    # Another round of local search
    if time.time() < deadline - 10:
        improved_result, improved_obj = _local_search(
            demands, capacities, open_cost, assign_cost,
            best_result, max_iters=200, deadline=deadline
        )
        if improved_obj < best_obj:
            best_result = improved_result
            best_obj = improved_obj
    
    # Ensure valid result
    # Verify all customers are assigned and capacities are respected
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            # Fallback assignment
            for f2 in range(n_fac):
                if load[f2] + demands[i] <= capacities[f2]:
                    best_result[i] = f2
                    load[f2] += demands[i]
                    break
            else:
                best_result[i] = 0
                load[0] += demands[i]
        else:
            load[f] += demands[i]
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
