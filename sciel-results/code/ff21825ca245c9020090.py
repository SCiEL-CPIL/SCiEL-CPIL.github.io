"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def assign(demands, capacities, open_cost, assign_cost):
    """
    Solve the capacitated facility location problem using MILP (HiGHS via SciPy)
    with local search polish.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    total_vars = n_facilities + n_customers * n_facilities
    
    # Objective vector
    c = np.concatenate([open_cost, assign_cost.ravel()])
    
    # Build sparse constraint matrix
    rows_a = []
    cols_a = []
    vals_a = []
    
    # Assignment constraints: sum_j x[i,j] = 1 for each customer i
    for i in range(n_customers):
        for j in range(n_facilities):
            rows_a.append(i)
            cols_a.append(n_facilities + i * n_facilities + j)
            vals_a.append(1.0)
    
    # Capacity constraints: sum_i demands[i]*x[i,j] - capacities[j]*y[j] <= 0
    for j in range(n_facilities):
        row_idx = n_customers + j
        rows_a.append(row_idx)
        cols_a.append(j)
        vals_a.append(-capacities[j])
        for i in range(n_customers):
            rows_a.append(row_idx)
            cols_a.append(n_facilities + i * n_facilities + j)
            vals_a.append(demands[i])
    
    A_sparse = csr_matrix((vals_a, (rows_a, cols_a)), shape=(n_customers + n_facilities, total_vars))
    
    cons_lb = np.concatenate([np.ones(n_customers), np.full(n_facilities, -np.inf)])
    cons_ub = np.concatenate([np.ones(n_customers), np.zeros(n_facilities)])
    
    constraints = LinearConstraint(A_sparse, cons_lb, cons_ub)
    
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    # Solve MILP with time management
    result = None
    elapsed = time.time() - start_time
    milp_time_limit = min(80, max(5, deadline - elapsed - 20))
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={
                       "time_limit": milp_time_limit,
                       "mip_rel_gap": 0.01,
                       "threads": 4,
                       "presolve": True,
                   })
        
        if res.status in (0, 2) and res.x is not None:
            x = res.x
            x_assign = x[n_facilities:].reshape(n_customers, n_facilities)
            result = np.zeros(n_customers, dtype=int)
            for i in range(n_customers):
                j = int(np.argmax(x_assign[i]))
                result[i] = j
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        if result is None:
            result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
        else:
            result = _verify_and_fix(result, demands, capacities, open_cost, assign_cost)
        return result
    
    if result is None:
        result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
    else:
        result = _verify_and_fix(result, demands, capacities, open_cost, assign_cost)
    
    # Local search polish with remaining time
    remaining = deadline - (time.time() - start_time) - 2
    if remaining > 1:
        result = _local_search_polish(result, demands, capacities, open_cost, assign_cost, remaining)
    
    return result


def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
    
    return result


def _verify_and_fix(result, demands, capacities, open_cost, assign_cost):
    """Verify the assignment is valid; if not, use greedy fallback."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if np.any(result < 0) or np.any(result >= n_facilities):
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    if np.any(load > capacities + 1e-9):
        return _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    return result


def _local_search_polish(result, demands, capacities, open_cost, assign_cost, time_budget):
    """Local search: try single moves and pairwise swaps to improve objective."""
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    def compute_obj(res, ld):
        open_mask = np.zeros(n_facilities, dtype=bool)
        for i in range(n_customers):
            open_mask[res[i]] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_customers), res])
    
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < time_budget - 1:
        improved = False
        
        # Try single customer moves
        for i in range(n_customers):
            if (time.time() - start) > time_budget - 1:
                break
            current_f = result[i]
            
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                new_load = load.copy()
                new_load[current_f] -= demands[i]
                new_load[new_f] += demands[i]
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                delta_open = 0.0
                if new_load[current_f] == 0 and load[current_f] > 0:
                    delta_open -= open_cost[current_f]
                if new_load[new_f] > 0 and load[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    result[i] = new_f
                    load = new_load
                    improved = True
                    break
        
        # Try pairwise swaps
        if (time.time() - start) > time_budget - 1:
            break
            
        for i in range(n_customers):
            if (time.time() - start) > time_budget - 1:
                break
            for j in range(i + 1, n_customers):
                if (time.time() - start) > time_budget - 1:
                    break
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                new_load_i = load[fi] - demands[i] + demands[j]
                new_load_j = load[fj] - demands[j] + demands[i]
                
                delta_open = 0.0
                if new_load_i == 0 and load[fi] > 0:
                    delta_open -= open_cost[fi]
                if new_load_j == 0 and load[fj] > 0:
                    delta_open -= open_cost[fj]
                if new_load_i > 0 and load[fi] == 0:
                    delta_open += open_cost[fi]
                if new_load_j > 0 and load[fj] == 0:
                    delta_open += open_cost[fj]
                
                delta = delta_assign + delta_open
                if delta < -1e-9:
                    result[i] = fj
                    result[j] = fi
                    load[fi] = new_load_i
                    load[fj] = new_load_j
                    improved = True
    
    return result
# EVOLVE-BLOCK-END
