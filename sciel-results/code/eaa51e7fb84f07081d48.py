# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _min_cost_assignment(demands, capacities, assign_cost, allowed):
    """
    Solve capacitated assignment for customers to a fixed set of allowed facilities.
    Returns (assignment array, total_assign_cost) or (None, inf) if infeasible.
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    if n_fac == 0:
        return None, float('inf')
    
    # Check total capacity
    if sum(capacities[f] for f in allowed) < sum(demands):
        return None, float('inf')
    
    # Build cost matrix: n_cust x (n_fac * capacity_units) is too large.
    # Instead, use a flow-based approach.
    # We'll use a greedy + local search on the assignment within allowed facilities.
    
    # Greedy: sort by demand desc, assign to cheapest feasible facility
    order = np.argsort(-demands)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1)
    
    # Precompute sorted facility indices by cost for each customer
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    feasible = True
    for i in order:
        placed = False
        for f_local in sorted_facs[i]:
            f_global = allowed[f_local]
            if load[f_local] + demands[i] <= capacities[f_global]:
                load[f_local] += demands[i]
                result[i] = f_global
                placed = True
                break
        if not placed:
            feasible = False
            break
    
    if not feasible:
        # Try a more careful assignment using min-cost flow
        return _min_cost_flow(demands, capacities, assign_cost, allowed)
    
    total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
    return result, total_cost


def _min_cost_flow(demands, capacities, assign_cost, allowed):
    """
    Min-cost flow for capacitated assignment.
    Source -> facilities (capacity = cap[f], cost 0)
    facilities -> customers (capacity = demand[c], cost = assign_cost[c,f])
    customers -> sink (capacity = demand[c], cost 0)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    
    # Actually, let's use a different approach: min-cost flow via successive shortest path
    # or just use the linear programming approach.
    # For 100 customers and up to 20 facilities, we can use a simple approach.
    
    # Let's use a penalty-based greedy with repair
    n_cust = len(demands)
    n_fac = len(allowed)
    
    # Try multiple random orderings
    best_result = None
    best_cost = float('inf')
    
    for trial in range(20):
        if trial == 0:
            order = np.argsort(-demands)
        else:
            order = np.random.permutation(n_cust)
        
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1)
        feasible = True
        
        costs_for_allowed = assign_cost[:, allowed]
        sorted_facs = np.argsort(costs_for_allowed, axis=1)
        
        for i in order:
            placed = False
            for f_local in sorted_facs[i]:
                f_global = allowed[f_local]
                if load[f_local] + demands[i] <= capacities[f_global]:
                    load[f_local] += demands[i]
                    result[i] = f_global
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        
        if feasible:
            total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
            if total_cost < best_cost:
                best_cost = total_cost
                best_result = result.copy()
    
    if best_result is None:
        return None, float('inf')
    
    return best_result, best_cost


def _local_search(result, demands, capacities, assign_cost, allowed_set, max_iters=5000):
    """
    Local search: try moving single customers and swapping pairs.
    Only consider facilities in allowed_set.
    """
    n_cust = len(demands)
    n_fac = len(allowed_set)
    allowed = np.array(sorted(allowed_set))
    
    # Build load array
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        idx = np.where(allowed == f)[0]
        if len(idx) > 0:
            load[idx[0]] += demands[i]
    
    # Precompute costs
    costs_allowed = assign_cost[:, allowed]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Single moves
        for i in range(n_cust):
            current_f = result[i]
            current_idx = np.where(allowed == current_f)[0]
            if len(current_idx) == 0:
                continue
            ci = current_idx[0]
            
            for f_local in range(n_fac):
                if f_local == ci:
                    continue
                # Check capacity
                if load[f_local] + demands[i] <= capacities[allowed[f_local]]:
                    delta = assign_cost[i, allowed[f_local]] - assign_cost[i, current_f]
                    if delta < -1e-9:
                        # Make the move
                        load[ci] -= demands[i]
                        load[f_local] += demands[i]
                        result[i] = allowed[f_local]
                        improved = True
                        break
        
        # Swap moves
        for i in range(n_cust):
            if not improved and iters < max_iters:
                fi = result[i]
                fi_idx = np.where(allowed == fi)[0]
                if len(fi_idx) == 0:
                    continue
                ci = fi_idx[0]
                
                for j in range(i+1, n_cust):
                    fj = result[j]
                    fj_idx = np.where(allowed == fj)[0]
                    if len(fj_idx) == 0:
                        continue
                    cj = fj_idx[0]
                    
                    if ci == cj:
                        continue
                    
                    # Check if swap is feasible
                    # After swap: i goes to fj, j goes to fi
                    # Load changes: fi loses demands[i], gains demands[j]
                    #               fj loses demands[j], gains demands[i]
                    new_load_ci = load[ci] - demands[i] + demands[j]
                    new_load_cj = load[cj] - demands[j] + demands[i]
                    
                    if new_load_ci <= capacities[allowed[ci]] and new_load_cj <= capacities[allowed[cj]]:
                        delta = (assign_cost[i, allowed[cj]] + assign_cost[j, allowed[ci]]) - \
                                (assign_cost[i, allowed[ci]] + assign_cost[j, allowed[cj]])
                        if delta < -1e-9:
                            # Make the swap
                            load[ci] = new_load_ci
                            load[cj] = new_load_cj
                            result[i] = allowed[cj]
                            result[j] = allowed[ci]
                            improved = True
                            break


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses facility subset enumeration + greedy/flow assignment + local search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute: for each facility, compute a "value" metric
    # to guide subset selection. Use average assignment cost / open cost.
    avg_assign_cost = assign_cost.mean(axis=0)  # shape (n_fac,)
    
    # Generate candidate subsets of facilities
    # Strategy: 
    # 1. All facilities
    # 2. Top-k by low open cost
    # 3. Top-k by low avg assign cost
    # 4. Combinations
    
    candidate_subsets = []
    
    # All facilities
    candidate_subsets.append(set(range(n_fac)))
    
    # Sort by open cost
    by_open = np.argsort(open_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_open[:k].tolist()))
    
    # Sort by avg assign cost
    by_assign = np.argsort(avg_assign_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_assign[:k].tolist()))
    
    # Sort by combined metric
    combined = open_cost + avg_assign_cost
    by_combined = np.argsort(combined)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_combined[:k].tolist()))
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in candidate_subsets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Evaluate each subset
    best_result = None
    best_total = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 10:
            break
        
        allowed = np.array(sorted(subset))
        result, assign_cost_total = _min_cost_assignment(demands, capacities, assign_cost, allowed)
        
        if result is None:
            continue
        
        # Compute total cost
        open_cost_total = sum(open_cost[f] for f in allowed if any(result[i] == f for i in range(n_cust)))
        total = open_cost_total + assign_cost_total
        
        if total < best_total:
            best_total = total
            best_result = result.copy()
    
    # Local search on best result
    if best_result is not None:
        # Determine which facilities are actually used
        used_facilities = set(best_result.tolist())
        _local_search(best_result, demands, capacities, assign_cost, used_facilities, max_iters=3000)
        
        # Recompute cost
        open_cost_total = sum(open_cost[f] for f in used_facilities)
        assign_cost_total = sum(assign_cost[i, best_result[i]] for i in range(n_cust))
        best_total = open_cost_total + assign_cost_total
    
    # Fallback: if no valid result, use greedy
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result
# EVOLVE-BLOCK-END
