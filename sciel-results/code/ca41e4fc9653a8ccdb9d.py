import numpy as np
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Solve LP Relaxation of the CFLP
    # Variables: x[0..N*M-1] are assignments, x[N*M..N*M+M-1] are open decisions
    # x_{ij} is at index i*M + j
    # y_j is at index N*M + j
    
    c = np.zeros(n_customers * n_facilities + n_facilities)
    for i in range(n_customers):
        for j in range(n_facilities):
            c[i * n_facilities + j] = assign_cost[i, j]
    for j in range(n_facilities):
        c[n_customers * n_facilities + j] = open_cost[j]
        
    # Equality constraints: sum_j x_{ij} = 1 for each i
    A_eq = np.zeros((n_customers, n_customers * n_facilities + n_facilities))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
            
    # Inequality constraints:
    # 1. Capacity: sum_i d_i x_{ij} <= C_j y_j  =>  sum_i d_i x_{ij} - C_j y_j <= 0
    # 2. Linking: x_{ij} <= y_j  =>  x_{ij} - y_j <= 0
    
    n_ineq = n_facilities + n_customers * n_facilities
    A_ub = np.zeros((n_ineq, n_customers * n_facilities + n_facilities))
    b_ub = np.zeros(n_ineq)
    
    # Capacity constraints
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
        A_ub[j, n_customers * n_facilities + j] = -capacities[j]
        
    # Linking constraints
    idx = n_facilities
    for i in range(n_customers):
        for j in range(n_facilities):
            A_ub[idx, i * n_facilities + j] = 1.0
            A_ub[idx, n_customers * n_facilities + j] = -1.0
            idx += 1
            
    # Bounds
    bounds = [(0, 1)] * (n_customers * n_facilities + n_facilities)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status != 0:
            # Fallback to greedy if LP fails
            return _greedy_assign(demands, capacities, open_cost, assign_cost)
        
        x_flat = res.x
        x_lp = x_flat[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        y_lp = x_flat[n_customers * n_facilities:]
        
    except Exception:
        return _greedy_assign(demands, capacities, open_cost, assign_cost)

    # 2. Generate candidate sets of open facilities based on y_lp
    # Sort facilities by y_lp descending
    sorted_facilities = np.argsort(-y_lp)
    
    total_demand = np.sum(demands)
    best_cost = np.inf
    best_assignment = None
    
    # Try subsets of top k facilities
    # Also try the set of all facilities with y_lp > 0
    candidates = set()
    
    # Top k for k in [1, 5, 10, 15, 20]
    for k in [1, 5, 10, 15, 20]:
        if k <= n_facilities:
            candidates.add(tuple(sorted(sorted_facilities[:k])))
            
    # All with y > 0.1
    y_threshold = 0.1
    facs_with_y = [j for j in range(n_facilities) if y_lp[j] > y_threshold]
    if facs_with_y:
        candidates.add(tuple(sorted(facs_with_y)))
        
    # If none valid, add all
    if not candidates:
        candidates.add(tuple(range(n_facilities)))
    
    for fac_tuple in candidates:
        if not fac_tuple:
            continue
            
        # Check capacity
        cap_sum = np.sum([capacities[j] for j in fac_tuple])
        if cap_sum < total_demand:
            continue
            
        # Solve Transportation Problem for this subset
        # This is a small LP: minimize sum c_ij x_ij subject to sum_j x_ij = 1, sum_i d_i x_ij <= C_j
        # Actually, since we fix the open set, we just need to assign.
        # We can use linprog again, or a simpler method.
        # Given small size, linprog is fine.
        
        m = len(fac_tuple)
        if m == 0:
            continue
            
        # Map global facility indices to local indices
        # Variables: x[i, l] for i in 0..N-1, l in 0..m-1
        c_sub = np.zeros(n_customers * m)
        for i in range(n_customers):
            for l, j in enumerate(fac_tuple):
                c_sub[i * m + l] = assign_cost[i, j]
                
        A_eq_sub = np.zeros((n_customers, n_customers * m))
        b_eq_sub = np.ones(n_customers)
        for i in range(n_customers):
            for l in range(m):
                A_eq_sub[i, i * m + l] = 1.0
                
        A_ub_sub = np.zeros((m, n_customers * m))
        b_ub_sub = np.array([capacities[j] for j in fac_tuple])
        for l, j in enumerate(fac_tuple):
            for i in range(n_customers):
                A_ub_sub[l, i * m + l] = demands[i]
                
        bounds_sub = [(0, 1)] * (n_customers * m)
        
        try:
            res_sub = linprog(c_sub, A_ub=A_ub_sub, b_ub=b_ub_sub, A_eq=A_eq_sub, b_eq=b_eq_sub, bounds=bounds_sub, method='highs')
            if res_sub.status == 0:
                # Calculate cost
                assign_cost_sum = res_sub.fun
                open_cost_sum = np.sum([open_cost[j] for j in fac_tuple])
                total_cost = assign_cost_sum + open_cost_sum
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    # Extract assignment
                    x_sub = res_sub.x.reshape(n_customers, m)
                    assignment = np.zeros(n_customers, dtype=int)
                    for i in range(n_customers):
                        # Find argmax
                        local_idx = np.argmax(x_sub[i])
                        assignment[i] = fac_tuple[local_idx]
                    best_assignment = assignment
                    
        except Exception:
            continue
            
    if best_assignment is not None:
        return best_assignment
    else:
        return _greedy_assign(demands, capacities, open_cost, assign_cost)

def _greedy_assign(demands, capacities, open_cost, assign_cost):
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
    return result
