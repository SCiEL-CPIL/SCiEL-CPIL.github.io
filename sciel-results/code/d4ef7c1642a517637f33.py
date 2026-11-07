"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _solve_transportation(demands, capacities, assign_cost, open_facilities):
    """
    Solve the capacitated assignment problem for a FIXED set of open facilities.
    This is a transportation problem, which is totally unimodular, so LP relaxation
    gives integer optimal solutions.
    
    Variables: x_{i,j} for i in customers, j in open_facilities
    Minimize: sum assign_cost[i,j] * x_{i,j}
    Subject to:
      sum_j x_{i,j} = 1 for all i
      sum_i x_{i,j} * demands[i] <= capacities[j] for all j
      x_{i,j} >= 0
    """
    n_cust = len(demands)
    n_fac = len(open_facilities)
    if n_fac == 0:
        return None
    
    # Map facility indices
    fac_idx = np.array(open_facilities)
    
    # Variables x_{i,j} flattened: i * n_fac + j
    n_vars = n_cust * n_fac
    
    # Objective: minimize sum assign_cost[i,j] * x_{i,j}
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j_idx, j in enumerate(fac_idx):
            c[i * n_fac + j_idx] = assign_cost[i, j]
    
    # Constraints:
    # 1. For each customer i: sum_j x_{i,j} = 1
    # 2. For each facility j: sum_i x_{i,j} * demands[i] <= capacities[j]
    
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for j_idx in range(n_fac):
            A_eq[i, i * n_fac + j_idx] = 1.0
    
    A_ub = np.zeros((n_fac, n_vars))
    b_ub = np.zeros(n_fac)
    for j_idx, j in enumerate(fac_idx):
        for i in range(n_cust):
            A_ub[j_idx, i * n_fac + j_idx] = demands[i]
        b_ub[j_idx] = capacities[j]
    
    bounds = [(0, None)] * n_vars
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None
    
    # Round to get integer solution (should already be near-integer due to TU)
    x_flat = np.round(res.x).astype(int)
    x = x_flat.reshape(n_cust, n_fac)
    
    # Verify: each customer assigned to exactly one facility
    for i in range(n_cust):
        if x[i].sum() != 1:
            # Try to fix: assign to cheapest feasible
            return None
    
    # Verify capacity constraints
    for j_idx in range(n_fac):
        load = sum(demands[i] for i in range(n_cust) if x[i, j_idx] == 1)
        if load > capacities[fac_idx[j_idx]] + 1e-6:
            return None
    
    # Convert to facility indices
    result = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        for j_idx in range(n_fac):
            if x[i, j_idx] == 1:
                result[i] = fac_idx[j_idx]
                break
    
    return result

def _greedy_assign(demands, capacities, assign_cost):
    """Greedy fallback: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(len(capacities))
    
    order = np.argsort(-demands)
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            result[i] = 0
    return result

def _local_search(result, demands, capacities, assign_cost, open_cost, max_iters=500, deadline=None):
    """
    Local search: try moving single customers or swapping pairs to reduce cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    current_cost = sum(open_cost[j] for j in range(n_fac) if any(result == j)) + \
                   sum(assign_cost[i, result[i]] for i in range(n_cust))
    
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if deadline is not None and time.time() > deadline:
                break
            current_fac = result[i]
            # Try all other facilities in order of cost
            for f in np.argsort(assign_cost[i]):
                if f == current_fac:
                    continue
                # Check capacity
                if load[f] + demands[i] <= capacities[f]:
                    new_cost = current_cost - assign_cost[i, current_fac] + assign_cost[i, f]
                    # Check if opening cost changes
                    if load[current_fac] == demands[i]:
                        # Closing current facility
                        if any(result == current_fac and result != i):
                            new_cost -= 0  # Still open
                        else:
                            new_cost -= open_cost[current_fac]
                    if load[f] == 0:
                        new_cost += open_cost[f]
                    
                    if new_cost < current_cost - 1e-9:
                        # Make the move
                        load[current_fac] -= demands[i]
                        load[f] += demands[i]
                        result[i] = f
                        current_cost = new_cost
                        improved = True
                        break
    
    return result, current_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Strategy: 
    # 1. Get an initial solution via greedy
    # 2. Use local search to improve
    # 3. Try LP-based approaches to find better open facility sets
    
    # Initial greedy solution
    result = _greedy_assign(demands, capacities, assign_cost)
    
    # Compute initial cost
    def compute_cost(res):
        open_fac = set(res.tolist())
        cost = sum(open_cost[j] for j in open_fac)
        cost += sum(assign_cost[i, res[i]] for i in range(n_cust))
        return cost
    
    best_result = result.copy()
    best_cost = compute_cost(best_result)
    
    # Local search
    result, cost = _local_search(result, demands, capacities, assign_cost, open_cost, 
                                  max_iters=100, deadline=deadline)
    if cost < best_cost:
        best_result = result.copy()
        best_cost = cost
    
    # Try LP-based approach: solve LP relaxation to identify good open facilities
    try:
        # LP relaxation of CFLP
        # Variables: x_{i,j} (assignment), y_j (open)
        n_vars = n_cust * n_fac + n_fac
        
        c = np.zeros(n_vars)
        for i in range(n_cust):
            for j in range(n_fac):
                c[i * n_fac + j] = assign_cost[i, j]
        for j in range(n_fac):
            c[n_cust * n_fac + j] = open_cost[j]
        
        # Constraints:
        # 1. sum_j x_{i,j} = 1 for all i
        # 2. sum_i x_{i,j} * demands[i] <= capacities[j] for all j
        # 3. x_{i,j} <= y_j for all i,j
        
        n_constraints = n_cust + n_fac + n_cust * n_fac
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        A_eq = np.zeros((n_cust, n_vars))
        b_eq = np.ones(n_cust)
        
        row = 0
        # Equality constraints: sum_j x_{i,j} = 1
        for i in range(n_cust):
            for j in range(n_fac):
                A_eq[i, i * n_fac + j] = 1.0
        
        # Capacity constraints
        for j in range(n_fac):
            for i in range(n_cust):
                A_ub[row, i * n_fac + j] = demands[i]
            b_ub[row] = capacities[j]
            row += 1
        
        # Linking: x_{i,j} - y_j <= 0
        for i in range(n_cust):
            for j in range(n_fac):
                A_ub[row, i * n_fac + j] = 1.0
                A_ub[row, n_cust * n_fac + j] = -1.0
                b_ub[row] = 0.0
                row += 1
        
        bounds = [(0, 1)] * n_vars
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method='highs')
        
        if res.status == 0:
            # Extract y values (open facility indicators)
            y_vals = res.x[n_cust * n_fac:]
            
            # Try different thresholds to select open facilities
            for threshold in [0.5, 0.3, 0.7, 0.1, 0.9]:
                if time.time() > deadline:
                    break
                open_facs = [j for j in range(n_fac) if y_vals[j] >= threshold]
                if len(open_facs) == 0:
                    continue
                
                # Solve transportation problem for this set
                sol = _solve_transportation(demands, capacities, assign_cost, open_facs)
                if sol is not None:
                    # Verify validity
                    load = np.zeros(n_fac)
                    for i in range(n_cust):
                        load[sol[i]] += demands[i]
                    valid = True
                    for j in range(n_fac):
                        if load[j] > capacities[j] + 1e-6:
                            valid = False
                            break
                    if valid:
                        cost = compute_cost(sol)
                        if cost < best_cost:
                            best_cost = cost
                            best_result = sol.copy()
                            # Local search from this solution
                            rs, cs = _local_search(sol, demands, capacities, assign_cost, 
                                                    open_cost, max_iters=50, deadline=deadline)
                            if cs < best_cost:
                                best_cost = cs
                                best_result = rs.copy()
    except Exception:
        pass
    
    # Final local search if time permits
    if time.time() < deadline - 5:
        rs, cs = _local_search(best_result, demands, capacities, assign_cost, open_cost,
                                max_iters=200, deadline=deadline)
        if cs < best_cost:
            best_result = rs.copy()
            best_cost = cs
    
    return best_result
# EVOLVE-BLOCK-END
