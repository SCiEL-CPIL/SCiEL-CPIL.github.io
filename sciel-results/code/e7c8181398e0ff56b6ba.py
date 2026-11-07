# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _compute_cost(result, demands, open_cost, assign_cost):
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open

def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(200):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            # Find customers at f_over, move those with most savings to other facilities
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            # Sort by demand descending
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                # Find cheapest feasible facility (not f_over)
                best_f = -1
                best_cost = np.inf
                for f in range(m):
                    if f == f_over:
                        continue
                    if load[f] + d <= capacities[f] + 1e-6:
                        c = assign_cost[i, f]
                        if c < best_cost:
                            best_cost = c
                            best_f = f
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    # Fallback
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=100, deadline=None, rng=None):
    """Local search with single moves."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n)
        
        # Single moves
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if not improved:
            break
    
    return result, current_cost

def _solve_mip(demands, capacities, open_cost, assign_cost, deadline):
    """Solve the CFLP using MIP."""
    n = len(demands)
    m = len(capacities)
    
    # Variables: x_ij (n*m) + y_j (m)
    # Total vars: n*m + m
    num_vars = n * m + m
    
    # Objective coefficients
    # Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i,j] * x[i,j])
    c = np.zeros(num_vars)
    # x_ij indices: i*m + j
    # y_j indices: n*m + j
    for j in range(m):
        c[n*m + j] = open_cost[j]
    for i in range(n):
        for j in range(m):
            c[i*m + j] = assign_cost[i, j]
            
    # Constraints
    # 1. Assignment: sum_j x_ij = 1 for each i
    # 2. Capacity: sum_i demands[i] * x_ij - capacities[j] * y_j <= 0 for each j
    
    # Number of constraints: n + m
    num_constraints = n + m
    
    A = np.zeros((num_constraints, num_vars))
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    # Assignment constraints (equalities)
    for i in range(n):
        row = i
        for j in range(m):
            A[row, i*m + j] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        
    # Capacity constraints (inequalities)
    for j in range(m):
        row = n + j
        for i in range(n):
            A[row, i*m + j] = demands[i]
        A[row, n*m + j] = -capacities[j]
        lb[row] = -np.inf
        ub[row] = 0.0
        
    # Bounds
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    
    # Integrality: all variables are integer
    integrality = np.ones(num_vars)
    
    # Time limit in seconds
    time_limit = deadline - time.time() - 5.0 # 5s safety margin
    if time_limit <= 0:
        return None
        
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality, options={"time_limit": time_limit})
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            x = res.x[:n*m].reshape(n, m)
            # x should be close to 0 or 1
            # Assign each customer to the facility with x_ij closest to 1
            result = np.argmax(x, axis=1).astype(int)
            
            # Verify feasibility
            load = np.zeros(m)
            for i in range(n):
                load[result[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                # If MIP returned slightly infeasible due to numerical issues, repair it
                result = _repair(result, demands, capacities, assign_cost)
                return result
        else:
            return None
    except Exception:
        return None

def _greedy_init(demands, capacities, assign_cost):
    """Greedy initialization: largest demand first."""
    n = len(demands)
    m = len(capacities)
    order = np.argsort(-demands)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    for i in order:
        d = demands[i]
        best_f = -1
        best_c = np.inf
        for f in range(m):
            if load[f] + d <= capacities[f] + 1e-6:
                c = assign_cost[i, f]
                if c < best_c:
                    best_c = c
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            rem = capacities - load
            best_f = int(np.argmax(rem))
            result[i] = best_f
            load[best_f] += d
    return result

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = np.inf
    
    # 1. Try MIP Solver
    mip_result = _solve_mip(demands, capacities, open_cost, assign_cost, deadline)
    if mip_result is not None:
        cost = _compute_cost(mip_result, demands, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = mip_result.copy()
            
        # Polish with local search if time permits
        if time.time() < deadline - 10:
            polished, pol_cost = _local_search(mip_result, demands, capacities, open_cost, assign_cost, max_iters=50, deadline=deadline, rng=rng)
            if pol_cost < best_cost:
                best_cost = pol_cost
                best_result = polished.copy()
                
    # 2. Fallback to Heuristics if MIP failed or didn't find a good solution
    if best_result is None or time.time() < deadline - 20:
        # Greedy Init
        result = _greedy_init(demands, capacities, assign_cost)
        result = _repair(result, demands, capacities, assign_cost)
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=100, deadline=deadline, rng=rng)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
            
        # Cheapest Init
        if time.time() < deadline - 10:
            result = np.argmin(assign_cost, axis=1).astype(int)
            result = _repair(result, demands, capacities, assign_cost)
            result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=100, deadline=deadline, rng=rng)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
                
    # Final validity check
    if best_result is not None:
        load = np.zeros(m)
        valid = True
        for i in range(n):
            f = best_result[i]
            if f < 0 or f >= m:
                valid = False
                break
            load[f] += demands[i]
            
        if not valid or np.any(load > capacities + 1e-6):
            # Fallback repair
            best_result = _repair(best_result, demands, capacities, assign_cost)
            
    if best_result is None:
        # Absolute last resort: assign everyone to facility 0 and repair (might fail but better than nothing)
        best_result = np.zeros(n, dtype=int)
        best_result = _repair(best_result, demands, capacities, assign_cost)
        
    return best_result
# EVOLVE-BLOCK-END
