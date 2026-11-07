# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

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
    
    for _ in range(300):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
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
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=200, deadline=None, rng=None):
    """Local search with single moves and 2-opt swaps."""
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
        
        # 2-opt swaps
        if deadline is not None and time.time() > deadline:
            break
        
        swap_order = rng.permutation(n)
        for i in swap_order:
            if deadline is not None and time.time() > deadline:
                break
            fi = result[i]
            di = demands[i]
            
            for j in range(i+1, n):
                fj = result[j]
                if fj == fi:
                    continue
                dj = demands[j]
                
                if load[fi] - di + dj > capacities[fi] + 1e-6:
                    continue
                if load[fj] - dj + di > capacities[fj] + 1e-6:
                    continue
                
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                
                if load[fi] > 0 and new_load_fi == 0:
                    delta -= open_cost[fi]
                elif load[fi] == 0 and new_load_fi > 0:
                    delta += open_cost[fi]
                
                if load[fj] > 0 and new_load_fj == 0:
                    delta -= open_cost[fj]
                elif load[fj] == 0 and new_load_fj > 0:
                    delta += open_cost[fj]
                
                if delta < -1e-9:
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    current_cost += delta
                    improved = True
                    break
        
        if not improved:
            break
    
    return result, current_cost

def _solve_mip(demands, capacities, open_cost, assign_cost, deadline):
    """Solve the MILP formulation using scipy.optimize.milp."""
    n = len(demands)
    m = len(capacities)
    
    # Variables: x[i, j] for i in 0..n-1, j in 0..m-1 (binary), y[j] for j in 0..m-1 (binary)
    # Total variables: n*m + m
    num_vars = n * m + m
    
    # Objective: sum c_ij x_ij + sum f_j y_j
    c = np.zeros(num_vars)
    for i in range(n):
        for j in range(m):
            c[i * m + j] = assign_cost[i, j]
    for j in range(m):
        c[n * m + j] = open_cost[j]
    
    # Constraints:
    # 1. For each customer i: sum_j x[i,j] = 1  (n constraints)
    # 2. For each facility j: sum_i d_i * x[i,j] <= cap_j  (m constraints)
    # 3. For each facility j: x[i,j] <= y[j] for all i  (n*m constraints)
    
    A_rows = []
    lb = []
    ub = []
    
    # Constraint 1: assignment (each customer assigned to exactly one facility)
    for i in range(n):
        row = np.zeros(num_vars)
        for j in range(m):
            row[i * m + j] = 1.0
        A_rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    
    # Constraint 2: capacity
    for j in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            row[i * m + j] = demands[i]
        A_rows.append(row)
        lb.append(0.0)
        ub.append(capacities[j])
    
    # Constraint 3: x[i,j] <= y[j]  =>  x[i,j] - y[j] <= 0
    for i in range(n):
        for j in range(m):
            row = np.zeros(num_vars)
            row[i * m + j] = 1.0
            row[n * m + j] = -1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
    
    A = np.array(A_rows)
    lb_arr = np.array(lb)
    ub_arr = np.array(ub)
    
    # Variable bounds: all in [0, 1]
    var_bounds = Bounds(0, 1)
    
    # Integrality: x variables binary (0), y variables binary (0)
    integrality = np.zeros(num_vars)
    
    try:
        res = milp(c, constraints=LinearConstraint(A, lb_arr, ub_arr),
                   integrality=integrality, bounds=var_bounds,
                   options={"time_limit": max(1, deadline - time.time())})
        
        if res.status == 0 and res.x is not None:
            x_vals = res.x[:n * m].reshape(n, m)
            y_vals = res.x[n * m:]
            
            # Round x to nearest integer (should be close to 0 or 1)
            result = np.argmax(x_vals, axis=1).astype(int)
            
            # Verify feasibility
            load = np.zeros(m)
            for i in range(n):
                load[result[i]] += demands[i]
            
            if np.all(load <= capacities + 1e-6):
                return result
            else:
                return _repair(result.copy(), demands, capacities, assign_cost)
        else:
            return None
    except Exception:
        return None

def _greedy_init(demands, capacities, assign_cost):
    """Greedy: largest demand first, assign to cheapest feasible facility."""
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

def _cheapest_init(demands, capacities, assign_cost):
    """Assign each customer to cheapest facility, then repair."""
    result = np.argmin(assign_cost, axis=1).astype(int)
    return result

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = np.inf
    
    # Strategy 1: Greedy init + local search
    if time.time() < deadline - 5:
        result = _greedy_init(demands, capacities, assign_cost)
        result = _repair(result, demands, capacities, assign_cost)
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost, 
                                      max_iters=150, deadline=deadline, rng=rng)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Strategy 2: Cheapest init + local search
    if time.time() < deadline - 5:
        result = _cheapest_init(demands, capacities, assign_cost)
        result = _repair(result, demands, capacities, assign_cost)
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost,
                                      max_iters=150, deadline=deadline, rng=rng)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    # Strategy 3: MILP (if time allows)
    if time.time() < deadline - 15 and best_result is not None:
        mip_result = _solve_mip(demands, capacities, open_cost, assign_cost, deadline)
        if mip_result is not None:
            cost = _compute_cost(mip_result, demands, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = mip_result.copy()
                # Polish with local search
                if time.time() < deadline - 3:
                    best_result, best_cost = _local_search(best_result, demands, capacities, 
                                                           open_cost, assign_cost,
                                                           max_iters=100, deadline=deadline, rng=rng)
    
    # Strategy 4: If we have time, do more local search on best result
    if time.time() < deadline - 2 and best_result is not None:
        best_result, best_cost = _local_search(best_result, demands, capacities, open_cost, assign_cost,
                                                max_iters=200, deadline=deadline, rng=rng)
    
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
            best_result = _repair(best_result.copy(), demands, capacities, assign_cost)
    
    if best_result is None:
        # Absolute fallback
        best_result = _greedy_init(demands, capacities, assign_cost)
        best_result = _repair(best_result, demands, capacities, assign_cost)
    
    return best_result
# EVOLVE-BLOCK-END
